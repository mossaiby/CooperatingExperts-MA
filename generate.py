"""
Generation with live switching between experts.

HF's .generate() can't directly accept "prepend one virtual embedding, then
continue normally with input_ids + KV cache" -- so this is a manual loop:
  1. Seed step: forward with inputs_embeds = [injected_vec?, prompt_embeds],
     use_cache=True -> get a KV cache seeded with the (optional) handoff
     vector and the prompt.
  2. Continue autoregressively with plain input_ids + past_key_values,
     sampling one token at a time.
  3. If the sampled token is a <switch:NAME> token belonging to a DIFFERENT
     expert than the currently active one, stop this expert, encode its
     hidden state as a handoff vector, and seed the target expert's own KV
     cache the same way, then keep generating with that expert active.

This is the piece that changes the most vs. v1's from-scratch Expert.forward,
since HF models don't expose the same low-level hooks your own nn.Module did.
"""
import torch
import torch.nn.functional as F

from config import Config
from models import FrozenExpert
from train_stitch import load_bridge_checkpoint


def sample_next_token(logits, temperature, top_k, generated_ids=None,
                       repetition_penalty=1.3, no_repeat_ngram_size=3):
    """
    Standard decoding-quality guards on top of temperature/top-k sampling --
    small models (1-3B) are prone to falling into repetition loops on longer
    generations without these.

    repetition_penalty: divides (if logit>0) or multiplies (if logit<=0) the
        logits of any token already seen in generated_ids, discouraging
        the model from just repeating itself. Standard trick from Keskar
        et al. 2019 / used by HF's own generate().
    no_repeat_ngram_size: if the last (n-1) tokens plus a candidate token
        would recreate an n-gram that's already appeared in generated_ids,
        that candidate is banned outright (hard -inf), not just penalized --
        this is what actually breaks exact-repeat loops like the
        factorial/fibo cycling seen in practice, since a soft penalty alone
        often isn't enough to stop a model from re-entering a loop it's
        already committed to.
    """
    logits = logits.clone()

    if generated_ids is not None and len(generated_ids) > 0 and repetition_penalty != 1.0:
        seen = set(generated_ids)
        for tok_id in seen:
            if logits[tok_id] > 0:
                logits[tok_id] /= repetition_penalty
            else:
                logits[tok_id] *= repetition_penalty

    if generated_ids is not None and no_repeat_ngram_size > 0 and \
            len(generated_ids) >= no_repeat_ngram_size - 1:
        n = no_repeat_ngram_size
        prefix = tuple(generated_ids[-(n - 1):]) if n > 1 else tuple()
        seen_ngrams = set()
        for i in range(len(generated_ids) - n + 1):
            seen_ngrams.add(tuple(generated_ids[i:i + n]))
        banned_next = {ng[-1] for ng in seen_ngrams if ng[:-1] == prefix}
        for tok_id in banned_next:
            logits[tok_id] = float("-inf")

    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        top_vals, top_idx = torch.topk(logits, top_k)
        probs = F.softmax(top_vals, dim=-1)
        choice = torch.multinomial(probs, 1)
        return top_idx.gather(-1, choice).squeeze(-1)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def _seed_kv_cache(expert: FrozenExpert, prompt_ids, injected_vec=None):
    """
    Returns (past_key_values, last_logits) after a forward pass over the
    prompt (optionally prefixed by an injected handoff vector via
    inputs_embeds). last_logits is the distribution for the NEXT token
    after the prompt.
    """
    emb_layer = expert.backbone.get_input_embeddings()
    tok_embeds = emb_layer(prompt_ids)  # [1, T, H]
    if injected_vec is not None:
        injected = injected_vec.unsqueeze(1).to(tok_embeds.dtype)
        full_embeds = torch.cat([injected, tok_embeds], dim=1)
    else:
        full_embeds = tok_embeds

    out = expert.backbone(inputs_embeds=full_embeds, use_cache=True)
    last_logits = out.logits[:, -1, :]
    return out.past_key_values, last_logits


def _step_with_cache(expert: FrozenExpert, token_id, past_key_values):
    ids = token_id.unsqueeze(-1)  # [1, 1]
    out = expert.backbone(input_ids=ids, past_key_values=past_key_values, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


@torch.no_grad()
def generate(experts, prompt_text, start_expert, cfg, device="cuda"):
    gen_cfg = cfg.gen
    active_name = start_expert
    active = experts[active_name]

    prompt_ids = torch.tensor(
        [active.tokenizer(prompt_text)["input_ids"]], device=device
    )
    past_kv, logits = _seed_kv_cache(active, prompt_ids, injected_vec=None)

    output_pieces = [(active_name, prompt_text)]
    current_piece_ids = []
    active_context_ids = list(prompt_ids[0].tolist())  # full context since this expert became active
    n_switches = 0

    for _ in range(gen_cfg.max_new_tokens):
        next_id = sample_next_token(logits[0], gen_cfg.temperature, gen_cfg.top_k,
                                     generated_ids=active_context_ids,
                                     repetition_penalty=gen_cfg.repetition_penalty,
                                     no_repeat_ngram_size=gen_cfg.no_repeat_ngram_size)
        next_id_batched = next_id.unsqueeze(0)

        # is this a switch token, and does it name a DIFFERENT expert?
        decoded_tok = active.tokenizer.convert_ids_to_tokens([next_id.item()])[0]
        target_name = None
        if decoded_tok.startswith("<switch:") and decoded_tok.endswith(">"):
            candidate = decoded_tok[len("<switch:"):-1]
            if candidate in experts and candidate != active_name:
                target_name = candidate

        if target_name is not None and n_switches < gen_cfg.max_switches:
            # flush current piece text
            if current_piece_ids:
                text = active.tokenizer.decode(current_piece_ids, skip_special_tokens=True)
                output_pieces.append((active_name, text))
                current_piece_ids = []

            # compute handoff vector from the ACTIVE expert's full context
            # since it became active (not just tokens since the last flush)
            context_ids = torch.tensor([active_context_ids], device=device)
            h = active.encode_handoff_vector(context_ids, torch.ones_like(context_ids))
            z = active.to_shared_space(h)

            target = experts[target_name]
            h0 = target.from_shared_space(z)

            # seed target's cache with the handoff vector, anchored on a
            # single real token (eos) since zero-length inputs_embeds isn't
            # reliably supported across model architectures
            anchor_id = torch.tensor([[target.tokenizer.eos_token_id]], device=device)
            past_kv, logits = _seed_kv_cache(target, anchor_id, injected_vec=h0)

            active_name = target_name
            active = target
            active_context_ids = [target.tokenizer.eos_token_id]
            n_switches += 1
            continue

        current_piece_ids.append(next_id.item())
        active_context_ids.append(next_id.item())
        if next_id.item() == active.tokenizer.eos_token_id:
            break
        past_kv, logits = _step_with_cache(active, next_id_batched, past_kv)

    if current_piece_ids:
        text = active.tokenizer.decode(current_piece_ids, skip_special_tokens=True)
        output_pieces.append((active_name, text))

    return output_pieces


def load_experts_for_generation(cfg: Config, bridge_ckpt_path: str, device="cuda"):
    experts = {
        "english": FrozenExpert(cfg.pair.english, cfg.shared, cfg.switch, cfg.handoff_layer, device),
        "python": FrozenExpert(cfg.pair.python, cfg.shared, cfg.switch, cfg.handoff_layer, device),
    }
    load_bridge_checkpoint(experts, bridge_ckpt_path)
    for e in experts.values():
        e.backbone.eval()
    return experts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", type=str)
    ap.add_argument("--expert", choices=["english", "python"], default="english")
    ap.add_argument("--bridge-ckpt", required=True)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = Config.debug() if args.debug else Config.default()
    experts = load_experts_for_generation(cfg, args.bridge_ckpt, device=args.device)
    pieces = generate(experts, args.prompt, args.expert, cfg, device=args.device)

    print("\n=== Generated (with switches) ===")
    for name, text in pieces:
        print(f"\n[{name}]\n{text}")
