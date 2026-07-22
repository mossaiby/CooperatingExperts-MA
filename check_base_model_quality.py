"""
Baseline check we should have run earlier: how good is each backbone at
these prompts with ZERO bridge/handoff involvement -- no injected vector,
no switch tokens, just the raw pretrained model completing the prompt the
normal way. This separates two very different explanations for the
incoherent generations seen so far:
  (a) the bridge/handoff is interfering with otherwise-good generation
  (b) the base model itself struggles with these prompts regardless

Usage:
    python check_base_model_quality.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from generate import sample_next_token


PROMPTS = [
    ("python", "def factorial(n):"),
    ("python", "def is_palindrome(s):"),
    ("english", "This function calculates the factorial of a number recursively."),
]


@torch.no_grad()
def generate_plain(model, tokenizer, prompt, max_new_tokens=60, temperature=0.8,
                    top_k=50, device="cuda"):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = ids[0].tolist()
    past_kv = None
    cur_ids = ids
    for _ in range(max_new_tokens):
        out = model(input_ids=cur_ids, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :]
        next_id = sample_next_token(logits[0], temperature, top_k, generated_ids=generated)
        generated.append(next_id.item())
        if next_id.item() == tokenizer.eos_token_id:
            break
        cur_ids = next_id.unsqueeze(0).unsqueeze(0)
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    cfg = Config.default()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for role, spec in (("english", cfg.pair.english), ("python", cfg.pair.python)):
        print(f"\n{'='*70}\nRAW BASE MODEL: {role} ({spec.hf_model_id}), NO bridge\n{'='*70}")
        tok = AutoTokenizer.from_pretrained(spec.hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_model_id, torch_dtype="auto", device_map={"": 0} if device == "cuda" else None,
        )
        model.eval()

        for prompt_role, prompt in PROMPTS:
            torch.manual_seed(0)
            text = generate_plain(model, tok, prompt, device=device)
            print(f"\n--- prompt (natural fit={'yes' if prompt_role == role else 'no'}): {prompt!r} ---")
            print(f"  {text}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
