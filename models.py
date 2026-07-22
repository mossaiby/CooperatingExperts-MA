"""
FrozenExpert: wraps one pretrained HF causal-LM as a frozen "expert" with:
  - a frozen backbone (4-bit if configured)
  - new <switch:*> special tokens, routed through small standalone
    [num_new, hidden] parameters patched into the embedding lookup and
    output head, so the pretrained vocab's ~100k+ rows are never touched
    and never carry optimizer state
  - to_shared / from_shared bridge projections (always trainable)
  - helpers to extract a handoff hidden vector from a chosen layer, and to
    run the backbone with a hidden vector injected as a virtual position 0

This is the "swap the from-scratch Expert for a pretrained one" piece of
the plan -- everything downstream (stitching loss, mixed loss, generation)
talks to this class the same way it talked to the old from-scratch Expert.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import ExpertSpec, SharedSpaceConfig, SwitchTokenConfig, HandoffLayerConfig


def _resolve_layer_index(num_layers: int, handoff_cfg: HandoffLayerConfig) -> int:
    if handoff_cfg.layer_index is not None:
        idx = handoff_cfg.layer_index
        if idx < 0:
            idx = num_layers + idx
        return max(0, min(idx, num_layers))
    return max(0, min(round(num_layers * handoff_cfg.layer_fraction), num_layers))


class _PatchedEmbedding(nn.Module):
    """
    Wraps a frozen nn.Embedding. Lookups for token ids below `new_start_idx`
    go through the frozen weight as normal (no grad). Lookups for ids >=
    new_start_idx (the newly added <switch:*> tokens, always appended at
    the END of the vocab by resize_token_embeddings) are replaced with rows
    from a small trainable parameter instead. This way autograd only ever
    tracks gradients for a [num_new, hidden] tensor, not the full vocab --
    replaces the old "full matrix trainable + gradient-masking hook"
    approach, which wasted optimizer memory/time on ~130k unused rows.
    """
    def __init__(self, frozen_embedding: nn.Embedding, new_start_idx: int, num_new: int):
        super().__init__()
        self.frozen = frozen_embedding
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        self.new_start_idx = new_start_idx
        # init the new rows from whatever resize_token_embeddings already
        # put there (mean/cov init), just detached into a small trainable tensor
        init_rows = self.frozen.weight[new_start_idx:new_start_idx + num_new].detach().clone()
        self.new_token_embed = nn.Parameter(init_rows)

    def forward(self, input_ids):
        base = torch.nn.functional.embedding(input_ids, self.frozen.weight)
        mask = input_ids >= self.new_start_idx
        if mask.any():
            local_idx = (input_ids[mask] - self.new_start_idx).long()
            out = base.clone()
            out[mask] = self.new_token_embed[local_idx].to(out.dtype)
            return out
        return base


class _PatchedLMHead(nn.Module):
    """
    Same idea for the output projection. Frozen weight computes logits for
    the original vocab (including the new tokens' garbage placeholder rows,
    which are simply never used); a small trainable [num_new, hidden]
    parameter supplies the logits for the new tokens instead, overwriting
    just the last num_new columns.
    """
    def __init__(self, frozen_head: nn.Linear, new_start_idx: int, num_new: int):
        super().__init__()
        self.frozen = frozen_head
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        self.new_start_idx = new_start_idx
        self.num_new = num_new
        init_rows = self.frozen.weight[new_start_idx:new_start_idx + num_new].detach().clone()
        self.new_token_head = nn.Parameter(init_rows)  # [num_new, hidden]

    def forward(self, hidden_states):
        base_logits = torch.nn.functional.linear(hidden_states, self.frozen.weight, self.frozen.bias)
        new_logits = torch.nn.functional.linear(
            hidden_states, self.new_token_head.to(hidden_states.dtype)
        )
        out = base_logits.clone()
        out[..., self.new_start_idx:self.new_start_idx + self.num_new] = new_logits
        return out


class FrozenExpert(nn.Module):
    def __init__(
        self,
        spec: ExpertSpec,
        shared_cfg: SharedSpaceConfig,
        switch_cfg: SwitchTokenConfig,
        handoff_cfg: HandoffLayerConfig,
        device: str = "cuda",
    ):
        super().__init__()
        self.spec = spec
        self.device = device
        self.switch_cfg = switch_cfg

        print(f"[{spec.name}] loading {spec.hf_model_id} "
              f"(4bit={spec.load_in_4bit}) ...")

        quant_kwargs = {}
        if spec.load_in_4bit:
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            quant_kwargs["torch_dtype"] = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_model_id, trust_remote_code=spec.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModelForCausalLM.from_pretrained(
            spec.hf_model_id,
            trust_remote_code=spec.trust_remote_code,
            device_map={"": 0} if device == "cuda" else None,
            **quant_kwargs,
        )
        if device != "cuda" and not spec.load_in_4bit:
            self.backbone.to(device)

        # ---- freeze everything first ----
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # ---- add special tokens, resize, patch in small trainable params ----
        n_before = len(self.tokenizer)
        new_tokens = switch_cfg.token_strings()
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens}
        )
        self.backbone.resize_token_embeddings(len(self.tokenizer))
        self.new_token_ids = list(range(n_before, n_before + num_added))
        self._patch_new_token_params(n_before, num_added)

        self.hidden_size = self.backbone.config.hidden_size
        n_layers = getattr(self.backbone.config, "num_hidden_layers",
                            getattr(self.backbone.config, "n_layer", None))
        self.handoff_layer_index = _resolve_layer_index(n_layers, handoff_cfg)
        print(f"[{spec.name}] hidden_size={self.hidden_size}, "
              f"n_layers={n_layers}, handoff_layer_index={self.handoff_layer_index}, "
              f"added {num_added} special tokens: {new_tokens}")

        # ---- bridge projections (always trainable, always full precision) ----
        self.to_shared = nn.Linear(self.hidden_size, shared_cfg.dim, bias=False)
        self.from_shared = nn.Linear(shared_cfg.dim, self.hidden_size, bias=False)
        self.to_shared.to(device=device, dtype=torch.float32)
        self.from_shared.to(device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Special-token params (small, standalone -- NOT the full vocab matrix)
    # ------------------------------------------------------------------
    def _patch_new_token_params(self, new_start_idx: int, num_added: int):
        """
        Swaps the backbone's input embedding and output head for wrapper
        modules that keep the pretrained vocab fully frozen and route only
        the new <switch:*> tokens through small standalone [num_added,
        hidden] parameters. Registered as nn.Module attributes (self.
        patched_embedding / self.patched_head) so their params show up
        automatically via self.parameters() if ever needed, and so they
        survive .to(device) calls on this FrozenExpert.

        This replaces the old approach of keeping the FULL embedding/head
        matrices "trainable" behind a gradient-zeroing hook -- that wasted
        AdamW optimizer memory and per-step update time on ~130k unused
        rows. Now there are only a few hundred to a couple thousand actual
        trainable scalars per expert for the special tokens.
        """
        orig_emb = self.backbone.get_input_embeddings()
        patched_emb = _PatchedEmbedding(orig_emb, new_start_idx, num_added)
        patched_emb.to(device=orig_emb.weight.device, dtype=orig_emb.weight.dtype)
        self.backbone.set_input_embeddings(patched_emb)
        self.patched_embedding = patched_emb

        out_emb = self.backbone.get_output_embeddings()
        self.patched_head = None
        if out_emb is not None and isinstance(out_emb, nn.Linear):
            patched_head = _PatchedLMHead(out_emb, new_start_idx, num_added)
            patched_head.to(device=out_emb.weight.device, dtype=out_emb.weight.dtype)
            self.backbone.set_output_embeddings(patched_head)
            self.patched_head = patched_head
        elif out_emb is not None:
            print(f"[{self.spec.name}] WARNING: output embeddings module is "
                  f"{type(out_emb)}, not nn.Linear -- couldn't patch the output "
                  f"head for new tokens. New-token logits will fall back to "
                  f"whatever resize_token_embeddings initialized, and won't "
                  f"be trainable. Generation should still work for non-switch "
                  f"tokens; investigate if switch tokens never get predicted.")

    def trainable_parameters(self):
        params = [self.to_shared.weight, self.from_shared.weight,
                  self.patched_embedding.new_token_embed]
        if self.patched_head is not None:
            params.append(self.patched_head.new_token_head)
        return params

    def switch_id(self, target_name: str) -> int:
        tok = f"<switch:{target_name}>"
        return self.tokenizer.convert_tokens_to_ids(tok)

    @property
    def self_switch_id(self):
        return self.tokenizer.convert_tokens_to_ids("<switch:self>")

    # ------------------------------------------------------------------
    # Core forward helpers
    # ------------------------------------------------------------------
    def encode_handoff_vector(self, input_ids, attention_mask=None):
        """
        Run the backbone on input_ids, return the hidden state at
        self.handoff_layer_index for the LAST real token of each sequence.
        Shape: [B, hidden_size], dtype float32.

        This is the original single-vector path (num_vectors=1 case).
        See encode_handoff_vectors() for the multi-vector variant.
        """
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hs = out.hidden_states[self.handoff_layer_index]  # [B, T, H]
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1  # [B]
        else:
            last_idx = torch.full((hs.shape[0],), hs.shape[1] - 1, device=hs.device)
        batch_idx = torch.arange(hs.shape[0], device=hs.device)
        last_hidden = hs[batch_idx, last_idx.long()]  # [B, H]
        return last_hidden.float()

    def encode_handoff_vectors(self, input_ids, attention_mask, num_vectors):
        """
        Multi-vector variant: instead of pooling the whole segment down to
        ONE summary vector, keep the hidden states of the last `num_vectors`
        REAL (non-padding) tokens at self.handoff_layer_index, per sequence.
        Shape: [B, num_vectors, hidden_size], dtype float32.

        Rationale: a single vector discards positional/sequential structure
        (e.g. "the recursive call is factorial(n-1), in that order" vs. just
        "this is about factorial and recursion"). Keeping a short SEQUENCE
        of vectors instead of one lets from_shared_space's injected prefix
        carry some of that structure into the target expert, at the cost of
        a slightly larger prefix (num_vectors positions instead of 1).

        If a sequence has fewer than num_vectors real tokens, the earliest
        slots are padded by repeating the first real token's hidden state
        (rather than zeros) so the target always sees num_vectors
        "meaningful" positions, not an artificial zero-vector signal that
        the target has never seen during its own pretraining.
        """
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hs = out.hidden_states[self.handoff_layer_index]  # [B, T, H]
        B, T, H = hs.shape
        device = hs.device

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long()  # [B], real token count per seq
        else:
            lengths = torch.full((B,), T, device=device, dtype=torch.long)

        out_vecs = torch.zeros(B, num_vectors, H, device=device, dtype=hs.dtype)
        for b in range(B):
            L = max(int(lengths[b].item()), 1)
            take = min(num_vectors, L)
            # last `take` real tokens, in order
            selected = hs[b, L - take:L, :]  # [take, H]
            if take < num_vectors:
                # pad earliest slots by repeating the first available token,
                # so every batch row has exactly num_vectors positions
                pad = selected[0:1, :].expand(num_vectors - take, H)
                selected = torch.cat([pad, selected], dim=0)
            out_vecs[b] = selected

        return out_vecs.float()  # [B, num_vectors, H]

    def to_shared_space(self, h):
        return self.to_shared(h.float())

    def from_shared_space(self, z):
        return self.from_shared(z.float()).to(self.backbone.dtype)

    def forward_with_injected_prefix(self, injected_vec, input_ids, attention_mask=None,
                                      labels=None):
        """
        Prepend `injected_vec` as a virtual prefix via inputs_embeds, then
        run the normal token embeddings for input_ids, concatenated after
        it. Returns logits over input_ids positions only.

        injected_vec can be EITHER:
          - [B, H]      (single-vector handoff, num_vectors=1) -- treated
                        as one virtual position, matching the original
                        behavior exactly.
          - [B, K, H]   (multi-vector handoff, num_vectors=K) -- treated as
                        K virtual positions.
        """
        emb_layer = self.backbone.get_input_embeddings()
        tok_embeds = emb_layer(input_ids)  # [B, T, H]

        if injected_vec.dim() == 2:
            injected = injected_vec.unsqueeze(1).to(tok_embeds.dtype)  # [B, 1, H]
        else:
            injected = injected_vec.to(tok_embeds.dtype)  # [B, K, H]
        n_injected = injected.shape[1]

        full_embeds = torch.cat([injected, tok_embeds], dim=1)  # [B, K+T, H]

        if attention_mask is not None:
            extra = torch.ones(attention_mask.shape[0], 1,
                                device=attention_mask.device, dtype=attention_mask.dtype)
            full_mask = torch.cat([extra, attention_mask], dim=1)
        else:
            full_mask = None

        out = self.backbone(inputs_embeds=full_embeds, attention_mask=full_mask)
        # full_embeds = [inj_0, ..., inj_{K-1}, tok_0, tok_1, ..., tok_{T-1}]
        # (K+T positions, K = n_injected). out.logits[:, i, :] predicts the
        # token that comes AFTER position i. input_ids[:, j] sits at
        # position K+j in full_embeds, so its prediction comes from
        # out.logits[:, K+j-1, :]. Sliding that over j=0..T-1 gives the
        # window out.logits[:, K-1 : K-1+T, :] == out.logits[:, K-1:-1, :],
        # which lines up 1:1 with input_ids with no further shift needed.
        # (K=1 reduces to the original out.logits[:, :-1, :] behavior.)
        logits = out.logits[:, n_injected - 1:-1, :]
        return logits

    def gradient_checkpointing_enable(self):
        self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()
