"""
FrozenExpert: wraps one pretrained HF causal-LM as a frozen "expert" with:
  - a frozen backbone (4-bit if configured)
  - new <switch:*> special tokens, with a gradient mask so only the NEW
    embedding/head rows are trainable (not the whole 100k+-row vocab)
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

        # ---- add special tokens, resize, unfreeze only the new rows ----
        n_before = len(self.tokenizer)
        new_tokens = switch_cfg.token_strings()
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens}
        )
        self.backbone.resize_token_embeddings(len(self.tokenizer))
        self.new_token_ids = list(range(n_before, n_before + num_added))
        self._register_new_token_grad_mask()

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
    # Special-token gradient masking
    # ------------------------------------------------------------------
    def _register_new_token_grad_mask(self):
        """
        Unfreeze the embedding + output-head weight matrices, but register a
        backward hook that zeros the gradient for every row EXCEPT the newly
        added special-token rows. This lets those new rows learn while never
        actually updating the pretrained vocabulary's embeddings.
        """
        emb = self.backbone.get_input_embeddings()
        emb.weight.requires_grad_(True)

        new_ids = set(self.new_token_ids)
        vocab_size = emb.weight.shape[0]
        mask = torch.zeros(vocab_size, 1, device=emb.weight.device, dtype=emb.weight.dtype)
        for i in new_ids:
            mask[i, 0] = 1.0

        def _mask_grad(grad):
            return grad * mask.to(grad.dtype)
        emb.weight.register_hook(_mask_grad)

        out_emb = self.backbone.get_output_embeddings()
        if out_emb is not None and out_emb.weight is not emb.weight:
            out_emb.weight.requires_grad_(True)
            out_emb.weight.register_hook(_mask_grad)
        # if tied, the single hook above already covers both

    def trainable_parameters(self):
        params = [self.to_shared.weight, self.from_shared.weight]
        emb = self.backbone.get_input_embeddings()
        params.append(emb.weight)
        out_emb = self.backbone.get_output_embeddings()
        if out_emb is not None and out_emb.weight is not emb.weight:
            params.append(out_emb.weight)
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

    def to_shared_space(self, h):
        return self.to_shared(h.float())

    def from_shared_space(self, z):
        return self.from_shared(z.float()).to(self.backbone.dtype)

    def forward_with_injected_prefix(self, injected_vec, input_ids, attention_mask=None,
                                      labels=None):
        """
        Prepend `injected_vec` ([B, H]) as a virtual position 0 via
        inputs_embeds, then run the normal token embeddings for input_ids,
        concatenated after it. Returns logits over input_ids positions only
        (i.e. the prepended position is stripped from the loss/logits, it
        only serves to condition the rest of the forward pass).
        """
        emb_layer = self.backbone.get_input_embeddings()
        tok_embeds = emb_layer(input_ids)  # [B, T, H]
        injected = injected_vec.unsqueeze(1).to(tok_embeds.dtype)  # [B, 1, H]
        full_embeds = torch.cat([injected, tok_embeds], dim=1)  # [B, T+1, H]

        if attention_mask is not None:
            extra = torch.ones(attention_mask.shape[0], 1,
                                device=attention_mask.device, dtype=attention_mask.dtype)
            full_mask = torch.cat([extra, attention_mask], dim=1)
        else:
            full_mask = None

        out = self.backbone(inputs_embeds=full_embeds, attention_mask=full_mask)
        # full_embeds = [injected, tok_0, tok_1, ..., tok_{T-1}]  (T+1 positions)
        # out.logits[:, i, :] is the model's prediction for the token that
        # comes AFTER position i. So:
        #   out.logits[:, 0, :] predicts tok_0        == input_ids[:, 0]
        #   out.logits[:, 1, :] predicts tok_1        == input_ids[:, 1]
        #   ...
        #   out.logits[:, T-1, :] predicts tok_{T-1}  == input_ids[:, T-1]
        # i.e. dropping the LAST position (which predicts one step beyond the
        # sequence) lines logits up 1:1 with input_ids, no further shift
        # needed by the caller.
        logits = out.logits[:, :-1, :]
        return logits

    def gradient_checkpointing_enable(self):
        self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()
