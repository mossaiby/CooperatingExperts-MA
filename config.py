"""
Single source of truth for the pretrained-experts pipeline.

This is the v2 of the "Cooperating Experts" project: instead of training two
small transformers from scratch, we freeze two *pretrained* HF models with
genuinely different tokenizers, and only train small bridge parameters
(projections + a handful of new special-token embedding rows, optionally
+ LoRA in phase 3).
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExpertSpec:
    name: str                      # "english" / "python"
    hf_model_id: str               # HF hub id
    role: str                      # free text, just for logging
    load_in_4bit: bool = True
    trust_remote_code: bool = False


@dataclass
class ModelPairConfig:
    """
    Which two pretrained models play the two experts.

    IMPORTANT: pick models from DIFFERENT labs with independently-trained
    tokenizers. Same-lab pairs (e.g. SmolLM2 + StarCoder2, or Qwen2.5 +
    Qwen2.5-Coder) often secretly share a tokenizer or its lineage, which
    defeats the "genuinely different vocabulary" premise of this project
    even when the models themselves are fine-tuned for different domains.
    ALWAYS run check_tokenizer_overlap.py after changing this and confirm
    low overlap before training anything.
    """
    english: ExpertSpec = field(default_factory=lambda: ExpertSpec(
        name="english",
        hf_model_id="meta-llama/Llama-3.2-1B",
        role="general / English prose",
    ))
    python: ExpertSpec = field(default_factory=lambda: ExpertSpec(
        name="python",
        hf_model_id="deepseek-ai/deepseek-coder-1.3b-base",
        role="Python code",
    ))
    # Non-gated fallback for the english slot if Llama-3.2 access isn't
    # approved on your HF account yet:
    #   hf_model_id="google/gemma-2-2b"   (SentencePiece, 256k vocab)


@dataclass
class DebugModelPairConfig:
    """
    Tiny stand-in pair for laptop debugging (4GB VRAM or CPU).
    Two *different* small models so the "different tokenizer" property still
    holds even in debug mode -- catches shape/logic bugs cheaply before
    burning Colab time on the real 1-3B pair.
    """
    english: ExpertSpec = field(default_factory=lambda: ExpertSpec(
        name="english",
        hf_model_id="distilgpt2",
        role="tiny English stand-in",
        load_in_4bit=False,   # too small to bother quantizing
    ))
    python: ExpertSpec = field(default_factory=lambda: ExpertSpec(
        name="python",
        hf_model_id="Salesforce/codegen-350M-mono",
        role="tiny Python stand-in",
        load_in_4bit=False,
    ))


@dataclass
class SharedSpaceConfig:
    dim: int = 512          # bottleneck dim; bump vs. the from-scratch v1 (256)
                             # since pretrained hidden spaces are larger/entangled


@dataclass
class SwitchTokenConfig:
    # one <switch:NAME> per expert, plus <switch:self> as a no-op
    experts: tuple = ("english", "python")
    include_self: bool = True

    def token_strings(self):
        toks = [f"<switch:{e}>" for e in self.experts]
        if self.include_self:
            toks.append("<switch:self>")
        return toks


@dataclass
class HandoffLayerConfig:
    # Which hidden layer to pull the handoff representation from.
    # None -> last layer. CALM found middle layers often connect better
    # than the final layer; worth sweeping this once the pipeline runs.
    layer_index: Optional[int] = None
    # fraction-of-depth fallback if layer_index is None and you want
    # "middle layer" behaviour: 0.5 = halfway through the stack
    layer_fraction: float = 0.5


@dataclass
class DataConfig:
    codesearchnet_config: str = "python"
    n_pairs: int = 20000
    min_docstring_words: int = 4
    min_function_lines: int = 2
    max_function_lines: int = 80     # keep sequences short enough for 16GB
    val_fraction: float = 0.1
    seed: int = 42
    cache_dir: str = "data/csn_cache"
    processed_path: str = "data/handoff_pairs.jsonl"


@dataclass
class StitchTrainConfig:
    """Phase 2: frozen backbones, train only bridge params."""
    max_seq_len: int = 256
    batch_size: int = 2
    grad_accum: int = 8            # effective batch 16
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    steps_max: int = 3000
    align_weight: float = 0.1
    val_every: int = 100
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-3
    grad_clip: float = 1.0
    grad_checkpointing: bool = True
    log_every: int = 20
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 200          # frequent, Colab disconnects


@dataclass
class LoraTrainConfig:
    """Phase 3: LoRA adapters on frozen backbones + bridge fine-tune."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")  # override per-arch if needed
    lr: float = 1e-4
    grad_accum: int = 16
    steps_max: int = 2000
    warmup_steps: int = 100
    grad_clip: float = 0.5
    switch_loss_weight: float = 0.1
    log_every: int = 20
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 200


@dataclass
class GenConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    max_switches: int = 4


@dataclass
class Config:
    pair: ModelPairConfig = field(default_factory=ModelPairConfig)
    shared: SharedSpaceConfig = field(default_factory=SharedSpaceConfig)
    switch: SwitchTokenConfig = field(default_factory=SwitchTokenConfig)
    handoff_layer: HandoffLayerConfig = field(default_factory=HandoffLayerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stitch: StitchTrainConfig = field(default_factory=StitchTrainConfig)
    lora: LoraTrainConfig = field(default_factory=LoraTrainConfig)
    gen: GenConfig = field(default_factory=GenConfig)

    @staticmethod
    def default():
        return Config()

    @staticmethod
    def debug():
        """Tiny-model config for laptop dev (see DebugModelPairConfig)."""
        cfg = Config()
        cfg.pair = DebugModelPairConfig()
        cfg.shared.dim = 128
        cfg.data.n_pairs = 200
        cfg.stitch.batch_size = 2
        cfg.stitch.grad_accum = 2
        cfg.stitch.steps_max = 60
        cfg.stitch.val_every = 10
        cfg.stitch.ckpt_every = 20
        cfg.stitch.max_seq_len = 64
        cfg.lora.steps_max = 40
        cfg.lora.grad_accum = 2
        return cfg
