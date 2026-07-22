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
    # Swapped from openbmb/MiniCPM5-1B once Llama-3.2-1B access was granted.
    # MiniCPM5-1B is heavily post-trained for code/agentic tasks, which
    # blurred the "English vs Python expert" domain separation the whole
    # project is testing -- the English expert already knew a lot of Python
    # going in. Llama-3.2-1B-base is a more general-purpose, less
    # code-specialized, and far more established base model. Its own
    # base-model-quality check (check_base_model_quality.py) should be
    # rerun after this swap -- MiniCPM5's baseline was already established,
    # Llama-3.2-1B's hasn't been.
    #
    # IMPORTANT: rerun check_tokenizer_overlap.py after this swap. Llama's
    # tokenizer is a different BPE than MiniCPM5's, so the overlap numbers
    # from the earlier check no longer apply and need to be re-verified
    # against deepseek-coder-1.3b-base specifically.
    #
    # Non-gated fallback if Llama-3.2-1B access issues resurface:
    #   hf_model_id="openbmb/MiniCPM5-1B"
    #   hf_model_id="google/gemma-2-2b"


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
    dim: int = 1024         # bumped from 512 after diagnosis: generation was
                             # incoherent almost immediately (by ~15 tokens)
                             # and identically so regardless of length, which
                             # pointed away from exposure bias/undertraining
                             # and toward the bottleneck itself being too
                             # narrow to carry enough content from the source
                             # expert. This is a cheap-ish test of that
                             # hypothesis before considering a bigger
                             # architecture change (e.g. multi-token/
                             # cross-attention handoff instead of a single
                             # summary vector).
                             #
                             # IMPORTANT: this makes existing checkpoints
                             # (trained with dim=512) INCOMPATIBLE -- their
                             # to_shared/from_shared weight shapes won't
                             # match. Phase 2 and Phase 3 both need to be
                             # rerun from scratch with this new dim; don't
                             # try to --resume-from an old checkpoint here.
    num_vectors: int = 1    # 1 = original single-summary-vector handoff.
                             # >1 = keep the last num_vectors token hidden
                             # states (instead of pooling to one) and inject
                             # all of them as a short virtual prefix. This
                             # preserves some positional/sequential
                             # structure that a single vector necessarily
                             # discards -- worth trying if dim alone (going
                             # 512->1024) doesn't fix the incoherence, since
                             # the failure mode observed (wrong almost
                             # immediately, not length-dependent) looks more
                             # like "wrong kind of representation" than
                             # "not enough raw capacity."
                             # NOTE: changing this also breaks checkpoint
                             # compatibility (to_shared/from_shared are the
                             # same Linear either way, but how many times
                             # they're applied per handoff changes).


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
    # code_search_net's own HF repo is a legacy loading SCRIPT, which
    # datasets>=4.0 no longer supports at all. Use a pre-converted,
    # script-free Parquet mirror instead. If this specific mirror ever
    # moves/breaks, swap in another Parquet-format CodeSearchNet Python
    # mirror here -- build_pairs() auto-detects column names across a few
    # common schemas.
    hf_dataset_id: str = "Nan-Do/code-search-net-python"
    hf_dataset_split: str = "train"
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
    val_every: int = 20
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-3
    grad_clip: float = 1.0
    grad_checkpointing: bool = True
    log_every: int = 5
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 20           # very frequent -- sessions have been dying <30 steps in


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
    log_every: int = 5
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 20           # frequent -- Phase 2 taught us sessions can die early


@dataclass
class GenConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    max_switches: int = 4
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 3


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

    @staticmethod
    def debug_multivector():
        """Tiny-model config with multi-vector handoff enabled, for smoke
        testing that path before spending real GPU time on it."""
        cfg = Config.debug()
        cfg.shared.num_vectors = 4
        return cfg
