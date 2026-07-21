# Cooperating Experts v2 — Pretrained, Frozen Backbones

Same idea as v1 (two experts with genuinely different vocabularies,
cooperating through switch tokens and a shared-space bridge), but the
experts are now **frozen pretrained HF models** instead of from-scratch
transformers. This sidesteps the local-GPU bottleneck for both backbone
pretraining and LLM-based data generation: no LLM generation step is
needed for Phase 2 at all, since CodeSearchNet's (docstring, function)
pairs already give the model paired English/Python text.

## What actually trains

Only:
- `to_shared` / `from_shared` linear projections, per expert
- the newly-added `<switch:NAME>` embedding/head rows (masked so the rest
  of each pretrained vocabulary never updates)
- (Phase 3 only) small LoRA adapters on each frozen backbone

Everything else is frozen. This is why it fits on a free-tier Colab T4 and
your 4GB laptop GPU can be used for dev, even though the backbones
themselves are 1-3B params each.

## Run order

```bash
pip install -r requirements.txt

# 0. FIRST: confirm the two chosen models really have different tokenizers.
#    Don't skip this -- it's the whole premise of the project.
python check_tokenizer_overlap.py

# 0b. Cheap end-to-end sanity check on tiny stand-in models, laptop or CPU.
python smoke_test.py

# 1. Build the CodeSearchNet handoff-pair dataset (no LLM generation needed)
python data.py

# 2. Phase 2: stitching (Colab T4 recommended for the real 1-3B pair)
python train_stitch.py

# 3. Phase 3: LoRA + mixed-session fine-tuning, continuing from the Phase-2 bridge
python train_lora.py --bridge-ckpt checkpoints/model_stitched_best.pt

# 4. Generate
python generate.py "def quicksort(arr):" --expert python \
    --bridge-ckpt checkpoints/model_stitched_best.pt
```

Use `--debug` on any script to run against tiny stand-in models
(distilgpt2 + codegen-350M-mono) instead of the real pair — useful for
iterating on your laptop before spending Colab session time.

## What changed vs. v1, and why

| v1 (from scratch) | v2 (pretrained, frozen) |
|---|---|
| 3-phase: pretrain each expert, then stitch, then unfreeze everything | 2 phases: stitch (frozen), then LoRA + bridge (still mostly frozen) — no from-scratch pretraining phase at all |
| Synthetic LLM-generated conversational corpus | Raw CodeSearchNet pairs directly, no generation step |
| ~44.6M total params, all trainable eventually | 2-6B total params, only a few hundred thousand to a few million ever trainable |
| Custom `Expert.forward` accepts an arbitrary prepended hidden state directly | Manual decode loop using `inputs_embeds` to seed a KV cache, since HF `.generate()` doesn't expose this |
| Full unfreeze in phase 3 | LoRA adapters in phase 3 (full unfreeze of a 1-3B model isn't realistic on this hardware, and risks catastrophic forgetting) |
| 256-dim shared bottleneck | 512-dim (larger, more entangled pretrained hidden spaces) |

## Honest caveats (carried over from the design discussion, now concrete)

- **The handoff was never part of these models' pretraining.** Your v1
  experts were trained from step one to expect a carried hidden state.
  These pretrained backbones weren't. Phase 2 is asking a few hundred
  thousand bridge parameters to retrofit a communication channel into
  models that never had one — this may need more stitching data or a
  larger bottleneck than v1's synthetic run suggested. Watch val loss
  closely; there's no guarantee this converges as cleanly as the toy
  version did.
- **Broken handoffs are harder to notice.** A 1-3B model has strong
  enough priors that it can produce fluent, plausible-looking output even
  if the injected vector is close to noise. Build the "zero out the
  carried vector and compare" check mentioned earlier into your eval loop
  before trusting any generated output qualitatively.
- **Sessions in Phase 3 are simple 2-switch sessions** (docstring →
  code → closer), not the richer multi-switch-per-reply structure v1's
  synthetic generator produced. If you want that richness, an LLM
  generation step could come back here specifically (much cheaper than
  regenerating the whole Phase 2 corpus, since Phase 2 no longer needs it)
  — but get the simple version validated first.
- **LoRA `target_modules` in `config.py` assumes an attention naming
  convention** (`q_proj`/`k_proj`/`v_proj`/`o_proj`) that fits most modern
  architectures (Llama/Mistral/Gemma-style) but not all (e.g. GPT-2-style
  `c_attn`). If you swap in a model family with different module names,
  update `LoraTrainConfig.target_modules` accordingly — `peft` will error
  out clearly if the names don't match, so this fails loudly rather than
  silently.
- **Checkpoints are small on purpose.** `save_bridge_checkpoint` only
  saves the bridge + new-token rows, not backbone weights — those are
  re-downloaded from the HF hub id at load time. Don't be surprised the
  `.pt` files are megabytes, not gigabytes.

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters, including the debug tiny-model pair |
| `check_tokenizer_overlap.py` | Run first: validates the vocabulary-disjointness premise |
| `models.py` | `FrozenExpert`: 4-bit loading, gradient-masked special tokens, bridge projections |
| `data.py` | CodeSearchNet loading, cleaning, dedup, pair extraction |
| `dataset.py` | `HandoffDataset` + directional batching for Phase 2 |
| `train_stitch.py` | Phase 2 training loop (frozen backbones, bridge only) |
| `train_lora.py` | Phase 3 training loop (LoRA + bridge, mixed sessions) |
| `generate.py` | Manual decode loop with live expert switching |
| `smoke_test.py` | Cheap end-to-end check on tiny models, run before Colab |
| `requirements.txt` | Dependencies |
