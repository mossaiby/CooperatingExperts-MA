"""
Prints every Linear-layer module name in each backbone, so you can confirm
(or fix) LoraTrainConfig.target_modules in config.py before launching
train_lora.py. peft fails loudly if target_modules don't match anything in
the model, but it's cheaper to check now than to load both 1-3B backbones
just to hit that error a minute in.

Usage:
    python check_lora_targets.py
    python check_lora_targets.py --debug
"""
import argparse

from transformers import AutoModelForCausalLM
import torch.nn as nn

from config import Config


def list_linear_module_names(model):
    """
    Returns the set of distinct 'leaf' module name patterns (with layer
    index replaced by 'N') for every nn.Linear in the model, so you see
    e.g. 'q_proj' once instead of 'model.layers.0.self_attn.q_proj',
    'model.layers.1.self_attn.q_proj', ... 24 times.
    """
    import re
    patterns = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            generic = re.sub(r"\.\d+\.", ".N.", name)
            patterns.add(generic)
    return sorted(patterns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    cfg = Config.debug() if args.debug else Config.default()

    for role, spec in (("english", cfg.pair.english), ("python", cfg.pair.python)):
        print(f"\n=== {role}: {spec.hf_model_id} ===")
        model = AutoModelForCausalLM.from_pretrained(spec.hf_model_id, torch_dtype="auto")
        names = list_linear_module_names(model)
        for n in names:
            print(f"  {n}")
        del model

    print(f"\nCurrent config target_modules: {cfg.lora.target_modules}")
    print("Confirm each of these strings appears as a SUFFIX of at least one "
          "module name printed above, for BOTH models. If a name printed "
          "above doesn't match (e.g. 'c_attn' instead of 'q_proj'/'k_proj'/"
          "'v_proj'), update LoraTrainConfig.target_modules in config.py.")


if __name__ == "__main__":
    main()
