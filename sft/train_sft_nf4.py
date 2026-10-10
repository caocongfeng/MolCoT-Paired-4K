#!/usr/bin/env python3
"""train_sft.py with Qwen3.5-27B loaded in 4-bit NF4 (double quantisation, BF16 compute); LoRA on top.

Only the model-loading function is replaced; arguments, training and evaluation are those of train_sft.py.
Used for the Qwen3.5-27B runs of the model-scale experiment (see sft/run_sft.sh).
"""
import json
from pathlib import Path

import train_sft as sft
from transformers import BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen3.5-27B"


def load_nf4_model(model_name):
    model_path = Path(model_name)
    adapter_path = model_path / "adapter_config.json"
    is_adapter = adapter_path.is_file()
    base_name = json.loads(adapter_path.read_text(encoding="utf-8"))["base_model_name_or_path"] if is_adapter else model_name
    if base_name != BASE_MODEL:
        raise ValueError(f"This NF4 loader expects {BASE_MODEL}, got {base_name}.")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=sft.torch.bfloat16,
    )
    model = sft.AutoModelForCausalLM.from_pretrained(
        base_name, dtype=sft.torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
        quantization_config=quantization_config,
    )
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("27B was not loaded in 4-bit mode.")
    print(f"27B loaded: NF4 + double quantization, BF16 compute; footprint={model.get_memory_footprint() / 2**30:.2f} GiB", flush=True)
    if is_adapter:
        model = sft.PeftModel.from_pretrained(model, str(model_path))
    return model


if __name__ == "__main__":
    args = sft.parse_args()
    if args.full_finetune:
        raise ValueError("The 27B NF4 branch supports LoRA only.")
    sft.load_model = load_nf4_model
    sft.main()
