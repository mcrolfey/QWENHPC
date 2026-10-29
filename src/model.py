"""Vision language model loading with QLoRA (bitsandbytes + PEFT).

The language model is quantised to 4-bit NF4 (or 8-bit) and adapted with
LoRA. The vision encoder is kept in bf16 (quantising it hurts fine-grained
visual detail such as thin fibres) and is frozen by default; pass
``tune_vision=True`` to also attach LoRA adapters to the vision blocks, which
is worth trying for the microscopic domain shift.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
LM_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
VISION_TARGETS = r".*visual\.blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.gate_proj|mlp\.up_proj|mlp\.down_proj)$"
SKIP_QUANT = ["visual", "lm_head"]


def compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def quant_config(quant: str) -> BitsAndBytesConfig | None:
    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(),
            llm_int8_skip_modules=SKIP_QUANT,
        )
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=SKIP_QUANT)
    if quant == "none":
        return None
    raise ValueError(f"Unknown quantisation '{quant}' (4bit | 8bit | none)")


def load_processor(model_name: str, min_pixels: int, max_pixels: int):
    """Processor with a bounded visual-token budget.

    Qwen2.5-VL uses 28x28 pixel patches per visual token, so max_pixels of
    1024*28*28 caps an image at ~1024 visual tokens.
    """
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def load_base_model(model_name: str, quant: str = "4bit", attn_implementation: str = "sdpa"):
    kwargs = dict(
        torch_dtype=compute_dtype(),
        quantization_config=quant_config(quant),
        attn_implementation=attn_implementation,
    )
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    elif quant != "none":
        raise RuntimeError("bitsandbytes quantisation requires a CUDA GPU; use --quant none on CPU.")
    if "qwen2.5-vl" in model_name.lower() or "qwen2_5_vl" in model_name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration as cls
    else:
        cls = AutoModelForVision2Seq
    model = cls.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    return model


def load_model_for_training(
    model_name: str = DEFAULT_MODEL,
    quant: str = "4bit",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    tune_vision: bool = False,
    gradient_checkpointing: bool = True,
    attn_implementation: str = "sdpa",
):
    model = load_base_model(model_name, quant, attn_implementation)
    for p in model.parameters():
        p.requires_grad_(False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    target = f"(?:{LM_TARGETS})|(?:{VISION_TARGETS})" if tune_vision else LM_TARGETS
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    if n_lora == 0:
        raise RuntimeError("No LoRA modules were attached - target regex did not match this model.")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("LoRA modules: %d | trainable params: %.2fM / %.2fM (%.3f%%)",
             n_lora, trainable / 1e6, total / 1e6, 100 * trainable / total)
    return model


def load_model_for_inference(
    model_name: str,
    adapter_dir: str | Path | None,
    quant: str = "4bit",
    attn_implementation: str = "sdpa",
):
    """Base model (+ LoRA adapter if given) in eval mode. ``adapter_dir=None`` -> zero-shot."""
    model = load_base_model(model_name, quant, attn_implementation)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        log.info("Loaded LoRA adapter from %s", adapter_dir)
    else:
        log.info("No adapter given: evaluating the zero-shot base model")
    model.config.use_cache = True
    model.eval()
    return model
