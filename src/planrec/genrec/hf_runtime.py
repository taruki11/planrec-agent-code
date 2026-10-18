from __future__ import annotations

from typing import Any

import torch

from planrec.genrec.semantic_ids import ResidualKMeansSemanticTokenizer


def torch_dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ValueError(f"unsupported dtype: {name}")
    return values[name]


def load_model_and_tokenizer(
    model_name_or_path: str,
    semantic_ids: ResidualKMeansSemanticTokenizer,
    dtype: str = "bfloat16",
    attention: str | None = None,
) -> tuple[Any, Any]:
    """Load a Hugging Face causal LM and register itemic tokens."""

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install requirements-genrec.txt before large-model training") from error

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_special_tokens({"additional_special_tokens": semantic_ids.added_tokens()})

    transformers_major = int(transformers.__version__.split(".", maxsplit=1)[0])
    dtype_key = "dtype" if transformers_major >= 5 else "torch_dtype"
    model_kwargs: dict[str, Any] = {dtype_key: torch_dtype(dtype)}
    if attention:
        model_kwargs["attn_implementation"] = attention
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if added:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def lora_config(enabled: bool, rank: int = 32, alpha: int = 64) -> Any | None:
    if not enabled:
        return None
    try:
        from peft import LoraConfig
    except ImportError as error:
        raise RuntimeError("PEFT is required when --lora is enabled") from error
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
