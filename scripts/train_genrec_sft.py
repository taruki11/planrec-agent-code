#!/usr/bin/env python3
"""Continue-pretrain or SFT a Qwen-style causal LM on semantic item tokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planrec.genrec.hf_runtime import load_model_and_tokenizer, lora_config
from planrec.genrec.provenance import write_run_manifest
from planrec.genrec.semantic_ids import ResidualKMeansSemanticTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="只取每个 train-file 前 N 行（诊断/冒烟用），None=全部")
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objective", choices=("pretrain", "sft"), default="sft")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--deepspeed", type=Path)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--optimizer", default="adamw_torch",
                        choices=("adamw_torch", "adamw_bnb_8bit"),
                        help="adamw_bnb_8bit halves AdamW memory (1.7B on 15 GB GPUs)")
    parser.add_argument("--train-embedding", action="store_true",
                        help="解锁 resize 新增的语义 token embedding 行（LoRA 不覆盖 embedding 层；"
                             "新增行是随机初始化，不解锁则条件信号无法进入输入表示）")
    args = parser.parse_args()

    try:
        from datasets import concatenate_datasets, load_dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError("install requirements-genrec.txt before training") from error

    # optimizer 参数签名自适应：部分 trl 版本 SFTConfig 不接收 optimizer
    # 仅当 SFTConfig 接受 optimizer 参数时才注入（部分 trl 版本不支持该 kwarg）
    optimizer_kw: dict[str, object] = {}
    try:
        import inspect
        if "optimizer" in inspect.signature(SFTConfig).parameters:
            optimizer_kw["optimizer"] = args.optimizer
        else:
            print("SFTConfig 不支持 optimizer 参数; 使用其默认优化器", flush=True)
    except Exception:
        print("SFTConfig 签名检测失败; 使用默认优化器", flush=True)

    semantic_ids = ResidualKMeansSemanticTokenizer.load(args.semantic_ids)
    model, tokenizer = load_model_and_tokenizer(args.model, semantic_ids, args.dtype, args.attention)
    if args.train_embedding:
        emb = model.get_input_embeddings().weight
        orig_size = model.config.vocab_size - len(tokenizer.get_added_vocab())
        with torch.no_grad():
            emb.data[:orig_size].requires_grad_(False)
            emb.data[orig_size:].requires_grad_(True)
        print(f"EMBEDDING_UNLOCKED rows {orig_size}:{emb.shape[0]} trainable "
              f"({emb.shape[0] - orig_size} semantic rows)", flush=True)
    train_parts = [load_dataset("json", data_files=str(path), split="train") for path in args.train_file]
    if args.limit is not None:
        train_parts = [p.select(range(min(args.limit, len(p)))) for p in train_parts]
    train_dataset = train_parts[0] if len(train_parts) == 1 else concatenate_datasets(train_parts)
    eval_dataset = load_dataset("json", data_files=str(args.eval_file), split="train") if args.eval_file else None

    # completion_only_loss/packing 按 SFTConfig 签名兼容注入（trl 0.14 无 flag，
    # prompt/completion 数据自动只算 completion loss，语义一致）
    config_kw = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length,
        gradient_checkpointing=True,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=500,
        report_to="none",
        deepspeed=str(args.deepspeed) if args.deepspeed else None,
        **optimizer_kw,
    )
    for flag, value in (("completion_only_loss", args.objective == "sft"),
                        ("packing", args.objective == "pretrain")):
        if flag in inspect.signature(SFTConfig).parameters:
            config_kw[flag] = value
        else:
            print(f"SFTConfig 无 {flag} 参数; trl 该版本自行处理该语义", flush=True)
    config = SFTConfig(**config_kw)
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config(args.lora, args.lora_rank, args.lora_rank * 2),
    )
    manifest_args = vars(args).copy()
    manifest_files = [*args.train_file, *([args.eval_file] if args.eval_file else [])]

    def mark(status: str) -> None:
        write_run_manifest(
            args.output_dir, args.objective, args.model, args.semantic_ids, manifest_files, manifest_args, status
        )
    mark("running")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if args.lora:
        # 训练后 merge 输出完整权重（L3 要求 full checkpoint；LoRA 仅作显存手段）
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(Path(args.output_dir) / "merged"))
        tokenizer.save_pretrained(str(Path(args.output_dir) / "merged"))
        print("LORA_MERGED_SAVED", flush=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    mark("complete")


if __name__ == "__main__":
    main()
