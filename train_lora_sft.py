#!/usr/bin/env python3
"""
LoRA / QLoRA SFT skeleton for TNWO chat-style datasets.

Example:
  python train_lora_sft.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --train-file train_dataset/aggressive_sft_20.jsonl \
    --output-dir outputs/lora_aggressive_qwen25_7b \
    --load-in-4bit
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA SFT adapter on TNWO chat-style data.")
    parser.add_argument("--model-name", required=True, help="Base model name or local path")
    parser.add_argument("--train-file", required=True, help="Path to chat-style JSONL file")
    parser.add_argument("--output-dir", required=True, help="Directory to save adapter")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training if supported")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit base model loading (QLoRA-style)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="LoRA target modules",
    )
    return parser.parse_args()


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(f"Line {line_no} missing 'messages'")
            rows.append(row)
    return rows


def build_text_dataset(rows, tokenizer):
    formatted = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted.append(
            {
                "text": text,
                "messages": row["messages"],
                "metadata": row.get("metadata", {}),
            }
        )
    return Dataset.from_list(formatted)


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16" if args.bf16 else "float16",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    rows = load_jsonl(args.train_file)
    train_dataset = build_text_dataset(rows, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary_path = Path(args.output_dir) / "train_config.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "train_file": args.train_file,
                "output_dir": args.output_dir,
                "num_samples": len(rows),
                "max_seq_length": args.max_seq_length,
                "load_in_4bit": args.load_in_4bit,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": args.target_modules,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
