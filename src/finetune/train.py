"""
Local fine-tuning script using PEFT (LoRA / QLoRA).

This script is for local GPU-based training. For Colab, use notebooks/colab_finetune_xauusd.ipynb instead.

Usage:
    python src/finetune/train.py \\
        --model_name "mistralai/Mistral-7B-Instruct-v0.1" \\
        --train_jsonl "data/datasets/train_xauusd.jsonl" \\
        --test_jsonl "data/datasets/test_xauusd.jsonl" \\
        --output_dir "models/checkpoints/xauusd_lora" \\
        --num_epochs 3
"""

import json
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset


def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_dataset(examples, tokenizer, max_length=256):
    """Prepare examples for training."""
    texts = [f"{ex['prompt']}\n{ex['response']}" for ex in examples]
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def train(
    model_name: str,
    train_jsonl: str,
    test_jsonl: str,
    output_dir: str = "models/checkpoints/xauusd_lora",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    use_4bit: bool = True,
):
    """
    Fine-tune model using LoRA.
    
    Args:
        model_name: base model name or path
        train_jsonl: path to training JSONL
        test_jsonl: path to test JSONL
        output_dir: where to save adapters
        num_epochs: number of training epochs
        batch_size: batch size
        learning_rate: learning rate
        use_4bit: whether to use 4-bit quantization
    """
    
    print(f"Loading data from {train_jsonl} and {test_jsonl}...")
    train_data = load_jsonl(train_jsonl)
    test_data = load_jsonl(test_jsonl)
    print(f"  Training: {len(train_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    
    print(f"\nLoading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model {model_name}...")
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    
    # Prepare model for k-bit training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    print("Setting up LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Create datasets
    print("Preparing datasets...")
    train_dataset = Dataset.from_dict({
        "text": [f"{ex['prompt']}\n{ex['response']}" for ex in train_data]
    })
    test_dataset = Dataset.from_dict({
        "text": [f"{ex['prompt']}\n{ex['response']}" for ex in test_data]
    })
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        save_steps=50,
        eval_steps=25,
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=0.001,
        warmup_steps=5,
        optim="paged_adamw_32bit",
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )
    
    print(f"Starting training for {num_epochs} epochs...")
    trainer.train()
    
    # Save adapter
    print(f"\nSaving adapters to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="Base model name or path",
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default="data/datasets/train_xauusd.jsonl",
        help="Path to training JSONL",
    )
    parser.add_argument(
        "--test_jsonl",
        type=str,
        default="data/datasets/test_xauusd.jsonl",
        help="Path to test JSONL",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/checkpoints/xauusd_lora",
        help="Output directory for adapters",
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        test_jsonl=args.test_jsonl,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_4bit=args.use_4bit,
    )
