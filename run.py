import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb
from config import TrainingConfig, default_config
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='LLM Finetuning Script')
    parser.add_argument('--base_model', type=str, help='Base model to finetune')
    parser.add_argument('--dataset', type=str, help='Dataset to use for finetuning')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    return parser.parse_args()

def setup_wandb(config):
    """設置 Weights & Biases 追蹤"""
    wandb.init(
        project="llm-finetuning",
        config={
            "base_model": config.base_model,
            "dataset": config.dataset_name,
            "lora_r": config.lora_r,
            "learning_rate": config.learning_rate,
        }
    )

def load_model_and_tokenizer(config):
    """載入模型和分詞器"""
    print(f"正在載入模型：{config.base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        load_in_4bit=config.load_in_4bit,
        device_map=config.device_map,
        trust_remote_code=True
    )
    
    return model, tokenizer

def setup_lora(model, config):
    """設置 LoRA"""
    print("正在設置 LoRA...")
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def prepare_dataset(tokenizer, config):
    """準備數據集"""
    print(f"正在載入數據集：{config.dataset_name}")
    
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    
    def preprocess_function(example):
        prompt = f"<s>[INST] {example['question']} [SEP] {example['context']} [/INST]"
        inputs = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=config.max_length
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def main():
    # 解析命令行參數
    args = parse_args()
    config = default_config
    
    # 更新配置
    if args.base_model:
        config.base_model = args.base_model
    if args.dataset:
        config.dataset_name = args.dataset
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # 創建輸出目錄
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 設置實驗追蹤
    setup_wandb(config)
    
    # 載入模型和分詞器
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 設置 LoRA
    model = setup_lora(model, config)
    
    # 準備數據集
    tokenized_dataset = prepare_dataset(tokenizer, config)
    
    # 設置訓練參數
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="wandb"
    )
    
    # 開始訓練
    print("開始訓練...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    
    # 儲存模型
    print(f"正在儲存模型到：{config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # 結束實驗追蹤
    wandb.finish()
    
    print(f"✅ 訓練完成！模型已儲存到：{config.output_dir}")

if __name__ == "__main__":
    main()
