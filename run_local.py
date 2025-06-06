import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
import pandas as pd
import json
import os
from config import TrainingConfig
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Local LLM Finetuning Script')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to local model directory')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data file (json or csv)')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for saving model')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    return parser.parse_args()

def load_local_dataset(data_path):
    """載入本地數據集"""
    print(f"正在載入本地數據：{data_path}")
    
    if data_path.endswith('.json'):
        # 載入 JSON 格式數據
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 假設 JSON 格式為 [{"instruction": "...", "input": "...", "output": "..."}]
        if isinstance(data, list):
            return Dataset.from_pandas(pd.DataFrame(data))
    
    elif data_path.endswith('.csv'):
        # 載入 CSV 格式數據
        df = pd.read_csv(data_path)
        return Dataset.from_pandas(df)
    
    else:
        raise ValueError("不支持的數據格式。請使用 .json 或 .csv 格式")

def format_instruction(example):
    """格式化指令數據"""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    if input_text:
        prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
    else:
        prompt = f"<s>[INST] {instruction} [/INST]"
        
    response = f"{output}</s>"
    return prompt + response

def prepare_dataset(dataset, tokenizer, max_length):
    """準備數據集"""
    def preprocess_function(example):
        # 格式化數據
        full_prompt = format_instruction(example)
        
        # 分詞
        inputs = tokenizer(
            full_prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None
        )
        
        # 設置標籤
        inputs["labels"] = inputs["input_ids"].copy()
        
        return inputs
    
    # 處理數據集
    tokenized_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        desc="正在處理數據集"
    )
    
    return tokenized_dataset

def setup_model_and_tokenizer(model_path):
    """設置本地模型和分詞器"""
    print(f"正在載入本地模型：{model_path}")
    
    # 載入分詞器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False  # 某些模型可能需要設置為 False
    )
    
    # 確保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 載入模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        load_in_4bit=True,  # 使用 4-bit 量化以節省顯存
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer

def setup_lora(model):
    """設置 LoRA"""
    print("正在設置 LoRA...")
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def main():
    # 解析命令行參數
    args = parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 載入模型和分詞器
    model, tokenizer = setup_model_and_tokenizer(args.model_path)
    
    # 設置 LoRA
    model = setup_lora(model)
    
    # 載入數據集
    dataset = load_local_dataset(args.data_path)
    
    # 準備數據集
    tokenized_dataset = prepare_dataset(dataset, tokenizer, args.max_length)
    
    # 設置訓練參數
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit"
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
    print(f"正在儲存模型到：{args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ 訓練完成！模型已儲存到：{args.output_dir}")

if __name__ == "__main__":
    main() 