import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# === 1. 基礎設置 ===
base_model = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
output_dir = "./lora_mistral24B"

# === 2. 載入 tokenizer 與模型 ===
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 防止 padding 錯誤

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    load_in_4bit=True,  # 使用 4bit 節省記憶體
    device_map="auto",
    trust_remote_code=True
)

# === 3. 設定 LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === 4. 載入 Hugging Face 數據集 ===
dataset = load_dataset("squad", split="train[:1000]")  # 載入 SQuAD 數據集

# 數據處理與分詞
def preprocess_function(example):
    prompt = f"<s>[INST] {example['question']} [SEP] {example['context']} [/INST]"
    inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=256)
    inputs["labels"] = inputs["input_ids"].copy()  # 訓練用的 labels
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# === 5. 訓練參數 ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    bf16=False,
    optim="paged_adamw_8bit",  # 高效優化器
    report_to="none"
)

# === 6. 訓練 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# === 7. 儲存 LoRA 權重 ===
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ LoRA 微調完成，儲存於：{output_dir}")
