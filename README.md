# LLMs Finetuner / LLM 模型微調工具

[English](#english) | [繁體中文](#繁體中文)

---

# 繁體中文

## 簡介
這是一個用於微調大型語言模型（LLMs）的工具，使用 LoRA（Low-Rank Adaptation）技術來實現高效微調。支持本地模型和線上模型的訓練，並提供多種數據格式的支持。

## 功能特點
- 支持多種開源 LLMs（如 Mistral、Llama 等）
- 使用 LoRA 技術進行高效微調
- 支持 4-bit 量化訓練，降低顯存需求
- 支持本地模型和數據集
- 整合 Weights & Biases 進行實驗追蹤
- 靈活的配置系統

## 安裝
1. 克隆專案：
```bash
git clone https://github.com/yourusername/LLMS_Finetuner.git
cd LLMS_Finetuner
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

3. （可選）設置 Weights & Biases：
```bash
wandb login
```

## 使用方法

### 1. 使用線上模型（run.py）
適用於直接使用 Hugging Face 上的模型：

```bash
# 基本使用
python run.py

# 自定義參數
python run.py \
  --base_model "mistralai/Mistral-7B-Instruct-v0.1" \
  --dataset "squad" \
  --output_dir "./output"
```

### 2. 使用本地模型（run_local.py）
適用於已下載的本地模型：

```bash
python run_local.py \
  --model_path "./models/your_model" \
  --data_path "./data/training_data.json" \
  --output_dir "./output" \
  --max_length 512
```

### 3. 數據格式
支持兩種格式：

#### JSON 格式：
```json
[
  {
    "instruction": "任務指令",
    "input": "輸入內容（可選）",
    "output": "期望輸出"
  }
]
```

#### CSV 格式：
```csv
instruction,input,output
"任務指令","輸入內容","期望輸出"
```

## 使用場景

### 場景一：微調通用對話模型
```bash
python run.py \
  --base_model "mistralai/Mistral-7B-Instruct-v0.1" \
  --dataset "./data/conversation.json" \
  --output_dir "./output/chat_model"
```

### 場景二：領域特定任務
```bash
python run_local.py \
  --model_path "./models/medical_model" \
  --data_path "./data/medical_qa.json" \
  --output_dir "./output/medical_model" \
  --max_length 1024
```

### 場景三：多語言模型微調
```bash
python run.py \
  --base_model "Helsinki-NLP/opus-mt-zh-en" \
  --dataset "./data/translation.json" \
  --output_dir "./output/translator"
```

## 系統需求
- Python 3.8+
- CUDA 支持的 GPU（建議至少 16GB 顯存）
- 至少 32GB RAM

## 注意事項
1. 請確保有足夠的硬體資源
2. 建議使用虛擬環境
3. 大型模型建議使用 4-bit 量化
4. 定期保存訓練檢查點
5. 注意數據隱私和安全

---

# English

## Introduction
This is a tool for fine-tuning Large Language Models (LLMs) using LoRA (Low-Rank Adaptation) technology. It supports both local and online models, with various data format support.

## Features
- Support for multiple open-source LLMs (e.g., Mistral, Llama)
- Efficient fine-tuning using LoRA technology
- 4-bit quantization training for reduced VRAM usage
- Support for local models and datasets
- Weights & Biases integration for experiment tracking
- Flexible configuration system

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/LLMS_Finetuner.git
cd LLMS_Finetuner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Setup Weights & Biases:
```bash
wandb login
```

## Usage

### 1. Using Online Models (run.py)
For using models directly from Hugging Face:

```bash
# Basic usage
python run.py

# Custom parameters
python run.py \
  --base_model "mistralai/Mistral-7B-Instruct-v0.1" \
  --dataset "squad" \
  --output_dir "./output"
```

### 2. Using Local Models (run_local.py)
For using downloaded local models:

```bash
python run_local.py \
  --model_path "./models/your_model" \
  --data_path "./data/training_data.json" \
  --output_dir "./output" \
  --max_length 512
```

### 3. Data Formats
Supports two formats:

#### JSON Format:
```json
[
  {
    "instruction": "Task instruction",
    "input": "Input content (optional)",
    "output": "Expected output"
  }
]
```

#### CSV Format:
```csv
instruction,input,output
"Task instruction","Input content","Expected output"
```

## Use Cases

### Case 1: Fine-tuning General Conversation Models
```bash
python run.py \
  --base_model "mistralai/Mistral-7B-Instruct-v0.1" \
  --dataset "./data/conversation.json" \
  --output_dir "./output/chat_model"
```

### Case 2: Domain-Specific Tasks
```bash
python run_local.py \
  --model_path "./models/medical_model" \
  --data_path "./data/medical_qa.json" \
  --output_dir "./output/medical_model" \
  --max_length 1024
```

### Case 3: Multilingual Model Fine-tuning
```bash
python run.py \
  --base_model "Helsinki-NLP/opus-mt-zh-en" \
  --dataset "./data/translation.json" \
  --output_dir "./output/translator"
```

## System Requirements
- Python 3.8+
- CUDA-compatible GPU (16GB+ VRAM recommended)
- 32GB+ RAM

## Important Notes
1. Ensure sufficient hardware resources
2. Use virtual environment recommended
3. Use 4-bit quantization for large models
4. Save checkpoints regularly
5. Mind data privacy and security

## License
MIT License
