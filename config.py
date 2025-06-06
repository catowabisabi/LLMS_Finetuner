from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainingConfig:
    # 模型配置
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.1"  # 預設使用較小的模型
    output_dir: str = "./output"
    
    # LoRA 配置
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # 訓練配置
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 1
    learning_rate: float = 2e-4
    max_length: int = 256
    
    # 數據集配置
    dataset_name: str = "squad"
    dataset_split: str = "train[:1000]"
    
    # 硬體配置
    load_in_4bit: bool = True
    device_map: str = "auto"
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 預設配置
default_config = TrainingConfig()
