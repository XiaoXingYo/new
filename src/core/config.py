import torch
import yaml
from pathlib import Path
from typing import Optional, Tuple
from pydantic import BaseModel


class DataConfig(BaseModel):
    height: int = 32
    width: int = 128
    chars: str = "0123456789"
    blank_label: int = 10
    emnist_dir: str = "./data"
    augment: bool = True
    seq_len_range: Tuple[int, int] = (3, 6)
    val_ratio: float = 0.1
    noise_prob: float = 0.5
    elastic_prob: float = 0.3
    overlap_range: Tuple[int, int] = (-1,2 )#允许外部控制字符的重叠度
    num_workers: int = 0

class ModelConfig(BaseModel):
    backbone: str = "vgg"
    hidden_size: int = 256
    rnn_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.2


class SchedulerConfig(BaseModel):
    type: str = "OneCycleLR"
    pct_start: float = 0.1
    div_factor: float = 25.0
    final_div_factor: float = 1000.0


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.01


class TrainConfig(BaseModel):
    batch_size: int = 128
    epochs: int = 50
    epoch_size: int = 2000
    lr: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 5.0
    save_every: int = 5
    val_every: int = 1
    log_every: int = 50
    device: str = "auto"
    output_dir: str = "./checkpoints"
    resume: Optional[str] = None
    scheduler: SchedulerConfig = SchedulerConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()


class InferenceConfig(BaseModel):
    model_path: str = "./checkpoints/best_ctc.pth"
    decode_type: str = "greedy"
    beam_width: int = 5


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "./logs"
    print_freq: int = 50


class Config:
    def __init__(self, config_path: str | None = None):
        self.project_name: str = "crnn_ocr"

        # 初始化结构
        self.data: DataConfig = DataConfig()
        self.model: ModelConfig = ModelConfig()
        self.train: TrainConfig = TrainConfig()
        self.inference: InferenceConfig = InferenceConfig()
        self.logging: LoggingConfig = LoggingConfig()

        # 加载 YAML 覆盖
        if config_path and Path(config_path).exists():
            self.load(config_path)

        # 动态计算属性
        self.data_num_classes: int = len(self.data.chars) + 1

        # 修复：动态设备选择
        if self.train.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.train.device)

    def load(self, path: str) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if 'project_name' in data:
            self.project_name = data['project_name']
        if 'data' in data:
            self.data = DataConfig(**data['data'])
        if 'model' in data:
            self.model = ModelConfig(**data['model'])
        if 'train' in data:
            self.train = TrainConfig(**data['train'])
        if 'inference' in data:
            self.inference = InferenceConfig(**data['inference'])
        if 'logging' in data:
            self.logging = LoggingConfig(**data['logging'])

    def dump(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump({
                'project_name': self.project_name,
                'data': self.data.model_dump(),
                'model': self.model.model_dump(),
                'train': self.train.model_dump(),
                'inference': self.inference.model_dump()
            }, f, default_flow_style=False)