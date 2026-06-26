import os
import cv2
import numpy as np
import random
import sys
from pathlib import Path
from tqdm import tqdm
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.core.config import Config
from src.data.generator import OCRDataGenerator
def build_static_dataset():
    base_dir = PROJECT_ROOT / "data" / "static_dataset"
    TOTAL_SAMPLES = 200000
    VAL_RATIO = 0.2
    dirs = {
        "train": base_dir / "train" / "images",
        "val": base_dir / "val" / "images"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    labels_file = {
        "train": open(base_dir / "train" / "labels.txt", "w", encoding="utf-8"),
        "val": open(base_dir / "val" / "labels.txt", "w", encoding="utf-8")
    }
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    cfg = Config(str(config_path))
    cfg.data.augment = False
    generator = OCRDataGenerator(cfg)
    print(f"🚀 静态数据集生成")
    print(f"📍 存储路径: {base_dir}")
    print(f"📊 规模: {TOTAL_SAMPLES} 张 (8:2 拆分)")
    for i in tqdm(range(TOTAL_SAMPLES), desc="Generating Data"):
        img_tensor, label = generator.generate_sample()
        img_np = (img_tensor[0] * 255).astype(np.uint8)
        split = "val" if random.random() < VAL_RATIO else "train"
        img_filename = f"{split}_{i:06d}.png"
        img_path = dirs[split] / img_filename
        cv2.imwrite(str(img_path), img_np)
        labels_file[split].write(f"{img_filename}\t{label}\n")
    labels_file["train"].close()
    labels_file["val"].close()
    print(f"\n20万张静态图片已完成！")
    print(f"👉 训练集路径: {dirs['train']}")
    print(f"👉 验证集路径: {dirs['val']}")
if __name__ == '__main__':
    build_static_dataset()