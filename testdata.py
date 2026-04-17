import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 确保能找到 src 目录
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.config import Config
from src.data.generator import OCRDataGenerator


def main():
    # 加载配置
    cfg = Config("configs/base.yaml")

    # 初始化你的数据生成工厂
    print("🏭 正在启动数据生成工厂 (第一次可能需要下载EMNIST数据集，请耐心等待)...")
    generator = OCRDataGenerator(cfg)

    # 准备画图
    fig, axes = plt.subplots(4, 2, figsize=(10, 8))
    axes = axes.flatten()

    print("🎨 正在生成 8 张模拟手写图片...")
    for i in range(8):
        # 核心：调用你的生成器！
        img_np, label = generator.generate_sample()

        # img_np 的形状是 (1, 32, 128)，去掉多余的维度用来显示
        img_show = img_np.squeeze()

        # 画在图表上
        axes[i].imshow(img_show, cmap='gray')
        axes[i].set_title(f"Label: {label}", color='blue', fontsize=14)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    print("✅ 生成完毕！看看你的代码造出来的图片逼不逼真吧！")


if __name__ == "__main__":
    main()