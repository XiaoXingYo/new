import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
def inspect_digits_dataset():
    npz_path = Path(r'D:\work\new\data\processed_digits.npz')
    output_dir = Path(r'D:\work\new\scripts\clean_samples_preview')
    output_dir.mkdir(parents=True, exist_ok=True)
    if not npz_path.exists():
        print(f"❌ 找不到数据文件: {npz_path}。请先运行 EMNIST 清洗打包脚本！")
        return
    print("🔍 正在加载数据，准备生成 0-9 抽样检查网格...")
    data = np.load(npz_path)
    for digit_label in data.files:
        images = data[digit_label]
        total_count = len(images)
        if total_count == 0:
            print(f"⚠️ 类别 '{digit_label}' 中没有数据，跳过。")
            continue
        sample_size = min(100, total_count)
        random_indices = random.sample(range(total_count), sample_size)
        sampled_images = images[random_indices]
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))
        fig.suptitle(f"Digit: {digit_label} | Randomly Sampled {sample_size} / {total_count}",
                     fontsize=18, fontweight='bold', y=0.98)
        for i, ax in enumerate(axes.flat):
            if i < sample_size:
                ax.imshow(sampled_images[i], cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        save_path = output_dir / f"check_digit_{digit_label}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"生成抽样报告: 数字 '{digit_label}' -> 保存于 {save_path.name}")
    data.close()
    print(f"\n0-9 全部抽查图片生成完毕！请前往 {output_dir} 进行人工复核。")
if __name__ == '__main__':
    inspect_digits_dataset()