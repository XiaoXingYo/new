import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
def inspect_dataset():
    npz_path = Path('../data/processed_symbols.npz')
    output_dir = Path(r'./inspection')
    output_dir.mkdir(parents=True, exist_ok=True)

    if not npz_path.exists():
        print(f"❌ 找不到数据文件: {npz_path}。请先运行清洗打包脚本！")
        return
    print("🔍 正在加载数据，准备生成抽样检查网格...")
    data = np.load(npz_path)
    for symbol_name in data.files:
        images = data[symbol_name]
        total_count = len(images)
        if total_count == 0:
            print(f"⚠️ 类别 '{symbol_name}' 中没有数据，跳过。")
            continue
        sample_size = min(100, total_count)
        random_indices = random.sample(range(total_count), sample_size)
        sampled_images = images[random_indices]
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))
        fig.suptitle(f"Class: {symbol_name} | Randomly Sampled {sample_size} / {total_count}", fontsize=16, y=0.98)
        for i, ax in enumerate(axes.flat):
            if i < sample_size:
                ax.imshow(sampled_images[i], cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        save_path = output_dir / f"check_{symbol_name}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"生成抽样报告: {symbol_name} -> 保存于 {save_path}")
    data.close()
    print("\n全部抽查图片生成完毕！请前往文件夹进行人工复核。")


if __name__ == '__main__':
    inspect_dataset()