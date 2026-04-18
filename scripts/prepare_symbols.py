import os
import glob
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path


def process_and_pack():
    # 严格限定目录，绝不乱扔文件
    raw_dir = Path('../data/math_symbols')
    output_path = Path('../data/processed_symbols.npz')

    # 映射字典：文件夹名 -> 对应的字符
    symbol_folders = {
        '+': '+',
        '-': '-',
        'times': '*',
        'div': '/',
        '=': '='
    }

    if not raw_dir.exists():
        print(f"❌ 找不到目录 {raw_dir}。请先将Kaggle下载的图片放入该目录。")
        return

    symbols_data = {}
    print("🧹 开始清洗数学符号数据 (黑底白字、尺寸统一)...")

    for folder_name, char_label in symbol_folders.items():
        folder_path = raw_dir / folder_name
        # 寻找 jpg 或 png
        image_paths = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))
        image_paths = image_paths[:2000]  # 限制数量，防止内存爆炸

        if not image_paths:
            print(f"⚠️ 找不到文件夹或图片: {folder_path}，跳过。")
            continue

        processed_images = []
        for img_path in image_paths:
            # 1. 灰度化
            img = Image.open(img_path).convert('L')
            # 2. 颜色反转：白底黑字 -> 黑底白字 (严格对齐 EMNIST)
            img = ImageOps.invert(img)
            # 3. 尺寸统一到 28x28
            img = img.resize((28, 28), Image.Resampling.LANCZOS)

            # 4. 二值化提纯
            arr = np.array(img, dtype=np.uint8)
            arr[arr < 50] = 0

            # 转为 float32 并归一化到 0~1 (对齐你 emnist_source.py 的逻辑)
            arr = arr.astype(np.float32) / 255.0
            processed_images.append(arr)

        symbols_data[folder_name] = np.array(processed_images)
        print(f"✅ 符号 '{char_label}' 处理完毕，共 {len(processed_images)} 张。")

    # 打包保存到规定的目录下
    np.savez(output_path, **symbols_data)
    print(f"🎉 全部清洗打包完毕！轻量级字库已生成: {output_path}")


if __name__ == '__main__':
    process_and_pack()