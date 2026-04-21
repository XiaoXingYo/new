import os
import cv2
import numpy as np
from pathlib import Path

def process_and_pack():
    raw_dir = Path('../data/math_symbols')
    output_path = Path('../data/processed_symbols.npz')

    symbol_folders = {
        '+': 'plus',
        '-': 'minus',
        'times': 'times',
        'div': 'div',
        '=': 'eq'
    }

    symbols_data = {}
    print("🧹 终极清洗：匹配 EMNIST 的绝对笔画粗细与亮度...")

    for folder_name, save_key in symbol_folders.items():
        folder_path = raw_dir / folder_name
        image_paths = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))[:2000]

        processed_images = []
        for img_path in image_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = 255 - img

            # 1. 基础二值化提取
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 2. 原图轻微膨胀
            kernel_pre = np.ones((2, 2), np.uint8)
            thresh = cv2.dilate(thresh, kernel_pre, iterations=1)

            coords = cv2.findNonZero(thresh)
            if coords is None: continue
            x, y, w, h = cv2.boundingRect(coords)
            cropped = thresh[y:y+h, x:x+w]

            # 3. 缩放到目标大小 (约 20x20)
            max_side = max(w, h)
            scale = 20.0 / max_side
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 4. 缩放后二次膨胀 (匹配 EMNIST 的粗细)
            kernel_post = np.ones((2, 2), np.uint8)
            thickened = cv2.dilate(resized, kernel_post, iterations=1)

            # 5. 先做平滑去锯齿 (注意：这次把平滑提前了)
            smoothed = cv2.GaussianBlur(thickened, (3, 3), 0)

            # 6. 🌟 暴力提亮：像素值翻倍，强行拉到纯白！🌟
            # 乘以 2.5 可以把被模糊变灰的主干笔画直接顶满到 255，同时由于 clip 的存在，能保留微弱的边缘平滑
            brightened = np.clip(smoothed.astype(np.float32) * 2.5, 0, 255).astype(np.uint8)

            # 7. 贴到 28x28 画布中心 (贴上去之后绝对不再做任何模糊操作)
            canvas = np.zeros((28, 28), dtype=np.uint8)
            start_y, start_x = (28 - new_h) // 2, (28 - new_w) // 2
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = brightened

            # 归一化存储
            canvas_float = canvas.astype(np.float32) / 255.0
            processed_images.append(canvas_float)

        symbols_data[save_key] = np.array(processed_images)
        print(f"✅ '{save_key}' 加粗拉亮处理完毕。")

    np.savez(output_path, **symbols_data)
    print("🎉 最终完美匹配版字库已生成！")

if __name__ == '__main__':
    process_and_pack()