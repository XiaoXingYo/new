import os
import cv2
import numpy as np
import shutil
from pathlib import Path


def is_clean_symbol(thresh_img, symbol_type):
    """
    基于连通区域分析(CCA)和空间特征排查不完整的“残疾”脏数据样本。
    """
    # 1. 基础像素检查：过滤掉空白或噪点极少的废图
    total_pixels = cv2.countNonZero(thresh_img)
    if total_pixels < 20:
        return False

    # 2. 计算连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img)
    if num_labels <= 1:
        return False

    # 提取所有非背景连通区域的面积并排序
    areas_with_labels = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)]
    areas_with_labels.sort(reverse=True, key=lambda x: x[0])

    sorted_areas = [area for area, label in areas_with_labels]
    num_components = len(sorted_areas)

    # 获取整个符号的全局外接矩形，用于计算长宽比
    x_all, y_all, w_all, h_all = cv2.boundingRect(thresh_img)
    aspect_ratio_all = w_all / float(h_all) if h_all > 0 else 1.0
    if symbol_type == 'plus':
        # 1. 收紧长宽比拦截 (专杀瘦长的 't', 'f' 和极扁的乱线)
        if aspect_ratio_all < 0.6 or aspect_ratio_all > 1.6:
            return False
        if sorted_areas[0] / total_pixels < 0.75:
            return False
        # # 2. 拓扑学孔洞探测 (专杀 '4', 'p' 等闭合图形)
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None and any(h[3] != -1 for h in hierarchy[0]):
            return False

        roi = thresh_img[y_all:y_all + h_all, x_all:x_all + w_all]
        row_sum = np.sum(roi, axis=1)
        col_sum = np.sum(roi, axis=0)
        max_row_idx = int(np.argmax(row_sum))
        max_col_idx = int(np.argmax(col_sum))
        # # 4. 四角绝对留白校验
        corner_w = max(1, int(w_all * 0.25))
        corner_h = max(1, int(h_all * 0.25))
        tl = roi[:corner_h, :corner_w]
        tr = roi[:corner_h, -corner_w:]
        bl = roi[-corner_h:, :corner_w]
        br = roi[-corner_h:, -corner_w:]
        corner_pixels = cv2.countNonZero(tl) + cv2.countNonZero(tr) + \
                        cv2.countNonZero(bl) + cv2.countNonZero(br)
        if corner_pixels > total_pixels * 0.15:
            return False
        #
        channel_w = max(3, int(w_all * 0.15))
        half_w = channel_w // 2

        up_branch = roi[:max_row_idx, max(0, max_col_idx - half_w): min(w_all, max_col_idx + half_w + 1)]
        down_branch = roi[max_row_idx:, max(0, max_col_idx - half_w): min(w_all, max_col_idx + half_w + 1)]

        channel_h = max(3, int(h_all * 0.15))
        half_h = channel_h // 2

        left_branch = roi[max(0, max_row_idx - half_h): min(h_all, max_row_idx + half_h + 1), :max_col_idx]
        right_branch = roi[max(0, max_row_idx - half_h): min(h_all, max_row_idx + half_h + 1), max_col_idx:]

        # 辅助函数：计算分支的实际延伸长度
        def get_valid_length(branch_img, axis):
            if cv2.countNonZero(branch_img) == 0:
                return 0
            coords = cv2.findNonZero(branch_img)
            x, y, w, h = cv2.boundingRect(coords)
            return h if axis == 0 else w

        len_up = get_valid_length(up_branch, 0)
        len_down = get_valid_length(down_branch, 0)
        len_left = get_valid_length(left_branch, 1)
        len_right = get_valid_length(right_branch, 1)
        if len_up < h_all * 0.05 or len_down < h_all * 0.05:
            return False
        if len_left < w_all * 0.05 or len_right < w_all * 0.05:
            return False
        if min(len_up, len_down) / (max(len_up, len_down) + 1e-5) < 0.2:
            return False
        if min(len_left, len_right) / (max(len_left, len_right) + 1e-5) < 0.2:
            return False

    # 策略 B：减号 '-'
    elif symbol_type == 'minus':
        if sorted_areas[0] / total_pixels < 0.8:
            return False
        if aspect_ratio_all < 1:
            return False

        # 矩形填充率 (Extent) 校验
        extent = total_pixels / (w_all * h_all)
        if extent < 0.13:
            return False

        # 核心行穿透校验 (专杀上下凹凸弯曲线)
        roi = thresh_img[y_all:y_all + h_all, x_all:x_all + w_all]
        center_y = h_all // 2
        center_band = roi[max(0, center_y - 1):min(h_all, center_y + 2), :]
        band_projection = np.max(center_band, axis=0)
        band_coverage = np.count_nonzero(band_projection)
        if band_coverage < w_all * 0.25:
            return False

    # 策略 C：乘号 'times'
    elif symbol_type == 'times':
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None and any(h[3] != -1 for h in hierarchy[0]):
            return False

        # 1. 长宽比重新收紧 (专杀“狗骨头”和极扁乱线)
        if aspect_ratio_all < 0.45 or aspect_ratio_all > 1.75:
            return False

        # 矩形填充率 (Extent) 收紧
        extent = total_pixels / (w_all * h_all)
        if extent > 0.48:
            return False

        # 凸包实心率 (Solidity) 收紧
        if contours:
            main_cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(main_cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = total_pixels / hull_area
                if solidity > 0.58:
                    return False

        if num_components > 3:
            return False

        if num_components >= 2:
            top2_area = sorted_areas[1]
            if top2_area / sorted_areas[0] < 0.15:
                return False
            if num_components >= 3 and sorted_areas[2] > sorted_areas[0] * 0.15:
                return False
            label1 = areas_with_labels[0][1]
            label2 = areas_with_labels[1][1]
            x1, y1, w1, h1, _ = stats[label1]
            x2, y2, w2, h2, _ = stats[label2]
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            if overlap_x == 0 or overlap_y == 0:
                return False

        # 九宫格空间形态特征矩阵
        roi = thresh_img[y_all:y_all + h_all, x_all:x_all + w_all]
        h3, w3 = max(1, int(h_all / 3)), max(1, int(w_all / 3))
        tm = roi[:h3, w3:2 * w3]
        bm = roi[-h3:, w3:2 * w3]
        lm = roi[h3:2 * h3, :w3]
        rm = roi[h3:2 * h3, -w3:]
        edge_pixels = cv2.countNonZero(tm) + cv2.countNonZero(bm) + \
                      cv2.countNonZero(lm) + cv2.countNonZero(rm)

        tl = roi[:h3, :w3]
        tr = roi[:h3, -w3:]
        bl = roi[-h3:, :w3]
        br = roi[-h3:, -w3:]
        corners = [cv2.countNonZero(tl), cv2.countNonZero(tr),
                   cv2.countNonZero(bl), cv2.countNonZero(br)]
        corner_pixels = sum(corners)

        if edge_pixels > corner_pixels * 1.2:
            return False
        if min(corners) < max(corners) * 0.05:
            return False
        center_cell = roi[h3:2 * h3, w3:2 * w3]
        if cv2.countNonZero(center_cell) < total_pixels * 0.06:
            return False
        # 1. 形态学提取骨架 (Skeletonization)
        skel = np.zeros(thresh_img.shape, np.uint8)
        temp_img = thresh_img.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(temp_img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(temp_img, temp)
            skel = cv2.bitwise_or(skel, temp)
            temp_img = eroded.copy()
            if cv2.countNonZero(temp_img) == 0:
                break

        # 2. 卷积核统计邻域，寻找端点
        skel_bin = (skel > 0).astype(np.uint8)

        # 设计一个 3x3 的卷积核：中心权重设为 10，周围 8 邻域权重设为 1
        # 这样卷积后，某像素点的值如果是 11，说明它是骨架的一部分(10)，且只有一个邻居(1)，即为“端点”
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]], dtype=np.uint8)

        neighbor_counts = cv2.filter2D(skel_bin, -1, kernel)

        # 统计端点数量
        num_endpoints = np.sum(neighbor_counts == 11)

        # 一个完美的乘号(x)应该有 4 个端点。
        # 考虑到手写连笔的粘连，如果端点数 < 3，说明它极有可能是一根没产生交叉的乱线（比如图3）
        if num_endpoints < 3:
            return False

    # 策略 D：等号 '='
    elif symbol_type == 'eq':
        if num_components < 2:
            return False
        top1_area = sorted_areas[0]
        top2_area = sorted_areas[1]
        # 1. 整体面积占比
        if (top1_area + top2_area) / total_pixels < 0.85:
            return False

        # 2. 上下两笔的面积均衡度 (专杀图3这种一粗一细、一长一短的)
        # 将阈值从 0.45 严格提升到 0.55
        if top2_area / top1_area < 0.55:
            return False

        if num_components >= 3:
            top3_area = sorted_areas[2]
            if top3_area > top1_area * 0.15:
                return False

        label1 = areas_with_labels[0][1]
        label2 = areas_with_labels[1][1]
        x1, y1, w1, h1, _ = stats[label1]
        x2, y2, w2, h2, _ = stats[label2]

        # 3. 严格宽度比对齐
        # 将阈值从 0.5 提升到 0.65，上下两根线长度必须差不多
        if min(w1, w2) / max(w1, w2) < 0.6:
            return False
        # 4. 单笔画"扁平度"校验 (专杀图1、图2的弯钩)
        # 正常的等号笔画，宽度至少要是高度的2.5倍以上（之前是1.5，太容易放过Z字形了）
        if w1 < h1 * 2.1 or w2 < h2 * 2.1:
            return False

        # 6. X轴投影重叠度
        # 将重合率从 0.35 提升到 0.55，要求上下两根线必须对得很齐
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        if overlap_x < max(w1, w2) * 0.45:
            return False

        # 7. Y轴间距校验 (保持不变)
        top_y = min(y1, y2)
        bottom_y = max(y1, y2)
        top_line_h = h1 if y1 < y2 else h2
        clearance = bottom_y - (top_y + top_line_h)
        if clearance < min(h1, h2) * 0.2:
            return False
        if clearance > max(h1, h2) * 5.5:
            return False
        # 8. 整体长宽比 (稍微收紧下限)
        w_combined = max(x1 + w1, x2 + w2) - min(x1, x2)
        h_combined = max(y1 + h1, y2 + h2) - min(y1, y2)
        eq_aspect = w_combined / float(h_combined) if h_combined > 0 else 1.0

        if eq_aspect < 0.65 or eq_aspect > 3.5:
            return False
    # 策略 E：除号 'div'
    elif symbol_type == 'div':
        if num_components < 3 or num_components > 6:
            return False
        top3_area = sum(sorted_areas[:3])
        if top3_area / total_pixels < 0.85:
            return False

    return True
def process_and_pack():
    raw_dir = Path('../data/math_symbols')
    output_path = Path('../data/processed_symbols.npz')
    error_base_dir = Path(r'./error_samples')

    error_base_dir.mkdir(parents=True, exist_ok=True)

    symbol_folders = {
        '+': 'plus',
        '-': 'minus',
        'times': 'times',
        'div': 'div',
        '=': 'eq'
    }

    symbols_data = {}
    print("🚀 启动终极特征清洗与打包程序 (已挂载全部符号专属拦截网)...")

    for folder_name, save_key in symbol_folders.items():
        folder_path = raw_dir / folder_name
        if not folder_path.exists():
            print(f"⚠️ 找不到文件夹: {folder_path}，已跳过。")
            continue

        image_paths = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))[:2000]

        error_dir = error_base_dir / save_key
        if error_dir.exists():
            shutil.rmtree(error_dir)
        error_dir.mkdir(parents=True, exist_ok=True)

        processed_images = []
        skip_count = 0

        for img_path in image_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = 255 - img

            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel_pre = np.ones((2, 2), np.uint8)
            thresh = cv2.dilate(thresh, kernel_pre, iterations=1)

            # --- 🛡️ 脏数据排查 & 物理隔离 🛡️ ---
            if not is_clean_symbol(thresh, save_key):
                skip_count += 1
                error_img_path = error_dir / img_path.name
                shutil.copy(str(img_path), str(error_img_path))
                continue

            # --- 🎨 核心图像预处理 ---
            coords = cv2.findNonZero(thresh)
            if coords is None: continue
            x, y, w, h = cv2.boundingRect(coords)
            cropped = thresh[y:y + h, x:x + w]

            max_side = max(w, h)
            scale = 20.0 / max_side
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

            kernel_post = np.ones((2, 2), np.uint8)
            thickened = cv2.dilate(resized, kernel_post, iterations=1)

            smoothed = cv2.GaussianBlur(thickened, (3, 3), 0)
            brightened = np.clip(smoothed.astype(np.float32) * 2.5, 0, 255).astype(np.uint8)

            canvas = np.zeros((28, 28), dtype=np.uint8)
            start_y, start_x = (28 - new_h) // 2, (28 - new_w) // 2
            canvas[start_y:start_y + new_h, start_x:start_x + new_w] = brightened

            canvas_float = canvas.astype(np.float32) / 255.0
            processed_images.append(canvas_float)

        symbols_data[save_key] = np.array(processed_images)
        print(f"✅ '{save_key}' 处理完毕。成功保留 {len(processed_images)} 张，剔除了 {skip_count} 张脏数据。")

    if symbols_data:
        np.savez(output_path, **symbols_data)
        print(f"\n🎉 完美匹配版纯净字库已生成至: {output_path}")
        print(f"📂 隔离的残疾/脏数据样本已保存至: {error_base_dir}，可随时人工复核。")
    else:
        print("\n❌ 警告：没有处理任何数据，请检查原始数据路径是否正确。")


if __name__ == '__main__':
    process_and_pack()