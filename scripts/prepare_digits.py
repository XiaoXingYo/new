import os
import cv2
import numpy as np
import struct
import shutil
from pathlib import Path
def is_clean_digit(thresh_img, digit_label):
    """
    针对 0-9 手写数字的拓扑与物理常识清洗防御网。
    终极版：融合了所有专属拦截特征与误杀豁免机制。
    """
    total_pixels = cv2.countNonZero(thresh_img)
    if total_pixels < 25:
        return False, "TooFewPixels"
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img)
    if num_labels <= 1:
        return False, "NoComponents"
    valid_components = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > total_pixels * 0.1)
    if valid_components > 3:
        return False, "FragmentedStrokes"
    x_all, y_all, w_all, h_all = cv2.boundingRect(thresh_img)
    aspect_ratio_all = w_all / float(h_all) if h_all > 0 else 1.0
    extent = total_pixels / (w_all * h_all)
    if digit_label != '1' and extent > 0.78:
        return False, "SolidBlackBlock"
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    num_holes = 0
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1 and cv2.contourArea(contours[i]) > 5:
                num_holes += 1
    if digit_label == '1':
        if aspect_ratio_all > 0.65:
            if extent > 0.4:
                return False, "TooWide_And_Solid"
        if aspect_ratio_all < 0.1: return False, "TooNarrow"
        if num_holes >= 1: return False, "HasHoles"
        h_third = h_all // 3
        if h_third > 2:
            top_roi = thresh_img[y_all: y_all + h_third, x_all: x_all + w_all]
            mid_roi = thresh_img[y_all + h_third: y_all + 2 * h_third, x_all: x_all + w_all]
            bot_roi = thresh_img[y_all + 2 * h_third: y_all + h_all, x_all: x_all + w_all]
            def get_cx(roi):
                M = cv2.moments(roi)
                return (M["m10"] / M["m00"]) if M["m00"] != 0 else (w_all / 2.0)
            cx_top = get_cx(top_roi)
            cx_mid = get_cx(mid_roi)
            cx_bot = get_cx(bot_roi)
            expected_cx_mid = (cx_top + cx_bot) / 2.0
            if abs(cx_mid - expected_cx_mid) > w_all * 0.15:
                return False, "SevereBend_Or_CShape"
    elif digit_label == '0':
        if num_holes >= 2: return False, "Holes>=2"
        ext_contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(ext_contours) > 0:
            main_contour = max(ext_contours, key=cv2.contourArea)
            ext_area = cv2.contourArea(main_contour)
            if ext_area < total_pixels:
                return False, "UnclosedLoop_Or_SolidBlob"
        center_h_start = int(h_all * 0.45)
        center_h_end = int(h_all * 0.55)
        center_w_start = int(w_all * 0.45)
        center_w_end = int(w_all * 0.55)
        if center_h_end > center_h_start and center_w_end > center_w_start:
            center_roi = thresh_img[y_all + center_h_start: y_all + center_h_end,
            x_all + center_w_start: x_all + center_w_end]
            center_pixels = cv2.countNonZero(center_roi)
            center_area = (center_h_end - center_h_start) * (center_w_end - center_w_start)
            if center_pixels / center_area > 0.70:
                return False, "SlashedCenter"
    elif digit_label == '2':
        if num_holes >= 2: return False, "Holes>=2"
        if aspect_ratio_all > 1.2: return False, "TooWide"
        if aspect_ratio_all < 0.35: return False, "TooNarrow"
        left_profile = np.zeros(h_all, dtype=int)
        for y_idx in range(h_all):
            row = thresh_img[y_all + y_idx, x_all: x_all + w_all]
            nz = np.nonzero(row)[0]
            left_profile[y_idx] = nz[0] if len(nz) > 0 else w_all
        min_left_x = np.min(left_profile)
        leftmost_y_indices = np.where(left_profile <= min_left_x + w_all * 0.08)[0]
        if len(leftmost_y_indices) > 0:
            is_top_left = np.any(leftmost_y_indices < h_all * 0.35)
            is_bot_left = np.any(leftmost_y_indices > h_all * 0.65)
            if not (is_top_left or is_bot_left):
                return False, "MidLeftBulge_Z_or_Alpha"
        bottom_y_start = int(h_all * 0.75)
        if bottom_y_start < h_all:
            bottom_roi = thresh_img[y_all + bottom_y_start: y_all + h_all, x_all: x_all + w_all]
            bottom_coords = cv2.findNonZero(bottom_roi)
            if bottom_coords is not None:
                bx, by, bw, bh = cv2.boundingRect(bottom_coords)
                if bw < w_all * 0.45:
                    return False, "NarrowBase_BirdMimic"
                col_sums = np.sum(bottom_roi > 0, axis=0)
                nz_cols = np.nonzero(col_sums)[0]
                if len(nz_cols) > 0:
                    first_col, last_col = nz_cols[0], nz_cols[-1]
                    max_gap = 0
                    current_gap = 0
                    for c in range(first_col, last_col + 1):
                        if col_sums[c] == 0:
                            current_gap += 1
                            max_gap = max(max_gap, current_gap)
                        else:
                            current_gap = 0
                    if max_gap > w_all * 0.15:
                        return False, "SplitBase_AlphaMimic"
        top_y_end = int(h_all * 0.25)
        if top_y_end > 0:
            top_roi = thresh_img[y_all: y_all + top_y_end, x_all: x_all + w_all]
            top_coords = cv2.findNonZero(top_roi)
            if top_coords is not None:
                tx, ty, tw, th = cv2.boundingRect(top_coords)
                if tw < w_all * 0.25:
                    return False, "TopArchTooNarrow"
    elif digit_label == '3':
        if aspect_ratio_all < 0.49: return False, "TooNarrow"
        if num_holes >= 2: return False, "Holes>=2"
        if num_holes == 1 and hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1 and cv2.contourArea(contours[i]) > 5:
                    hx, hy, hw, hh = cv2.boundingRect(contours[i])
                    hole_center_y = hy + hh / 2.0
                    height_ratio = hh / h_all
                    if height_ratio > 0.45:
                        continue
                    if hole_center_y < y_all + h_all * 0.30:
                        return False, "TopLoop"
                    if hole_center_y > y_all + h_all * 0.70:
                        return False, "BottomLoop"
        left_profile = np.zeros(h_all, dtype=int)
        for y_idx in range(h_all):
            row = thresh_img[y_all + y_idx, x_all: x_all + w_all]
            nz = np.nonzero(row)[0]
            left_profile[y_idx] = nz[0] if len(nz) > 0 else w_all
        h_10, h_45 = int(h_all * 0.1), int(h_all * 0.45)
        h_55, h_90 = int(h_all * 0.55), int(h_all * 0.9)
        h_30, h_70 = int(h_all * 0.3), int(h_all * 0.7)
        top_slice = left_profile[h_10:h_45]
        bot_slice = left_profile[h_55:h_90]
        mid_slice = left_profile[h_30:h_70]
        top_min_x = np.min(top_slice) if len(top_slice) > 0 else 0
        bot_min_x = np.min(bot_slice) if len(bot_slice) > 0 else 0
        mid_min_x = np.min(mid_slice) if len(mid_slice) > 0 else 0
        top_y_rel = np.argmin(top_slice) + h_10 if len(top_slice) > 0 else h_10
        bot_y_rel = np.argmin(bot_slice) + h_55 if len(bot_slice) > 0 else h_90
        if mid_min_x < top_min_x - w_all * 0.13 and mid_min_x < bot_min_x - w_all * 0.13:
            return False, "MiddleBulgeLeft_Zigzag"
    elif digit_label == '4':
        if aspect_ratio_all < 0.45: return False, "TooNarrow"
        if aspect_ratio_all > 1.2:
            if extent < 0.4 and num_holes != 1:
                return False, "TooWideAndThin"
        if num_holes == 1 and hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1 and cv2.contourArea(contours[i]) > 5:
                    hx, hy, hw, hh = cv2.boundingRect(contours[i])
                    hole_center_y = hy + hh / 2.0
                    if hole_center_y > y_all + h_all * 0.52:
                        return False, "HoleTooLow"
        bottom_y_start = int(h_all * 0.75)
        if bottom_y_start < h_all:
            bottom_roi = thresh_img[y_all + bottom_y_start: y_all + h_all, x_all: x_all + w_all]
            bottom_coords = cv2.findNonZero(bottom_roi)
            if bottom_coords is not None and num_holes != 1:
                bx, by, bw, bh = cv2.boundingRect(bottom_coords)
                if bw > w_all * 0.55 :
                    return False, "WideBottomBase_UShape"
                bn_labels, blabels, bstats, bcentroids = cv2.connectedComponentsWithStats(bottom_roi)
                valid_bottom_parts = sum(1 for i in range(1, bn_labels) if bstats[i, cv2.CC_STAT_AREA] > 5)
                if valid_bottom_parts > 1:
                    return False, "MultipleLegs"
        top_y_end = int(h_all * 0.3)
        if top_y_end > 0:
            top_roi = thresh_img[y_all: y_all + top_y_end, x_all: x_all + w_all]
            top_coords = cv2.findNonZero(top_roi)
            if top_coords is not None:
                tx, ty, tw, th = cv2.boundingRect(top_coords)
                if tx + tw < w_all * 0.55:
                    return False, "MissingTopRightStem_ChairMimic"
        col_ink_counts = np.sum(thresh_img[y_all:y_all + h_all, x_all:x_all + w_all] > 0, axis=0)
        if len(col_ink_counts) > 0:
            max_ink_col = np.max(col_ink_counts)
            main_stem_x_rel = np.argmax(col_ink_counts)
            if max_ink_col < h_all * 0.3:
                return False, "NoVerticalPillar_DiagonalMutant"
            if main_stem_x_rel < w_all * 0.35:
                right_half_pixels = np.sum(col_ink_counts[int(w_all * 0.5):])
                total_pixels = np.sum(col_ink_counts)
                right_pixel_ratio = right_half_pixels / float(total_pixels) if total_pixels > 0 else 0
                if right_pixel_ratio > 0.35:
                    pass
                else:
                    return False, "LeftWeighted_Mutant"
    elif digit_label == '5':
        if num_holes >= 2: return False, "Holes>=2"
        if aspect_ratio_all > 1.4: return False, "TooWide"
        if aspect_ratio_all < 0.3: return False, "TooNarrow"
    elif digit_label == '6':
        if num_holes == 0: return False, "UnclosedLoop"
        if num_holes >= 2: return False, "Holes>=2"
        if aspect_ratio_all > 1.2: return False, "TooWide"
        if aspect_ratio_all < 0.35: return False, "TooNarrow"
    elif digit_label == '7':
        # 1. 基础拓扑
        if num_holes >= 2: return False, "Holes>=2"
        if aspect_ratio_all > 1.5: return False, "TooWide"
        if aspect_ratio_all < 0.35: return False, "TooNarrow"
        top_y_end = max(1, int(h_all * 0.25))
        top_roi = thresh_img[y_all: y_all + top_y_end, x_all: x_all + w_all]
        w_third = w_all // 3
        if w_third > 0:
            left_region = top_roi[:, :w_third]
            mid_region = top_roi[:, w_third: 2 * w_third]
            left_nz = np.nonzero(left_region)
            mid_nz = np.nonzero(mid_region)
            if len(left_nz[0]) > 0 and len(mid_nz[0]) > 0:
                left_min_y = np.min(left_nz[0])
                mid_min_y = np.min(mid_nz[0])
                if left_min_y > mid_min_y + h_all * 0.08:
                    return False, "CandyCaneArchTop"
        top_coords = cv2.findNonZero(top_roi)
        if top_coords is not None:
            tx, ty, tw, th = cv2.boundingRect(top_coords)
            if tw < w_all * 0.45:
                return False, "TopBarTooShort"
            top_right_edge = tx + tw
            if top_right_edge < w_all * 0.85:
                return False, "RightElbowStickingOut"
        bottom_y_start = int(h_all * 0.8)
        if bottom_y_start < h_all:
            bottom_roi = thresh_img[y_all + bottom_y_start: y_all + h_all, x_all: x_all + w_all]
            bottom_coords = cv2.findNonZero(bottom_roi)
            if bottom_coords is not None:
                bx, by, bw, bh = cv2.boundingRect(bottom_coords)
                if bw > w_all * 0.5:
                    return False, "WideBottomBase"

    elif digit_label == '8':
        if aspect_ratio_all > 1.8: return False, "Ratio>1.8"
        if aspect_ratio_all < 0.35: return False, "TooNarrow"
        if num_holes >= 3: return False, "Holes>=3"
        if num_holes == 0: return False, "NoHolesBlob"
        if extent > 0.55: return False, "SolidBlob"
        bottom_y_start = int(h_all * 0.7)
        if bottom_y_start < h_all:
            bottom_roi = thresh_img[y_all + bottom_y_start: y_all + h_all, x_all: x_all + w_all]
            bottom_coords = cv2.findNonZero(bottom_roi)
            if bottom_coords is not None:
                bx, by, bw, bh = cv2.boundingRect(bottom_coords)
                if bw < w_all * 0.45 and num_holes!=2 and num_holes!=1:
                    return False, "StickBottom"
        top_y_end = int(h_all * 0.25)
        if top_y_end > 0:
            top_roi = thresh_img[y_all: y_all + top_y_end, x_all: x_all + w_all]
            M = cv2.moments(top_roi)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                if cX > w_all * 0.75 or cX < w_all * 0.25 and num_holes!=2 and num_holes!=1 :
                    return False, "WildAntenna"
    elif digit_label == '9':
        if num_holes == 0: return False, "UnclosedLoop"
        if num_holes >= 3: return False, "Holes>=3"
        if extent > 0.55: return False, "SolidBlob"
        if aspect_ratio_all > 1.2: return False, "TooWide"
        if aspect_ratio_all < 0.35: return False, "TooNarrow"
        if hierarchy is not None:
            max_hole_area = 0
            hole_center_y = 0
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1:
                    area = cv2.contourArea(contours[i])
                    if area > max_hole_area:
                        max_hole_area = area
                        hx, hy, hw, hh = cv2.boundingRect(contours[i])
                        hole_center_y = hy + hh / 2.0

            if max_hole_area > 5 and hole_center_y > y_all + h_all * 0.55:
                return False, "HoleTooLow"

    return True, "Valid"


def read_idx(filename):
    """解析 IDX 二进制文件"""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def process_and_pack_digits():
    # 配置输入输出路径
    base_dir = Path(r'D:\work\new\data\EMNIST\raw')
    output_path = Path(r'D:\work\new\data\processed_digits.npz')
    error_base_dir = Path(r'D:\work\new\scripts\error_samples_digits')
    images_path = base_dir / 'emnist-digits-train-images-idx3-ubyte'
    labels_path = base_dir / 'emnist-digits-train-labels-idx1-ubyte'
    print("📦 正在将 IDX 文件加载到内存 (EMNIST Digits 海量版)...")
    images = read_idx(images_path)
    labels = read_idx(labels_path)
    digits_data = {str(i): [] for i in range(10)}
    skip_counts = {str(i): 0 for i in range(10)}
    for i in range(10):
        error_dir = error_base_dir / str(i)
        if error_dir.exists():
            shutil.rmtree(error_dir)
        error_dir.mkdir(parents=True, exist_ok=True)
    print("🚀 启动 EMNIST 0-9 全量数字拓扑清洗与打包程序...")
    for i in range(len(labels)):
        label = labels[i]
        label_str = str(label)
        raw_img = images[i]
        img = raw_img.T
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        is_valid, reason = is_clean_digit(thresh, label_str)
        if not is_valid:
            skip_counts[label_str] += 1
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            comparison_img = np.hstack((img_color, thresh_color))
            error_img_path = error_base_dir / label_str / f"idx{i}_{reason}.jpg"
            cv2.imwrite(str(error_img_path), comparison_img)
            continue
        canvas_float = img.astype(np.float32) / 255.0
        digits_data[label_str].append(canvas_float)
    print("\n📊 EMNIST Digits 数据清洗最终报告：")
    for k in range(10):
        label_str = str(k)
        valid_len = len(digits_data[label_str])
        print(f"✅ 数字 '{label_str}': 成功保留 {valid_len:>5} 张，剔除 {skip_counts[label_str]:>4} 张脏数据。")
    final_data = {k: np.array(v) for k, v in digits_data.items() if len(v) > 0}
    if final_data:
        np.savez(output_path, **final_data)
        print(f"\n🎉 纯净版全量 0-9 数字字库已生成至: {output_path}")
        print(f"📂 隔离的脏数据已保存至: {error_base_dir}，可随时人工复核。")
    else:
        print("\n❌ 警告：未提取到任何有效数据。请检查 IDX 文件路径是否正确指向了 emnist-digits。")
if __name__ == '__main__':
    process_and_pack_digits()