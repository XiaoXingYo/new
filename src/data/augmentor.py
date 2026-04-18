import numpy as np
import cv2
import random
import scipy.ndimage as ndi


class OCRAugmentor:
    """OCR专用数据增强"""

    def __init__(self, config):
        self.config = config
        self.noise_prob = config.data.noise_prob
        self.elastic_prob = getattr(config.data, 'elastic_prob', 0.3)
        # ⚡️ 新增：形态学变换概率 (默认 0.4 触发率)
        self.morph_prob = getattr(config.data, 'morph_prob', 0.4)

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, 0.05, image.shape)
            image = np.clip(image + noise, 0, 1)
        if random.random() < self.noise_prob:
            salt_mask = np.random.random(image.shape) < 0.01
            pepper_mask = np.random.random(image.shape) < 0.01
            image[salt_mask] = 1.0
            image[pepper_mask] = 0.0
        return image

    def elastic_transform(self, image: np.ndarray) -> np.ndarray:
        """真正的弹性形变 (基于像素网格的局部扭曲)"""
        if random.random() < self.elastic_prob:
            alpha = image.shape[1] * 0.2  # 形变强度
            sigma = image.shape[1] * 0.08  # 平滑度

            random_state = np.random.RandomState(None)
            shape = image.shape

            dx = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dy = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

            image = ndi.map_coordinates(image, indices, order=1, mode='constant', cval=0.0).reshape(shape)
        return image

    def random_morphology(self, image: np.ndarray) -> np.ndarray:
        """⚡️ 新增：随机形态学变换 (模拟笔画变粗或变细)"""
        if random.random() < self.morph_prob:
            # 随机选择核大小 (2x2 是轻微粗细变化，3x3 是极限胖瘦)
            kernel_size = random.choice([2, 3])
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # OpenCV 对 float64 兼容性不好，统一转成 float32 处理
            original_dtype = image.dtype
            img_cv = image.astype(np.float32)

            if random.random() > 0.5:
                # 膨胀 (Dilation)：如果是黑底白字，会让白字变粗；如果是白底黑字，会让白底变粗(黑字变细)
                img_cv = cv2.dilate(img_cv, kernel, iterations=1)
            else:
                # 腐蚀 (Erosion)：如果是黑底白字，会让白字变细；如果是白底黑字，会让黑字变粗
                img_cv = cv2.erode(img_cv, kernel, iterations=1)

            # 转回原来的数据类型
            image = img_cv.astype(original_dtype)

        return image
    def simulate_pen_skip(self, image: np.ndarray) -> np.ndarray:
        """⚡️ 新增：模拟笔画断裂与未闭合 (Cutout)"""
        # 默认 40% 的概率触发断笔模拟
        if random.random() < getattr(self.config.data, 'cutout_prob', 0.4):
            h, w = image.shape

            # 随机挖掉 1 到 3 个小缺口
            num_holes = random.randint(1, 3)
            for _ in range(num_holes):
                # 缺口大小控制在 2~5 个像素，刚好能切断一根笔画的宽度
                hole_h = random.randint(2, 5)
                hole_w = random.randint(2, 5)

                # 随机选择擦除的起始坐标
                y = random.randint(0, h - hole_h)
                x = random.randint(0, w - hole_w)

                # 将该区域强行涂黑（你的背景是 0.0）
                image[y:y + hole_h, x:x + hole_w] = 0.0

        return image

    def random_perspective_stretch(self, image: np.ndarray) -> np.ndarray:
        """⚡️ 新增：随机透视拉伸 (模拟局部被扯拽或倾斜)"""
        # 默认 30% 的概率触发拉伸
        if random.random() < getattr(self.config.data, 'stretch_prob', 0.3):
            h, w = image.shape

            # 原始图片的四个角：左上, 右上, 左下, 右下
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

            # 定义拉伸的最大幅度 (比如最多扯出去图像宽度的 15%)
            max_shift_x = int(w * 0.15)
            max_shift_y = int(h * 0.15)

            # 给四个角加上随机的偏移量 (模拟人类写字的各种畸变)
            pts2 = np.float32([
                [0 + random.randint(-max_shift_x, max_shift_x), 0 + random.randint(-max_shift_y, max_shift_y)],  # 左上角乱飞
                [w + random.randint(-max_shift_x, max_shift_x), 0 + random.randint(-max_shift_y, max_shift_y)],  # 右上角乱飞
                [0 + random.randint(-max_shift_x, max_shift_x), h + random.randint(-max_shift_y, max_shift_y)],  # 左下角乱飞
                [w + random.randint(-max_shift_x, max_shift_x), h + random.randint(-max_shift_y, max_shift_y)]  # 右下角乱飞
            ])

            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(pts1, pts2)

            # OpenCV 处理需要 float32
            original_dtype = image.dtype
            img_cv = image.astype(np.float32)

            # 应用变换，超出边界的部分用 0.0 (纯黑背景) 填充
            img_cv = cv2.warpPerspective(img_cv, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

            image = img_cv.astype(original_dtype)

        return image
    def apply(self, image: np.ndarray) -> np.ndarray:
        # 1. 模拟笔画粗细变化
        image = self.random_morphology(image)
        # 2. 模拟网格形变
        image = self.elastic_transform(image)
        # 3. 增加噪点
        image = self.add_noise(image)
        # 4. 模拟断笔缺口
        image = self.simulate_pen_skip(image)
        # 5. 增加噪点
        image = self.add_noise(image)

        return image

