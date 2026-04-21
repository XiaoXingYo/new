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
        self.morph_prob = getattr(config.data, 'morph_prob', 0.2)

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """极轻微的噪点：告别老电视机雪花，只保留零星杂讯"""
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, 0.03, image.shape)
            image = np.clip(image + noise, 0, 1)
        if random.random() < self.noise_prob:
            salt_mask = np.random.random(image.shape) < 0.003
            # 修复：之前 pepper 是 0.03 (3%的黑点太多了)，现在降为 0.003
            pepper_mask = np.random.random(image.shape) < 0.003
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
        """废弃：不再进行加粗或变细操作，保留原生真实笔迹"""
        return image

    def simulate_pen_skip(self, image: np.ndarray) -> np.ndarray:
        """微创级：模拟极轻微的笔画断裂 (Cutout)"""
        if random.random() < getattr(self.config.data, 'cutout_prob', 0.15):
            h, w = image.shape

            # 随机挖掉 1 到 2 个极小缺口
            num_holes = random.randint(1, 2)
            for _ in range(num_holes):
                # 缺口大小严格控制在 1~2 个像素，绝对不伤筋动骨
                hole_h = random.randint(1, 2)
                hole_w = random.randint(1, 2)

                y = random.randint(0, max(0, h - hole_h))
                x = random.randint(0, max(0, w - hole_w))

                image[y:y + hole_h, x:x + hole_w] = 0.0

        return image

    def random_perspective_stretch(self, image: np.ndarray) -> np.ndarray:
        """随机透视拉伸 (模拟局部被扯拽或倾斜)"""
        if random.random() < getattr(self.config.data, 'stretch_prob', 0.3):
            h, w = image.shape

            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

            max_shift_x = int(w * 0.15)
            max_shift_y = int(h * 0.15)

            pts2 = np.float32([
                [0 + random.randint(-max_shift_x, max_shift_x), 0 + random.randint(-max_shift_y, max_shift_y)],
                [w + random.randint(-max_shift_x, max_shift_x), 0 + random.randint(-max_shift_y, max_shift_y)],
                [0 + random.randint(-max_shift_x, max_shift_x), h + random.randint(-max_shift_y, max_shift_y)],
                [w + random.randint(-max_shift_x, max_shift_x), h + random.randint(-max_shift_y, max_shift_y)]
            ])

            M = cv2.getPerspectiveTransform(pts1, pts2)

            original_dtype = image.dtype
            img_cv = image.astype(np.float32)

            img_cv = cv2.warpPerspective(img_cv, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            image = img_cv.astype(original_dtype)

        return image

    def apply(self, image: np.ndarray) -> np.ndarray:
        """理顺流水线，杜绝重复破坏"""
        # 1. 模拟网格轻微形变
        image = self.elastic_transform(image)

        # 2. 全局提亮：拯救 EMNIST 里那些下笔太轻、发灰发虚的数字
        image = np.clip(image * 1.15, 0.0, 1.0)

        # 3. 模拟微创断笔
        image = self.simulate_pen_skip(image)

        # 4. 增加轻微噪点 (修复了之前连续调用两次的问题)
        image = self.add_noise(image)

        return image