import torch
from torch.utils.data import Dataset, DataLoader
from .generator import OCRDataGenerator


# ==========================================
# 🟢 路线 A：CTC 专属翻译官
# ==========================================
class LabelConverter:
    """处理 CRNN+CTC 的标签转换"""

    def __init__(self, chars: str, blank_label: int):
        self.chars = chars
        self.blank_label = blank_label
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for idx, char in enumerate(chars)}

    def encode(self, text_list):
        # CTC 的标签是变长的 1D Tensor，配合 target_lengths 使用
        length = [len(s) for s in text_list]
        targets = []
        for text in text_list:
            targets.extend([self.char2idx[c] for c in text])
        return torch.tensor(targets, dtype=torch.long), torch.tensor(length, dtype=torch.long)

    def decode(self, text_idx, length=None):
        if length is not None:
            # 批量解码
            texts = []
            index = 0
            for l in length:
                t = text_idx[index:index + l]
                char_list = []
                for i in range(l):
                    val = t[i].item() if hasattr(t[i], 'item') else t[i]
                    if val != self.blank_label:
                        prev_val = t[i - 1].item() if hasattr(t[i - 1], 'item') else t[i - 1]
                        if i == 0 or val != prev_val:
                            # 使用 get 防止越界报错，未知字符标为 ?
                            char_list.append(self.idx2char.get(val, '?'))
                texts.append(''.join(char_list))
                index += l
            return texts
        else:
            # 单条解码
            char_list = []
            for i in range(len(text_idx)):
                val = text_idx[i].item() if hasattr(text_idx[i], 'item') else text_idx[i]
                if val != self.blank_label:
                    prev_val = text_idx[i - 1].item() if hasattr(text_idx[i - 1], 'item') else text_idx[i - 1]
                    if i == 0 or val != prev_val:
                        char_list.append(self.idx2char.get(val, '?'))
            return ''.join(char_list)


# ==========================================
# 🔴 路线 B：Attention 专属翻译官
# ==========================================
class AttentionLabelConverter:
    """处理 Seq2Seq+Attention 的标签转换"""

    def __init__(self, chars: str, max_seq_len: int = 12):
        self.chars = chars
        self.max_seq_len = max_seq_len

        # 定义三大护法控制符的索引
        self.pad_idx = len(chars)  # 填充符 <PAD>
        self.eos_idx = len(chars) + 1  # 结束符 <EOS>
        self.sos_idx = len(chars) + 2  # 开始符 <SOS>

        self.num_classes = len(chars) + 3

        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for idx, char in enumerate(chars)}

    def encode(self, text_list):
        """把文字标签变成张量，比如 '123' 变成 [1, 2, 3, EOS, PAD, PAD...]"""
        batch_size = len(text_list)
        targets = torch.full((batch_size, self.max_seq_len), self.pad_idx, dtype=torch.long)

        for i, text in enumerate(text_list):
            text = text[:self.max_seq_len - 1]
            for j, char in enumerate(text):
                targets[i, j] = self.char2idx[char]
            targets[i, len(text)] = self.eos_idx

        return targets

    def decode(self, pred_idx):
        """把模型的输出数字翻译回人类文字"""
        result = []
        for seq in pred_idx:
            text = ""
            for idx in seq:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                if idx == self.eos_idx:  # 遇到结束符，立刻停止翻译！
                    break
                if idx < len(self.chars):
                    text += self.idx2char[idx]
            result.append(text)
        return result


# ==========================================
# 📊 数据集与加载器构建
# ==========================================
class OCRDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.generator = OCRDataGenerator(config)
        # 根据你的配置，算出一个 epoch 大概要跑多少张图
        self.length = config.train.epochs * 1000 if is_train else 500

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_np, label = self.generator.generate_sample()
        img_tensor = torch.from_numpy(img_np).float()

        # ⚡️ 核心修复：维度自适应
        # 如果 generator 返回的是 (32, 128) 二维灰度图，加上通道维度变成 (1, 32, 128)
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0)
        # 如果它已经是 (1, 32, 128) 或类似的三维矩阵了，就什么都不做，防止维度爆炸
        elif img_tensor.dim() == 3 and img_tensor.shape[2] == 1:
            # 如果形状是 (32, 128, 1) 这种通道在最后的，把它挪到前面变成 (1, 32, 128)
            img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor, label


def build_dataloaders(config):
    """注意：这里返回的标签还是纯字符串，真正的 Tensor 转换交给了 Engine！"""
    train_dataset = OCRDataset(config, is_train=True)
    val_dataset = OCRDataset(config, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader