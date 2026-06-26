import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from pathlib import Path
import torchvision.transforms as T


# ==========================================
# 🟢 路线 A：CTC 专属翻译官 (已升级：支持 Greedy/Beam Search 且大幅提速)
# ==========================================
class LabelConverter:
    def __init__(self, chars: str, blank_label: int):
        self.chars = chars
        self.blank_label = blank_label
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for idx, char in enumerate(chars)}
    def encode(self, text_list):
        lengths = [len(s) for s in text_list]
        targets = [self.char2idx[c] for text in text_list for c in text]
        return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
    def decode(self, model_output, length=None, decode_type='greedy', beam_size=10):
        if decode_type == 'greedy':
            return self._decode_greedy(model_output, length)
        elif decode_type == 'beam_search':
            return self._decode_beam_search(model_output, length, beam_size)
        else:
            raise ValueError("decode_type 必须是 'greedy' 或 'beam_search'")
    def _decode_greedy(self, text_idx, length=None):
        if hasattr(text_idx, 'tolist'):
            text_idx = text_idx.tolist()
        if length is not None:
            if hasattr(length, 'tolist'):
                length = length.tolist()
            texts = []
            index = 0
            for l in length:
                texts.append(self._greedy_single(text_idx[index: index + l]))
                index += l
            return texts
        else:
            return self._greedy_single(text_idx)
    def _greedy_single(self, seq):
        char_list = []
        for i, val in enumerate(seq):
            if val != self.blank_label:
                if i == 0 or val != seq[i - 1]:
                    char_list.append(self.idx2char.get(val, '?'))
        return ''.join(char_list)
    def _decode_beam_search(self, probs, length=None, beam_size=10):
        if torch.is_tensor(probs):
            probs = probs.detach().cpu().numpy()
        if length is not None:
            if hasattr(length, 'tolist'):
                length = length.tolist()
            texts = []
            for i, l in enumerate(length):
                valid_probs = probs[i, :l, :]
                texts.append(self._ctc_beam_search_single(valid_probs, beam_size))
            return texts
        else:
            if len(probs.shape) == 3:
                probs = probs[0]
            return self._ctc_beam_search_single(probs, beam_size)
    def _ctc_beam_search_single(self, probs, beam_size):
        T, num_classes = probs.shape
        beam = {tuple(): (1.0, 0.0)}
        for t in range(T):
            next_beam = defaultdict(lambda: (0.0, 0.0))
            for prefix, (p_b, p_nb) in beam.items():
                p_total = p_b + p_nb
                prob_blank = probs[t, self.blank_label]
                if prob_blank > 0:
                    n_p_b, n_p_nb = next_beam[prefix]
                    next_beam[prefix] = (n_p_b + p_total * prob_blank, n_p_nb)
                for c in range(num_classes):
                    if c == self.blank_label:
                        continue
                    prob_c = probs[t, c]
                    if prob_c == 0:
                        continue
                    prefix_extended = prefix + (c,)
                    n_p_b, n_p_nb = next_beam[prefix_extended]
                    if len(prefix) > 0 and c == prefix[-1]:
                        next_beam[prefix_extended] = (n_p_b, n_p_nb + p_b * prob_c)
                        n_p_b_old, n_p_nb_old = next_beam[prefix]
                        next_beam[prefix] = (n_p_b_old, n_p_nb_old + p_nb * prob_c)
                    else:
                        next_beam[prefix_extended] = (n_p_b, n_p_nb + p_total * prob_c)
            beam = dict(sorted(next_beam.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)[:beam_size])
        best_prefix = max(beam.keys(), key=lambda k: beam[k][0] + beam[k][1])
        return ''.join([self.idx2char.get(idx, '?') for idx in best_prefix])


class AttentionLabelConverter:
    """处理 Seq2Seq+Attention 的标签转换"""
    def __init__(self, chars: str, max_seq_len: int = 12):
        self.chars = chars
        self.max_seq_len = max_seq_len
        self.pad_idx = len(chars)
        self.eos_idx = len(chars) + 1
        self.sos_idx = len(chars) + 2
        self.num_classes = len(chars) + 3
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for idx, char in enumerate(chars)}
    def encode(self, text_list):
        batch_size = len(text_list)
        targets = torch.full((batch_size, self.max_seq_len), self.pad_idx, dtype=torch.long)
        for i, text in enumerate(text_list):
            text = text[:self.max_seq_len - 1]
            for j, char in enumerate(text):
                targets[i, j] = self.char2idx[char]
            targets[i, len(text)] = self.eos_idx
        return targets
    def decode(self, pred_idx):
        if hasattr(pred_idx, 'tolist'):
            pred_idx = pred_idx.tolist()
        result = []
        for seq in pred_idx:
            text = ""
            for idx in seq:
                if idx == self.eos_idx:
                    break
                if idx < len(self.chars):
                    text += self.idx2char[idx]
            result.append(text)
        return result
class StaticOCRDataset(Dataset):
    def __init__(self, data_dir, config, is_train=False):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.config = config
        self.is_train = is_train
        self.samples = []
        labels_path = self.data_dir / "labels.txt"
        if not labels_path.exists():
            raise FileNotFoundError(f"❌ 找不到标签文件: {labels_path}，请确保已经运行了静态生成脚本！")
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    self.samples.append((parts[0], parts[1]))
        self.transform = None
        if self.is_train and getattr(config.data, 'augment', False):
            self.transform = T.Compose([
                T.ToPILImage(),
                T.ColorJitter(brightness=0.2, contrast=0.2),  # 轻微亮度与对比度随机扰动
                T.ToTensor(),
            ])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = self.img_dir / img_name
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.config.data.height, self.config.data.width), dtype=np.uint8)
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

        return img_tensor, label

def build_dataloaders(cfg):
    """构建极速读取本地数据的加载器"""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    train_dir = PROJECT_ROOT / "data" / "static_dataset" / "train"
    val_dir = PROJECT_ROOT / "data" / "static_dataset" / "val"
    train_dataset = StaticOCRDataset(train_dir, cfg, is_train=True)
    val_dataset = StaticOCRDataset(val_dir, cfg, is_train=False)
    workers = getattr(cfg.data, 'num_workers', 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    print(f"数据挂载完毕: 训练集 {len(train_dataset)} 张 | 验证集 {len(val_dataset)} 张")
    return train_loader, val_loader