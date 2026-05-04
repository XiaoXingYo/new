import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from .generator import OCRDataGenerator


# ==========================================
# 🟢 路线 A：CTC 专属翻译官 (已升级：支持 Greedy/Beam Search 且大幅提速)
# ==========================================
class LabelConverter:
    """处理 CRNN+CTC 的标签转换"""

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


# ==========================================
# 🔴 路线 B：Attention 专属翻译官 (已去除了掉速的 item() 调用)
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
        batch_size = len(text_list)
        targets = torch.full((batch_size, self.max_seq_len), self.pad_idx, dtype=torch.long)

        for i, text in enumerate(text_list):
            text = text[:self.max_seq_len - 1]
            for j, char in enumerate(text):
                targets[i, j] = self.char2idx[char]
            targets[i, len(text)] = self.eos_idx

        return targets

    def decode(self, pred_idx):
        # 优化：提前转为 Python List，大幅提升解码速度
        if hasattr(pred_idx, 'tolist'):
            pred_idx = pred_idx.tolist()

        result = []
        for seq in pred_idx:
            text = ""
            for idx in seq:
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
        self.is_train = is_train

        # 1. 隔离配置：验证集绝对不能有数据增强
        import copy
        local_config = copy.deepcopy(config)
        if not is_train:
            local_config.data.augment = False  # 强制关闭验证集的数据增强

        self.generator = OCRDataGenerator(local_config)

        # 2. 定义单个 Epoch 的数据量 (千万不要乘以 epochs 数量)
        self.length = 10000 if is_train else 1000

        # 3. 验证集固化：考试用同一张试卷！
        self.val_cache = []
        if not is_train:
            print("⏳ 正在预生成固定的验证集，请稍候...")
            import random
            # 临时固定随机种子，确保每次重新启动训练，验证集题目是一样的
            random.seed(42)
            for _ in range(self.length):
                self.val_cache.append(self.generator.generate_sample())
            random.seed()  # 恢复系统的随机状态，不影响后续训练

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_train:
            # 训练集：无限动态生成，每次都不一样
            img_np, label = self.generator.generate_sample()
        else:
            # 验证集：直接从初始化好的题库里拿，保证一致性
            img_np, label = self.val_cache[idx]

        img_tensor = torch.from_numpy(img_np).float()

        # ⚡️ 核心修复：维度自适应
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.dim() == 3 and img_tensor.shape[2] == 1:
            img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor, label