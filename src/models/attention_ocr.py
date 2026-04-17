import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    注意力计算模块：决定当前时刻应该把目光聚焦在图片的哪个位置
    """

    def __init__(self, hidden_size):
        super().__init__()
        # 计算注意力的三个线性层
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: 上一个时刻大脑的记忆 (1, B, Hidden)
        # encoder_outputs: CNN提取的全部图片线索 (B, T, Hidden)

        batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)

        # 把记忆复制 T 份，准备和 T 个线索一一比对
        hidden = hidden.repeat(max_len, 1, 1).transpose(0, 1)  # (B, T, Hidden)

        # 打分机制：当前记忆和哪个线索最匹配？
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)  # (B, T)

        # 用 softmax 把分数变成百分比（注意力权重），加起来等于 1
        return F.softmax(attention_scores, dim=1)


class AttentionDecoder(nn.Module):
    """
    带注意力的解码器：逐个吐出字符
    """

    def __init__(self, num_classes, hidden_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # 词嵌入：把上一步输出的数字类别，变成向量
        self.embedding = nn.Embedding(num_classes, hidden_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, encoder_outputs):
        # input_step: 上一步输出的字符 (B, 1)
        # 1. 把字符变成特征向量
        embedded = self.dropout(self.embedding(input_step))  # (B, 1, Hidden)

        # 2. 计算注意力权重 (B, T)
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)  # (B, 1, T)

        # 3. 根据注意力权重，从 CNN 线索中提取"精华" (Context Vector)
        # 相当于用目光锁定了图片里的某一块
        context = torch.bmm(a, encoder_outputs)  # (B, 1, Hidden)

        # 4. 把"精华"和"上一个字符"结合起来，送进 GRU 思考
        rnn_input = torch.cat((embedded, context), dim=2)  # (B, 1, Hidden * 2)
        output, hidden = self.gru(rnn_input, hidden)

        # 5. 得出最终预测结果
        prediction = self.out(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))  # (B, num_classes)
        return prediction, hidden


class Seq2SeqAttention(nn.Module):
    """
    完整的 Attention OCR 模型封装
    """

    def __init__(self, cnn_backbone, num_classes, hidden_size=256, max_seq_len=12):
        super().__init__()
        self.encoder = cnn_backbone  # 直接复用你的 cnn6
        self.decoder = AttentionDecoder(num_classes, hidden_size)
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

    def forward(self, x, target_tensor=None, teacher_forcing_ratio=0.5):
        batch_size = x.size(0)

        # 1. 眼睛看图 (Encoder)
        # 你的 cnn6 输出是 (T, B, C)，我们需要转成 (B, T, C)
        encoder_outputs = self.encoder(x)
        if encoder_outputs.dim() == 4:  # 如果没经过RNN的维度大挪移
            encoder_outputs = encoder_outputs.squeeze(2).permute(0, 2, 1)
        else:
            encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # 2. 初始化大脑记忆
        hidden = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

        # 3. 准备解码
        # 在 Attention 里，我们需要 SOS (Start Of Sequence) 标记来启动
        # 假设我们把 num_classes - 1 作为 SOS_token
        SOS_token = self.decoder.num_classes - 1
        decoder_input = torch.tensor([[SOS_token]] * batch_size).to(x.device)

        # 用来装每一步的预测结果
        outputs = torch.zeros(batch_size, self.max_seq_len, self.decoder.num_classes).to(x.device)

        # 4. 循环吐字（自回归）
        for t in range(self.max_seq_len):
            prediction, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = prediction

            # 拿到概率最大的那个字，作为下一步的输入
            top1 = prediction.argmax(1).unsqueeze(1)

            # 训练时的一个小技巧：Teacher Forcing (老师硬核纠错)
            if target_tensor is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # 哪怕模型猜错了，下一步也喂给它正确的答案，防止一步错步步错
                decoder_input = target_tensor[:, t].unsqueeze(1)
            else:
                decoder_input = top1

        return outputs