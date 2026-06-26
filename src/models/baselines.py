import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from .core import BidirectionalLSTM, OCR_ResNet18
class MobileNetV3_CRNN(nn.Module):
    def __init__(self, img_channel=1, num_classes=11, hidden_size=256):
        super().__init__()
        mobilenet = mobilenet_v3_small(weights=None)
        original_conv1 = mobilenet.features[0][0]
        self.conv1 = nn.Conv2d(img_channel, original_conv1.out_channels,
                               kernel_size=original_conv1.kernel_size,
                               stride=original_conv1.stride,
                               padding=original_conv1.padding, bias=False)
        self.features = nn.Sequential(*list(mobilenet.features.children())[1:])
        self.out_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.Sequential(
            BidirectionalLSTM(576, hidden_size, hidden_size, dropout=0.2),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes, dropout=0.2)
        )
    def forward(self, x, targets=None):
        x = self.conv1(x)
        x = self.features(x)
        x = self.out_pool(x).squeeze(2).permute(2, 0, 1)
        return self.rnn(x)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, encoder_dim):
        super().__init__()
        self.w1 = nn.Linear(encoder_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        h = hidden.permute(1, 0, 2)
        e = encoder_outputs.permute(1, 0, 2)
        score = self.v(torch.tanh(self.w1(e) + self.w2(h)))
        weights = torch.softmax(score, dim=1)
        context = torch.bmm(weights.permute(0, 2, 1), e)
        return context, weights


class ResNet_Attention(nn.Module):
    def __init__(self, num_classes, hidden_size=256, max_seq_len=15):
        super().__init__()
        self.cnn = OCR_ResNet18(img_channel=1)
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.attention = BahdanauAttention(hidden_size, 512)
        self.embedding = nn.Embedding(num_classes, hidden_size)
        self.gru = nn.GRU(hidden_size + 512, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, x, targets=None, teacher_forcing_ratio=0.5):
        encoder_outputs = self.cnn(x).squeeze(2).permute(2, 0, 1)  # [T, B, 512]
        batch_size = x.size(0)
        hidden = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        decoder_input = torch.full((batch_size, 1), self.num_classes - 1, dtype=torch.long).to(x.device)
        outputs = []
        for t in range(self.max_seq_len):
            embedded = self.embedding(decoder_input)
            context, _ = self.attention(hidden, encoder_outputs)
            rnn_input = torch.cat((embedded, context), dim=2)
            out, hidden = self.gru(rnn_input, hidden)
            pred = self.classifier(out.squeeze(1))
            outputs.append(pred)
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio and t < targets.size(1):
                decoder_input = targets[:, t].unsqueeze(1)
            else:
                decoder_input = pred.argmax(1).unsqueeze(1)
        return torch.stack(outputs, dim=1)