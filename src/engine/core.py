import torch
from tqdm import tqdm


class OCREngine:
    """高度内聚的训练与推理引擎"""

    def __init__(self, model, device, converter, config):
        self.model = model.to(device)
        self.device = device
        self.converter = converter
        self.cfg = config

    def train_loop(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc="Training")

        for images, labels_str in pbar:
            images = images.to(self.device)
            targets, target_lengths = self.converter.encode(labels_str)

            optimizer.zero_grad()
            preds = self.model(images)
            preds_log = preds.log_softmax(2)

            input_lengths = torch.full((images.size(0),), preds.size(0), dtype=torch.long)
            loss = criterion(preds_log, targets.to(self.device), input_lengths, target_lengths.to(self.device))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        correct, total = 0, 0

        for images, labels_str in tqdm(val_loader, desc="Evaluating"):
            images = images.to(self.device)
            preds = self.model(images)
            pred_indices = preds.argmax(2).permute(1, 0)

            for idx, label in zip(pred_indices, labels_str):
                pred_str = self.converter.decode(idx)
                if pred_str == label:
                    correct += 1
                total += 1

        return correct / total

    @torch.no_grad()
    def infer(self, image_tensor):
        self.model.eval()
        preds = self.model(image_tensor.to(self.device))
        pred_indices = preds.argmax(2).permute(1, 0)
        return self.converter.decode(pred_indices[0])