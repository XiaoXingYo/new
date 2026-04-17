import torch
from pathlib import Path
from typing import Dict, Any, Optional
import json


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        # 符号链接
        self.latest_path = self.dir / "latest.pth"
        self.best_path = self.dir / "best_ctc.pth"
        self.metrics_path = self.dir / "metrics.json"

    def save(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            step: int,
            metrics: Dict[str, Any],
            is_best: bool = False
    ):
        """保存完整checkpoint"""
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'metrics': metrics
        }

        # 保存具体文件
        filename = f"epoch_{epoch:04d}_step_{step:06d}.pth"
        filepath = self.dir / filename
        torch.save(state, filepath)
        try:
            # 更新符号链接
            if self.latest_path.exists() or self.latest_path.is_symlink():
                self.latest_path.unlink()
            self.latest_path.symlink_to(filename)
        except OSError:
            pass    # Windows权限不足就算了，不影响主流程
        if is_best:
            try:
                if self.best_path.exists() or self.best_path.is_symlink():
                    self.best_path.unlink()
                self.best_path.symlink_to(filename)
            except OSError:
                import shutil
                shutil.copy(filepath, self.best_path)
            # 保存最佳指标
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

    def load(
            self,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """加载checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.latest_path

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint不存在: {path}")

        state = torch.load(path, map_location='cpu')
        model.load_state_dict(state['model'])

        if optimizer and 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])

        return {
            'epoch': state.get('epoch', 0),
            'step': state.get('step', 0),
            'metrics': state.get('metrics', {})
        }