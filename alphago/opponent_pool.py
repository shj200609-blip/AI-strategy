"""对手池 — 防止策略坍塌。Self-play 时随机选历史模型作为对手。"""

import os
import random
from typing import Optional, List, Tuple
import torch


class OpponentPool:
    def __init__(self, capacity: int = 5, save_dir: str = "checkpoints/opponent_pool"):
        self._capacity = max(1, capacity)
        self._save_dir = save_dir
        self._entries: List[Tuple[int, str]] = []
        os.makedirs(save_dir, exist_ok=True)

    def add(self, model_state_dict: dict, step: int) -> None:
        path = os.path.join(self._save_dir, f"pool_{step}.pt")
        torch.save({"step": step, "model": model_state_dict}, path)
        self._entries.append((step, path))
        while len(self._entries) > self._capacity:
            _, old_path = self._entries.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

    def sample(self) -> Optional[dict]:
        if not self._entries:
            return None
        _, path = random.choice(self._entries)
        return torch.load(path, map_location="cpu", weights_only=False)["model"]

    def load_from_disk(self) -> None:
        self._entries.clear()
        if not os.path.isdir(self._save_dir):
            return
        candidates = []
        for fname in os.listdir(self._save_dir):
            if not fname.startswith("pool_") or not fname.endswith(".pt"):
                continue
            path = os.path.join(self._save_dir, fname)
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                candidates.append((int(ckpt["step"]), path))
            except Exception:
                continue
        candidates.sort(key=lambda x: x[0])
        self._entries = candidates[-self._capacity:]

    def __len__(self) -> int:
        return len(self._entries)
