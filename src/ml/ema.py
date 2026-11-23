"""Exponential moving average of the model weights (evaluated and exported)."""
from __future__ import annotations

import copy

import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd)

    def to(self, device) -> "ModelEMA":
        self.module.to(device)
        return self
