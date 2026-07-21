"""
Weighted layer pooling for transformer speech encoders.

Supports two modes:
- norm="none": no normalization across layers
- norm="batchnorm": BatchNorm over the layer dimension (channels = layers)

Expected input hidden states shape after stacking: [B, L, T, D]
where L = number of layers (including embedding output if provided).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_layers: int, norm: str = "none"):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")

        norm = (norm or "none").lower()
        if norm in {"bn", "batchnorm", "batch_norm"}:
            self.norm_type = "batchnorm"
            self.bn = nn.BatchNorm2d(num_layers)
        elif norm in {"none", "no", "identity"}:
            self.norm_type = "none"
            self.bn = None
        else:
            raise ValueError(f"Unknown weighted layer norm '{norm}'. Use 'none' or 'batchnorm'.")

        # Learnable layer weights (softmaxed in forward)
        self.layer_weights = nn.Parameter(torch.ones(num_layers, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            hidden_states:
              - Tensor [B, L, T, D], OR
              - list/tuple of Tensors each [B, T, D] (will be stacked on dim=1)

        Returns:
            pooled: Tensor [B, T, D]
        """
        if isinstance(hidden_states, (list, tuple)):
            x = torch.stack(list(hidden_states), dim=1)  # [B, L, T, D]
        else:
            x = hidden_states

        if x.dim() != 4:
            raise ValueError(f"Expected hidden states with 4 dims [B,L,T,D], got shape {tuple(x.shape)}")

        # Optional normalization over layers (channels=L).
        if self.bn is not None:
            b, l, t, d = x.shape
            x = self.bn(x)

        weights = torch.softmax(self.layer_weights, dim=0).to(dtype=x.dtype)  # [L]
        pooled = (x * weights.view(1, -1, 1, 1)).sum(dim=1)  # [B, T, D]
        return pooled

