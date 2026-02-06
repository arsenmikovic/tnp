import copy
import warnings
from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from .attention_layers import (
    MultiHeadCrossAttentionLayer
)
from .mamba_layers import MambaEncoderLayer
from .transformer import _get_clones
# Mamba cache helper
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm import Mamba, Mamba2





class TNPMambaEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mamba_layer: MambaEncoderLayer,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mamba_layers = _get_clones(mamba_layer, num_layers)
    
    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        for mamba_layer, mhca_layer in zip(self.mamba_layers, self.mhca_layers):
            if mask is not None:
                warnings.warn("mask is not currently being used.")

            xc = mamba_layer(xc)
            xt = mhca_layer(xt, xc)

        return xt

class MNP_NDMambaEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mamba_layer: MambaEncoderLayer,
    ):
        super().__init__()

        self.mamba_layers = _get_clones(mamba_layer, num_layers)

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:

        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)

        return xt





"""
Helper functions for fast inference.

"""

def _repeat0(t: torch.Tensor, repeats: int) -> torch.Tensor:
    return t.repeat_interleave(repeats, dim=0)

def tile_inference_params(inf: InferenceParams, repeats: int, new_max_batch: int) -> InferenceParams:
    new_inf = copy.copy(inf)
    new_inf.max_batch_size = new_max_batch

    new_kv = {}
    for k, v in inf.key_value_memory_dict.items():
        if torch.is_tensor(v):
            new_kv[k] = _repeat0(v, repeats)
        elif isinstance(v, (tuple, list)):
            out = []
            for item in v:
                out.append(_repeat0(item, repeats) if torch.is_tensor(item) else item)
            new_kv[k] = type(v)(out)
        elif isinstance(v, dict):
            out = {}
            for kk, vv in v.items():
                out[kk] = _repeat0(vv, repeats) if torch.is_tensor(vv) else vv
            new_kv[k] = out
        else:
            raise TypeError(f"Unsupported cache type: {type(v)} for key {k}")

    new_inf.key_value_memory_dict = new_kv
    return new_inf

def assign_mamba_layer_indices(root: nn.Module, start: int = 0) -> int:
    """
    Assign a unique layer_idx to every Mamba/Mamba2 module under `root`.
    Required for inference_params caching in mamba_ssm.
    Returns the next available index.
    """
    idx = start
    for m in root.modules():
        if isinstance(m, (Mamba, Mamba2)):
            m.layer_idx = idx
            idx += 1
    return idx





"""
A) Sequential Mamba.

For each target point, create a sequence [context tokens..., target token],
run Mamba over the full sequence, and use ONLY the last hidden state to represent that target.

"""
class SequentialMambaEncoder(nn.Module):
    def __init__(self, num_layers: int, mamba_layer: MambaEncoderLayer):
        super().__init__()
        self.mamba_layers = _get_clones(mamba_layer, num_layers)
        assign_mamba_layer_indices(self)

    @check_shapes("xc: [b, nc, d]", "xt: [b, nt, d]", "return: [b, nt, d]")
    def _forward_train(self, xc: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """
        training implementation: fully differentiable, no caching.
        """
        b, nc, d = xc.shape
        _, nt, d2 = xt.shape
        assert d == d2, f"embed dim mismatch: xc has {d}, xt has {d2}"

        # Replicate context for each target: [b, nt, nc, d] -> [b*nt, nc, d]
        xc_rep = xc[:, None, :, :].expand(b, nt, nc, d).reshape(b * nt, nc, d)

        # Target token: [b*nt, 1, d]
        xt_tok = xt.reshape(b * nt, 1, d)

        # Full sequence: [b*nt, nc+1, d]
        x = torch.cat([xc_rep, xt_tok], dim=1)

        # Mamba over full sequence
        for layer in self.mamba_layers:
            x = layer(x)

        # Only last hidden state: [b*nt, d] -> [b, nt, d]
        return x[:, -1, :].reshape(b, nt, d)

    @torch.no_grad()
    def _forward_cached_inference(self, xc: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """
        Inference-only cached path:
          1) Prefill cache with context once
          2) Branch cache to evaluate all targets as 1-token continuations
        """
        b, nc, d = xc.shape
        _, nt, d2 = xt.shape
        assert d == d2, f"embed dim mismatch: xc has {d}, xt has {d2}"

        # 1) Prefill cache with context
        inf = InferenceParams(max_seqlen=nc + 1, max_batch_size=b)

        x = xc
        for layer in self.mamba_layers:
            x = layer(x, inference_params=inf)

        # 2) Branch to targets
        xt_flat = xt.reshape(b * nt, 1, d)  # [b*nt, 1, d]
        inf_tiled = tile_inference_params(inf, repeats=nt, new_max_batch=b * nt)

        y = xt_flat
        for layer in self.mamba_layers:
            y = layer(y, inference_params=inf_tiled)

        # y: [b*nt, 1, d] -> [b, nt, d]
        return y[:, 0, :].reshape(b, nt, d)

    @check_shapes("xc: [b, nc, d]", "xt: [b, nt, d]", "return: [b, nt, d]")
    def forward(self, xc: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """
        Automatic routing:
        - Training / grads enabled -> training path (your current one)
        - Eval + no grads -> cached inference path
        """
        if (not self.training) and (not torch.is_grad_enabled()):
            return self._forward_cached_inference(xc, xt)
        return self._forward_train(xc, xt)



