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
A) Sequential Mamba.

For each target point, create a sequence [context tokens..., target token],
run Mamba over the full sequence, and use ONLY the last hidden state to represent that target.

"""
class SequentialMambaEncoder(nn.Module):
    def __init__(self, num_layers: int, mamba_layer: MambaEncoderLayer):
        super().__init__()
        self.mamba_layers = _get_clones(mamba_layer, num_layers)

    @check_shapes("xc: [b, nc, d]", "xt: [b, nt, d]", "return: [b, nt, d]")
    def forward(self, xc: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        b, nc, d = xc.shape
        _, nt, d2 = xt.shape
        assert d == d2, f"embed dim mismatch: xc has {d}, xt has {d2}"

        # Replicate context for each target: [b, nt, nc, d] -> [b*nt, nc, d]
        xc_rep = xc[:, None, :, :].expand(b, nt, nc, d).reshape(b * nt, nc, d)

        # Make a single target token per replicated context: [b*nt, 1, d]
        xt_tok = xt.reshape(b * nt, 1, d)

        # Full sequence: [b*nt, nc+1, d]
        x = torch.cat([xc_rep, xt_tok], dim=1)

        # Mamba over the sequence
        for layer in self.mamba_layers:
            x = layer(x)

        # Take ONLY last hidden state as target embedding: [b*nt, d] -> [b, nt, d]
        h_last = x[:, -1, :].reshape(b, nt, d)
        return h_last






"""
B) Sequential Mamba Option

    1) Run Mamba over zc ONCE, caching the final internal state.
    2) For each target token zt_i, run ONE decode step from that same cached state.
    3) Return [B, Nt, D] (one representation per target).

This plugs into TNPEncoder unchanged (it consumes embedded tokens):
    zc: [B, Nc, D]
    zt: [B, Nt, D]
"""




def _repeat_cache_for_targets(cache: dict, repeat: int) -> dict:
    out = {}
    for k, v in cache.items():
        if isinstance(v, dict):
            out[k] = _repeat_cache_for_targets(v, repeat)
        elif isinstance(v, tuple):
            out[k] = tuple(t.repeat_interleave(repeat, dim=0) for t in v)
        elif torch.is_tensor(v):
            out[k] = v.repeat_interleave(repeat, dim=0)
        else:
            out[k] = v
    return out

def _clone_cache(cache: dict) -> dict:
    # mamba cache is typically: {layer_idx: (conv_state, ssm_state)}
    out = {}
    for k, v in cache.items():
        if isinstance(v, tuple):
            out[k] = tuple(t.clone() for t in v)
        elif isinstance(v, dict):
            out[k] = _clone_cache(v)
        elif torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = v
    return out



class SequentialMambaEncoderB(nn.Module):
    def __init__(self, num_layers: int, mamba_layer: MambaEncoderLayer):
        super().__init__()
        self.mamba_layers = _get_clones(mamba_layer, num_layers)

        # IMPORTANT: set unique layer_idx for mamba_ssm caching
        for i, layer in enumerate(self.mamba_layers):
            if hasattr(layer, "mamba_layer") and hasattr(layer.mamba_layer, "layer_idx"):
                layer.mamba_layer.layer_idx = i
            if getattr(layer, "bidirectional_mamba", False) and hasattr(layer, "mamba_layer_backward"):
                if hasattr(layer.mamba_layer_backward, "layer_idx"):
                    layer.mamba_layer_backward.layer_idx = i + 10_000  # any distinct index

    @check_shapes(
        "zc: [b, nc, d]",
        "zt: [b, nt, d]",
        "mask: [b, nt, nc]",
        "return: [b, nt, d]",
    )
    def forward(
        self,
        zc: torch.Tensor,
        zt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, nc, d = zc.shape
        b2, nt, d2 = zt.shape
        assert b == b2 and d == d2, f"Shape mismatch: zc={zc.shape}, zt={zt.shape}"

        # --------------------------
        # 1) Encode context once, while building cache
        # --------------------------
        # Create inference params + cache containers per layer
        # NOTE: max_seqlen must be >= nc + 1 (we'll decode 1 token later)
        inf = InferenceParams(
            max_seqlen=nc + 1,
            max_batch_size=b,
            seqlen_offset=0,
            key_value_memory_dict={},
        )

        x = zc
        for layer in self.mamba_layers:
            x = layer(x, inference_params=inf)

        # Save a *copy* of the context cache state (so targets can branch from it
        ctx_cache = _clone_cache(inf.key_value_memory_dict)


        # --------------------------
        # 2) Decode one step per target (batched as B*Nt)
        # --------------------------
        # Expand caches to match batch=B*Nt so every target gets the same context state
        tgt_cache = _repeat_cache_for_targets(ctx_cache, repeat=nt)

        inf_t = InferenceParams(
            max_seqlen=nc + 1,
            max_batch_size=b * nt,
            seqlen_offset=nc,  # we're decoding AFTER consuming nc context tokens
            key_value_memory_dict=tgt_cache,
        )

        # One-token decode inputs: [B*Nt, 1, D]
        xtok = zt.reshape(b * nt, 1, d)

        y = xtok
        for layer in self.mamba_layers:
            y = layer(y, inference_params=inf_t)

        # y is [B*Nt, 1, D] -> take last token (only token) -> [B, Nt, D]
        out = y[:, -1, :].reshape(b, nt, d)
        return out