import copy
import warnings
from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn
from mamba_ssm import Mamba, Mamba2
from .convolutions import DilatedConv1dBlock

class MambaEncoderLayer(nn.Module):
    """
    Code based on: https://github.com/automl/Mamba4Cast/blob/main/src_torch/training/blocks.py
    """
    def __init__(
            self, 
            embed_dim, 
            norm=True, 
            residual=False, # Not implemented currently
            mamba2=False, 
            enc_conv=False, # Not implemented currently 
            bidirectional_mamba=False,
            enc_conv_kernel=5, 
            enc_conv_dilation=0, 
            d_state=128, 
            block_expansion=2, 
            d_conv=4,
            **kwargs
    ):
        super().__init__(**kwargs)
        
        self.enc_conv = enc_conv
        self.norm = norm
        self.bidirectional_mamba = bidirectional_mamba

        if not mamba2:
            self.mamba_layer = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=embed_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=block_expansion,    # Block expansion factor
            )

        else:
            self.mamba_layer = Mamba2(
                d_model=embed_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=block_expansion,    # Block expansion factor
            )

        if self.bidirectional_mamba:
            if not mamba2:
                self.mamba_layer_backward = Mamba(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=embed_dim, # Model dimension d_model
                    d_state=d_state,  # SSM state expansion factor
                    d_conv=d_conv,    # Local convolution width
                    expand=block_expansion,    # Block expansion factor
                )
            else:
                self.mamba_layer_backward = Mamba2(
                    d_model=embed_dim, # Model dimension d_model
                    d_state=d_state,  # SSM state expansion factor
                    d_conv=d_conv,    # Local convolution width
                    expand=block_expansion,    # Block expansion factor
                )
            
        if self.enc_conv:
            # Optional convolutional layer instead of feed-forward with activation
            self.stage_2_layer = DilatedConv1dBlock(embed_dim, embed_dim, enc_conv_kernel, 
                                                         enc_conv_dilation, single_conv=False)
        else:
            # Feed-forward layer with activation
            self.stage_2_layer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

        if self.norm:
            self.norm_layer_1 = nn.LayerNorm(embed_dim)
            self.norm_layer_2 = nn.LayerNorm(embed_dim)

        self.residual = residual

    def forward(self, x, inference_params=None):
        
        if self.norm:
            x = self.norm_layer_1(x)

        x_ssm = self.mamba_layer(x, inference_params=inference_params)

        if self.bidirectional_mamba:
            x_ssm = x_ssm + self.mamba_layer_backward(x.flip(dims=[1])).flip(dims=[1])

        if self.residual:
            x = x + x_ssm
        else:
            x = x_ssm

        if self.norm:
            x_out = self.stage_2_layer(self.norm_layer_2(x))
        else:
            x_out = self.stage_2_layer(x) 
        
        if self.residual:
            x_out = x_out + x

        return x_out