import copy
import warnings
from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

class DilatedConv1dBlock(nn.Module):
    """
    Code based on: https://github.com/automl/Mamba4Cast/blob/main/src_torch/training/blocks.py
    """
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=5,
                 max_dilation=3,
                 single_conv=False,
                 conv_gelu=True):
        super(DilatedConv1dBlock, self).__init__()
        self.conv_gelu = conv_gelu
        self.single_conv = single_conv
        if self.single_conv:
            padding = (kernel_size - 1) * 2**max_dilation
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=2**max_dilation,
                bias=True
            )
        else:
            self.conv = nn.ModuleList()
            conv_out_channels = out_channels // (max_dilation + 1)
            for dilation in range(max_dilation + 1):
                padding = (kernel_size - 1) * 2**dilation
                self.conv.append(nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=conv_out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=2**dilation,
                    bias=True
                ))
        
            self.inception_conv = nn.Conv1d(
                in_channels=out_channels,  # Total number of channels from all convolutions
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

    def forward(self, x):
        # x is expected to be of shape (batch_size, in_channels, sequence_length)
        x = x.transpose(1, 2)
        seq_len = x.shape[-1]
        if self.single_conv:
            x = self.conv(x)
            if self.conv_gelu:
                x = F.gelu(x)
        else:
            if self.conv_gelu:
                x_list = [F.gelu(conv_layer(x))[:,:,:seq_len] for conv_layer in self.conv]
            else:
                x_list = [conv_layer(x)[:,:,:seq_len] for conv_layer in self.conv]
            x = torch.cat(x_list, dim=1)
            x = self.inception_conv(x)
        return x.transpose(1, 2)