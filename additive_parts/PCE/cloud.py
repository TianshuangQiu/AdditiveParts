import torch
import torch.nn as nn
import pdb
from torch import Tensor
from typing import Optional, Any, Union, Callable
from einops import rearrange
from torch.nn import functional as F


class PointEncoder(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.nhead = nhead
        self.linear1 = nn.Linear(d_model // nhead, dim_feedforward, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model // nhead, **factory_kwargs)

    # def _ff_block(self, x: Tensor) -> Tensor:
    #     c = self.d_model // self.nhead
    #     n = x.shape[-1] // c
    #     x = rearrange(x, "b h (c n) -> b (h n) c", c=c)
    #     x = self.linear2(self.dropout(self.activation(self.linear1(x))))
    #     return rearrange(self.dropout2(x), "b (h n) c -> b h (c n)", c=c, n=n)


class PointCloudProcessor(nn.Module):
    def __init__(
        self,
        input_features,
        nhead,
        nlayer,
        dim_feedforward,
        batch_first,
        linear_layers,
        device=None,
        regression=False,
    ):
        super().__init__()
        self.nhead = nhead
        self.input_features = input_features
        self.encoder = nn.TransformerEncoder(
            PointEncoder(
                input_features * nhead,
                nhead,
                dim_feedforward=dim_feedforward,
                device=device,
                batch_first=batch_first,
            ),
            nlayer,
        )
        self.post_process = nn.Sequential()
        for i, l in enumerate(linear_layers):
            if i == 0:
                self.post_process.append(
                    nn.Linear(self.nhead * self.input_features, l)
                )
                self.post_process.append(nn.ReLU())
            elif i == len(linear_layers) - 1:
                if regression:
                    self.post_process.append(nn.Linear(linear_layers[i - 1], 1))
                else:
                    self.post_process.append(nn.Linear(linear_layers[i - 1], 2))
                    self.post_process.append(nn.Softmax(-1))
            else:
                self.post_process.append(nn.Linear(linear_layers[i - 1], l))
                self.post_process.append(nn.ReLU())

        self.float()

    def forward(self, tensors, src_key_padding=None):
        out = self.encoder(
            tensors.repeat(1, 1, self.nhead), src_key_padding_mask=src_key_padding
        )
        stacked = torch.mean(out, dim=1)
        return self.post_process(stacked)
