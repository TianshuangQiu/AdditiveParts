import torch
import torch.nn as nn
import pdb


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
    ):
        super().__init__()
        self.nhead = nhead
        self.input_features = input_features
        self.encoder_layer = nn.TransformerEncoderLayer(
            input_features * nhead,
            nhead,
            dim_feedforward=dim_feedforward,
            device=device,
            batch_first=batch_first,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, nlayer)
        self.post_process = nn.Sequential()
        for i, l in enumerate(linear_layers):
            if i == 0:
                self.post_process.append(
                    nn.Linear(self.nhead * self.input_features * 3, l)
                )
                # self.post_process.append(nn.ReLU())
            elif i == len(linear_layers) - 1:
                self.post_process.append(nn.Linear(linear_layers[i - 1], 1))
            else:
                self.post_process.append(nn.Linear(linear_layers[i - 1], l))
                self.post_process.append(nn.ReLU())

        self.float()

    def forward(self, tensors, src_key_padding=None):
        # out = self.encoder_layer(tensors, src_key_padding_mask=src_key_padding)
        out = self.encoder(
            tensors.repeat(1, 1, self.nhead), src_key_padding_mask=src_key_padding
        )
        stacked = torch.concat(
            [
                torch.min(out, dim=1)[0],
                torch.mean(out, dim=1),
                torch.max(out, dim=1)[0],
            ],
            dim=1,
        )
        return self.post_process(stacked)
