# model.py

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    # Sinusoidal positional encoding.

    def __init__(
        self, d_model, dropout, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Position index: [0, 1, 2, ..., max_len - 1]
        position = torch.arange(max_len, dtype=torch.float64).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float64)
            * (-math.log(10000.0) / d_model)
        )

        # Positional encoding buffer shape: [max_len, 1, d_model]
        pe = torch.zeros(
            max_len,
            1,
            d_model,
            dtype=torch.float64,
        )
        
        # Even embedding dimensions use sine.
        pe[:, 0, 0::2] = torch.sin(position * div_term) 
        
        # Odd embedding dimensions use cosine.
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PedestrianCrossingTransformer(nn.Module):
    # Transformer used for pedestrian crossing prediction.

    def __init__(
        self,
        input_dim: int = 9,
        d_model: int = 8,
        ff_dim: int = 16,
        n_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 2,
        seq_len: int = 30,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim
        self.seq_len = seq_len

        # Input projection
        self.encoder = nn.Linear(
            in_features=input_dim,
            out_features=d_model,
            bias=True,
        )
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=5000,
        )
        # Transformer encoder layer for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            layer_norm_eps=1e-5,
            batch_first=False,
            norm_first=False,
            bias=True,
        )
    
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        # Readout layer to flatten the sequence dimension
        self.readout = nn.Flatten(
            start_dim=1,
            end_dim=-1,
        )
        # Output projection 
        self.decoder = nn.Linear(
            in_features=self.seq_len * d_model,
            out_features=output_dim,
            bias=True,
        )

    def forward(self, x):
        # Input:[B, 30, 9]

        # Transformer used sequence-first representation.
        x = x.transpose(0, 1)

        # [30, B, 9] ->[30, B, 8]
        x = self.encoder(x)

        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        # [30, B, 8] -> [B, 30, 8]
        x = x.transpose(0, 1)

        # [B, 30, 8] -> [B, 240]
        x = self.readout(x)

        # [B, 240] -> # [B, 2]
        x = self.decoder(x)

        return x
