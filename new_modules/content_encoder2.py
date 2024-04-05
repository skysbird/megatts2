import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from .embedding import SinePositionalEmbedding

class FastSpeechContentEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(FastSpeechContentEncoder, self).__init__()
        self.d_model = d_model
        self.pos_encoder = SinePositionalEmbedding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                 dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=d_model)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output
