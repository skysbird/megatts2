import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from new_modules.embedding import SinePositionalEmbedding


class ADM(nn.Module):
    def __init__(self, d_model=2048, nhead=16, num_layers=24, dim_feedforward=8192):
        super().__init__()
        self.pos_encoder = SinePositionalEmbedding(d_model)
        self.decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_decoder = TransformerDecoder(self.decoder_layers, num_layers)
        self.output_layer = nn.Linear(d_model, 1)  # Output layer for duration prediction

    def forward(self, src, memory):
        src = self.pos_encoder(src)
        output = self.transformer_decoder(src, memory)
        return self.output_layer(output)

if __name__=='__main__':
    # Example usage
    # src has shape [seq_len, batch_size, d_model], for example:
    seq_len, batch_size, d_model = 10, 1, 2048
    src = torch.rand(seq_len, batch_size, d_model)
    memory = torch.rand(seq_len, batch_size, d_model)

    adm_model = ADM()
    output = adm_model(src, memory)
    print(output.shape)  # Should be [seq_len, batch_size, 1] where last dimension is the predicted duration
