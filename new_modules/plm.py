import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from embedding import SinePositionalEmbedding

class PLMModel(nn.Module):
    def __init__(self, d_model=2048, nhead=16, num_decoder_layers=24, dim_feedforward=8192, dropout=0.1):
        super(PLMModel, self).__init__()
        self.positional_encoding =  SinePositionalEmbedding(
            dim_model=d_model,
            dropout=dropout,
        )
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, src, memory):
        src = self.positional_encoding(src)
        output = self.transformer_decoder(src, memory)
        return self.output_layer(output)

if __name__=='__main__':
    # Assuming the inputs are:
    # src with shape [sequence_length, batch_size, d_model]
    # memory (encoder outputs) with shape [sequence_length, batch_size, d_model]
    sequence_length, batch_size, d_model = 100, 2, 2048
    src = torch.rand(sequence_length, batch_size, d_model)
    memory = torch.rand(sequence_length, batch_size, d_model)

    # Create the PLM model
    plm_model = PLMModel()

    # Forward pass through the PLM model
    output = plm_model(src, memory)

    print(output.shape)  # Expected shape: [sequence_length, batch_size, d_model]
