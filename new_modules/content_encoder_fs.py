import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer.Layers import FFTBlock
import transformer.Constants as Constants


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x, offset = 0):
        """Reset the positional encodings."""
        x_size = x.size(1) + offset
        if self.pe is not None:
            if self.pe.size(1) >= x_size:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x_size, self.dim_model)
        if self.reverse:
            position = torch.arange(
                x_size - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(
                0, x_size, dtype=torch.float32
            ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor, offset : int = 0) -> torch.Tensor:
        self.extend_pe(x, offset)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, offset : x.size(1) + offset]
        return self.dropout(output)


class ContentEncoder2(nn.Module):
    def __init__(
                 self,
                 d_model: int = 512, 
                 d_inner = 1024,
                 n_head = 2,
                 d_k = 512 // 2,
                 d_v = 512 // 2,
                 num_encoder_layers: int = 8, 
                 dropout: float = 0.1, 
                 vocab_size: int = 320
                 ):
        super(ContentEncoder2, self).__init__()
        self.phone_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder =  SinePositionalEmbedding(
            dim_model=d_model,
            dropout=dropout,
        )
        self.transformer_layers = nn.ModuleList([
            FFTBlock(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(num_encoder_layers)
        ])

    def pre_decode(self, content):
        phonemes = self.pos_encoder(phonemes)

    def forward(self, phonemes):
        slf_attn_mask = get_attn_key_pad_mask(seq_k=phonemes, seq_q=phonemes)
        non_pad_mask = get_non_pad_mask(phonemes)

        phonemes = self.phone_embedding(phonemes)
        phonemes = self.pos_encoder(phonemes)
        for layer in self.transformer_layers:
            phonemes, _ = layer(phonemes,
                             non_pad_mask,
                             slf_attn_mask) #    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):

        return phonemes

# # Initialize the Content Encoder
# d_model = 512  # Embedding size and the size for each transformer layer
# num_layers = 8  # Number of transformer layers
# dim_feedforward = 1024  # Dimension of the feedforward network model in transformer layer
# dropout = 0.1  # Dropout value
# phoneme_embedding_size = len(phoneme_dict)  # Assuming phoneme_dict is a dictionary of phonemes

# content_encoder = ContentEncoder(phoneme_embedding_size, num_layers, d_model, dim_feedforward, dropout)

# Example
