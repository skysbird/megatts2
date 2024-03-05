import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional
import math

class TokenEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.dim_model)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings.weight[index : index + 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = self.word_embeddings(x)
        X = self.dropout(X)

        return X


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


class ContentEncoder(nn.Module):
    def __init__(self, 
                 d_model: int = 512, 
                 nhead: int = 2, 
                 num_encoder_layers: int = 8, 
                 dim_feedforward: int = 1024, 
                 dropout: float = 0.1, 
                 kernel_size: int = 5,
                 vocab_size: int = 320):
        super().__init__()


        self.phone_embedding = TokenEmbedding(
            dim_model=d_model,
            vocab_size=vocab_size,
            dropout=dropout,
        )

        self.positional_encoding = SinePositionalEmbedding(
            dim_model=d_model,
            dropout=dropout,
        )

        self.conv1d = nn.Conv1d(in_channels=d_model, 
                                out_channels=d_model, 
                                kernel_size=kernel_size, 
                                padding=kernel_size//2)
        
        encoder_layer = TransformerEncoderLayer(d_model=d_model, 
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward, 
                                                dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, 
                                                      num_layers=num_encoder_layers)
        

    def forward(self, 
                phone: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        src = self.phone_embedding(phone)
        src = self.positional_encoding(src)
        src = src.permute(0, 2, 1)
        src = self.conv1d(src)
        src = src.permute(0, 2, 1)
        output = self.transformer_encoder(src)
        return output


if __name__ == "__main__":
    # Example input for testing the content encoder
    # input_sequence = torch.randint(1, 320) 
    input_sequence = torch.randint(0, 320, (1, 100))
    # Initialize the content encoder
    content_encoder = ContentEncoder()

    # Forward pass of the content encoder
    encoded_sequence = content_encoder(input_sequence)
    print(encoded_sequence.shape)  # Expected shape: (batch_size, sequence_length, embedding_dim)
    print(input_sequence[0])
    print(encoded_sequence[0])