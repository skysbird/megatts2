import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.quantization.core_vq import VectorQuantization

# class VectorQuantizer(nn.Module):
#     def __init__(self, hidden_channels, num_embeddings, embedding_dim, commitment_cost):
#         super(VectorQuantizer, self).__init__()
#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim
#         self.commitment_cost = commitment_cost

#         self.embedding = nn.Embedding(num_embeddings, embedding_dim)
#         self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
#         self.adjust_conv = nn.Conv1d(hidden_channels, embedding_dim, kernel_size=1)  # 新增加的卷积层

#     def forward(self, inputs):
#         #inputs = inputs.permute(0, 2, 1).contiguous()
#         #print(inputs.shape)
#         inputs = self.adjust_conv(inputs)  # 新增加的卷积层来调整维度

#         flat_inputs = inputs.view(-1, self.embedding_dim)

#         # Calculate distances
#         distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True)
#                      + torch.sum(self.embedding.weight**2, dim=1)
#                      - 2 * torch.matmul(flat_inputs, self.embedding.weight.t()))

#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)

#         # Quantize and reshape
#         quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
#         quantized = inputs + (quantized - inputs).detach()

#         # Loss
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)
#         q_latent_loss = F.mse_loss(quantized, inputs.detach())
#         loss = q_latent_loss + self.commitment_cost * e_latent_loss

#         #quantized = quantized.permute(0, 2, 1).contiguous()
#         return loss, quantized, encoding_indices

class VQProsodyEncoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels,
                 kernel_size, 
                 num_embeddings, 
                 embedding_dim, 
                 commitment_cost,
                 dim = 256,
                 bins = 1024,
                 decay = 0.99,
                 kmeans_init: bool = True,
                 kmeans_iters: int = 50,
                 threshold_ema_dead_code: int = 2,
                ):
        super(VQProsodyEncoder, self).__init__()
        num_layers = 5

        self.num_layers = num_layers
        self.conv1d_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=in_channels if i == 0 else hidden_channels,
                          out_channels=hidden_channels,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.LayerNorm([hidden_channels, 1]),
                nn.GELU()
            ) for i in range(num_layers)
        ])
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8, padding=0)

        self.last_conv1d_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=in_channels if i == 0 else hidden_channels,
                          out_channels=hidden_channels,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.LayerNorm([hidden_channels, 1]),
                nn.GELU()
            ) for i in range(num_layers)
        ])
        # self.conv1d = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        # self.vq = VectorQuantizer(hidden_channels, num_embeddings, embedding_dim, commitment_cost)
        self.vq = VectorQuantization(
            dim=hidden_channels,
            codebook_size=bins,
            decay=decay,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            threshold_ema_dead_code=threshold_ema_dead_code
        )

    def forward(self, mel_spec):
        # Assuming mel_spec is of shape (batch_size, channels, time)
        #x = self.conv1d(mel_spec)  # Apply Conv1D
        x = mel_spec
        for i in range(self.num_layers):
            x = self.conv1d_blocks[i](x)
        
        x = self.pool(x) 

        for i in range(self.num_layers):
            x = self.last_conv1d_blocks[i](x)

        quantize, embed_ind, loss = self.vq(x)  # Apply Vector Quantization
        return quantize, loss, embed_ind

# Define hyperparameters
# in_channels = 80  # Number of mel bins
# hidden_channels = 384
# kernel_size = 5
# num_embeddings = 1024
# embedding_dim = 256
# commit
