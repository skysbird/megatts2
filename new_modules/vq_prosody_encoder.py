import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        print(inputs.shape)
        flat_inputs = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_inputs, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and reshape
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        quantized = inputs + (quantized - inputs).detach()

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = quantized.permute(0, 2, 1).contiguous()
        return loss, quantized, encoding_indices

class VQProsodyEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, num_embeddings, embedding_dim, commitment_cost):
        super(VQProsodyEncoder, self).__init__()

        self.conv1d = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, mel_spec):
        # Assuming mel_spec is of shape (batch_size, channels, time)
        x = self.conv1d(mel_spec)  # Apply Conv1D
        loss, quantized, encoding_indices = self.vq(x)  # Apply Vector Quantization
        return quantized, loss, encoding_indices

# Define hyperparameters
# in_channels = 80  # Number of mel bins
# hidden_channels = 384
# kernel_size = 5
# num_embeddings = 1024
# embedding_dim = 256
# commit
