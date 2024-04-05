import torch
import torch.nn as nn
import torch.nn.functional as F

# Let's fix the VectorQuantizer implementation and verify it works with a mock input.

class VectorQuantizer(nn.Module):
    """
    Vector quantization using a codebook.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        # Flatten input
        flat_inputs = inputs.view(-1, self.embedding_dim)

        # Calculate distances between inputs and codebook embeddings
        distances = (torch.sum(flat_inputs ** 2, dim=1, keepdim=True) 
                     + torch.sum(self.codebook.weight ** 2, dim=1) 
                     - 2 * torch.matmul(flat_inputs, self.codebook.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.codebook.weight).view(inputs.size())
        
        # Calculate VQ Loss
        e_latent_loss = F.mse_loss(quantized, inputs.detach())
        q_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Add the quantized values to the input as a residual
        quantized = inputs + (quantized - inputs).detach()

        return loss, quantized


class VQEncoder(nn.Module):
    """
    Encodes the input with a VQ layer after max-pooling.
    """
    def __init__(self, input_channels, num_embeddings, embedding_dim, commitment_cost):
        super(VQEncoder, self).__init__()
        self.max_pool = nn.MaxPool1d(kernel_size=8, stride=8)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x):
        x = self.max_pool(x)
        loss, quantized = self.vq_layer(x)
        return loss, quantized

# Example hyperparameters
num_embeddings = 512  # Number of embeddings in VQ layer
embedding_dim = 64   # Dimension of each embedding vector
commitment_cost = 0.25  # Beta for commitment loss

# Mock input for testing (batch_size, channels, time)
mel_spectrogram = torch.randn(16, 80, 128)

# VQ Encoder
vq_encoder = VQEncoder(input_channels=80, num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)

# Reshape the input as expected (batch_size, time, channels)
mel_spectrogram = mel_spectrogram.transpose(1, 2)

# Forward pass
vq_loss, quantized_output = vq_encoder(mel_spectrogram)

print(f"VQ Loss: {vq_loss}")
print(f"Quantized Output Shape: {quantized_output.shape}")
# Should print: Quantized Output Shape: torch.Size([16, 16, 64]) after max-pooling and VQ.
