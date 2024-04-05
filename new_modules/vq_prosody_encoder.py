import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Implements the vector quantization layer.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        # Flatten input
        flat_inputs = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances between inputs and embedding weights
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_inputs, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Use the encodings to get the quantized vector
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # Compute the VQ loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Add the quantized vector to the inputs (pass gradients through)
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, e_latent_loss, encoding_indices

# Example usage:
num_embeddings = 512  # Number of embeddings in VQ layer
embedding_dim = 64   # Dimension of each embedding vector
commitment_cost = 0.25  # Beta for commitment loss

# Create the VectorQuantizer instance
vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

# Example input tensor (batch_size, channels, time)
mel_spectrogram = torch.randn(10, embedding_dim, 50)  # An example mel spectrogram

# Forward pass through the VectorQuantizer
quantized, vq_loss, commit_loss, encodings = vector_quantizer(mel_spectrogram)

print(f"Quantized Output Shape: {quantized.shape}")
print(f"VQ Loss: {vq_loss.item()}")
print(f"Commitment Loss: {commit_loss.item()}")
print(f"Encodings Shape: {encodings.shape}")
# The shapes of the outputs should match the expected shapes based on the input tensor's dimensions and VQ settings
