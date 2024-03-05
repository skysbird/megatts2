import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class VectorQuantization(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantization, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Convert inputs from BxCxT to BxTxC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantized and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Convert quantized from BxTxC back to BxCxT
        quantized = quantized.permute(0, 2, 1).contiguous()

        return loss, quantized, perplexity, encodings

class VQProsodyEncoder(nn.Module):
    def __init__(self, hidden_size=320, conv_kernel_size=5, num_vq_embeddings=2048, vq_embedding_dim=256):
        super(VQProsodyEncoder, self).__init__()
        self.conv_blocks = nn.Sequential(*[ConvBlock(hidden_size if i else 1, hidden_size, conv_kernel_size) for i in range(5)])
        self.vq_layer = VectorQuantization(num_embeddings=num_vq_embeddings, embedding_dim=vq_embedding_dim)

    def forward(self, x):
        x = self.conv_blocks(x)
        loss, quantized, perplexity, _ = self.vq_layer(x)
        return loss, quantized, perplexity

if __name__ == "__main__":
    # Example input tensor (batch size, channels, sequence length)
    x = torch.randn(1, 1, 100)  # Example mel-spectrogram input

    # Initialize VQProsodyEncoder
    encoder = VQProsodyEncoder()
    loss, quantized, perplexity = encoder(x)

    print('Loss:', loss)
    print('Quantized shape:', quantized.shape)
    print('Perplexity:', perplexity)
