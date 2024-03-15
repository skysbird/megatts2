import torch
import torch.nn as nn
import torch.nn.functional as F

class MelDecoder(nn.Module):
    def __init__(self, first_channel, last_channel, num_layers=5, hidden_size=512, kernel_size=5):
        super(MelDecoder, self).__init__()
        self.num_layers = num_layers
        # Assuming the input dimension and output dimension are the same as hidden_size
        layers = [
            nn.ConvTranspose1d(first_channel if l == 0 else hidden_size, last_channel if l == num_layers-1 else hidden_size, kernel_size, stride=1, padding=kernel_size // 2)
            for l in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.leaky_relu(x)
        return x

if __name__=='__main__':
    # Example usage of MelDecoder
    # Assuming the input latent representation has shape (batch_size, hidden_size, num_mel_bins)
    batch_size, hidden_size, num_mel_bins = 16, 320, 80
    latent_representation = torch.randn(batch_size, hidden_size, num_mel_bins)

    # Initialize the MelDecoder
    mel_decoder = MelDecoder()

    # Forward pass through the MelDecoder
    reconstructed_mel = mel_decoder(latent_representation)

    print(reconstructed_mel.shape)  # Expected shape: (batch_size, hidden_size, num_mel_bins)
