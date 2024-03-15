import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MelGenerator(nn.Module):
    def __init__(self, num_blocks=5, hidden_size=512, kernel_size=5):
        super(MelGenerator, self).__init__()
        
        # Create a sequence of convolutional blocks
        self.conv_blocks = nn.Sequential(
            *[ConvolutionalBlock(hidden_size , hidden_size, kernel_size) for i in range(num_blocks)]
        )

    def forward(self, x):
        return self.conv_blocks(x)

if __name__ == "__main__":
    # Example usage
    # Assuming mel-spectrogram input size [batch, channels, time], e.g., [1, 1, 400]
    mel_input = torch.randn(1, 1, 400)

    # Initialize the Mel Generator
    mel_generator = MelGenerator()

    # Pass the mel-spectrogram input through the Mel Generator
    mel_output = mel_generator(mel_input)

    print(mel_output.shape)  # Expected shape [1, 512, 400]
