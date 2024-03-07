import torch
import torch.nn as nn
import torch.nn.functional as F
from .mel_encoder import MelGenerator

# 假设全局编码器(GE)是一个简单的全连接层
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalEncoder(nn.Module):
    def __init__(self, num_layers=5, hidden_size=512, kernel_size=31, first_channels=80):
        super(GlobalEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(in_channels=first_channels if i == 0 else hidden_size, 
                      out_channels=hidden_size, 
                      kernel_size=kernel_size, 
                      padding=kernel_size // 2)
            for i in range(num_layers)
        ])
        self.activation = nn.ReLU()

    def forward(self, mel_spectrogram):
        x = mel_spectrogram
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        # Global average pooling to create a fixed size output regardless of the input length
        # x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return x

# # Example usage:
# # Assuming mel-spectrogram input of size [batch_size, 1, time_steps]
# batch_size, time_steps = 1, 400
# mel_spectrogram = torch.randn(batch_size, 1, time_steps)  # Example input

# # Create the Timbre Encoder
# encoder = TimbreEncoder()

# # Forward pass through the Timbre Encoder
# timbre_features = encoder(mel_spectrogram)

# print(timbre_features.shape)  # Expected output shape: [batch_size, hidden_size]


# Length Regulator 调整序列长度
class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()
        # Length regulator的具体实现取决于具体应用，这里我们使用一个简单的例子

    def forward(self, x, target_length):
        # 简单重复序列来匹配目标长度
        print(target_length)
        print(x.shape)
        return x.repeat_interleave(target_length, dim=0)  # [B, T*target_length, D]

# 定义一个MRTE，这里我们假设Mel Encoder输出和Multi-Head Attention的结构和维度
class MRTE(nn.Module):
    def __init__(self, mel_dim, global_mel_dim, hidden_size, n_heads):
        super(MRTE, self).__init__()

        self.mel_conv = nn.Conv1d(in_channels=mel_dim, out_channels=hidden_size, kernel_size=3, padding=1)
        self.mel_encoder = MelGenerator() 
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads)
        self.global_encoder = GlobalEncoder(first_channels=global_mel_dim)
        self.length_regulator = LengthRegulator()

    def forward(self, mel_spec, global_mel_spec, target_length):

        # 先卷到目标维
        mel_spec = self.mel_conv(mel_spec)
        print(mel_spec.shape)
        # Mel Encoder
        mel_encoded = self.mel_encoder(mel_spec)  # [B, T, mel_dim]
        print(mel_encoded.shape)

        mel_encoded = mel_encoded.permute(2,0,1)
        # Multi-Head Attention
        attn_output, _ = self.multihead_attention(mel_encoded, mel_encoded, mel_encoded)  # [B, T, mel_dim]

        # Global Encoder
        global_features = self.global_encoder(global_mel_spec)  # [B, global_dim]
        # global_features = global_features.unsqueeze(1).expand(-1, attn_output.size(1), -1)  # [B, T, global_dim]
        # global_features = self.global_encoder(attn_output.mean(dim=0))

        # Length Regulator
        global_features = global_features.permute(2,0,1)
        attn_output = attn_output.permute(0,1,2)
        
        print(attn_output.shape)
        print(global_features.shape)
        # 合并Attention输出和全局特征
        combined_output = torch.cat((attn_output, global_features), dim=0)  # [B, T*target_length, mel_dim+global_dim]
        print(combined_output.shape)

        combined_output = combined_output.permute(1,0,2)
        print(combined_output.shape)

        #TODO c
        regulated_output = self.length_regulator(combined_output, target_length)  # [ T*target_length, B,mel_dim]

        return combined_output

if __name__=='__main__':
    # Example usage
    # Assuming mel-spectrogram with
    mel_dim = 80
    global_mel_dim = 80
    hidden_size = 512
    mrte = MRTE(mel_dim=mel_dim,global_mel_dim=global_mel_dim,hidden_size=hidden_size, n_heads=2)
    # Create a batch of test Mel spectrograms (batch_size, channels, time)
    test_mels = torch.randn(4, mel_dim, 120)  # Example with 4 items in a batch and 120 time-steps

    # Assume the target length for each item after the length regulator is fixed at 100 for this test
    regulated_lengths = torch.full((4,), 100, dtype=torch.long)  # Example target lengths

    # print(regulated_lengths.shape)
    # Forward pass through the MRTE module
    mrte_output = mrte(test_mels, test_mels, regulated_lengths)

    print(f"MRTE output shape: {mrte_output.shape}")  # Expected shape: (target_length, batch_size, hidden_size)
    # print(mrte_output)