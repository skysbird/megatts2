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


def create_alignment(base_mat, duration_tokens):
    N, L = duration_tokens.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_tokens[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_tokens[i][j]
    return base_mat

# Length Regulator 调整序列长度
class LengthRegulator(nn.Module):
    """ Length Regulator from FastSpeech """

    def __init__(self):
        super(LengthRegulator, self).__init__()

        # assert (mel_frames / sample_rate * 1000 / duration_token_ms) == 1

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D)
        duration_tokens: torch.Tensor,  # (B, T) int for duration
        mel_max_length=None
    ):

        bsz, input_len, _ = x.size()

        expand_max_len = torch.max(torch.sum(duration_tokens, -1), -1)[0].int()

        alignment = torch.zeros(bsz, expand_max_len, input_len).numpy()
        alignment = create_alignment(alignment, duration_tokens.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)
        print(alignment)
        print(alignment.shape)
        print(x.shape)
        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output
    
# 定义一个MRTE，这里我们假设Mel Encoder输出和Multi-Head Attention的结构和维度
class MRTE(nn.Module):
    def __init__(self, mel_dim, global_mel_dim, hidden_size, n_heads):
        super(MRTE, self).__init__()

        self.mel_conv = nn.Conv1d(in_channels=mel_dim, out_channels=hidden_size, kernel_size=3, padding=1)
        self.mel_encoder = MelGenerator() 
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads)
        self.global_encoder = GlobalEncoder(first_channels=global_mel_dim)
        self.length_regulator = LengthRegulator()

    def forward(self, phone, mel_spec, global_mel_spec, target_length):

        # 先卷到目标维
        mel_spec = self.mel_conv(mel_spec)
        # print(mel_spec.shape)
        # Mel Encoder
        mel_encoded = self.mel_encoder(mel_spec)  # [B, T, mel_dim]
        # print(mel_encoded.shape)

        mel_encoded = mel_encoded.permute(2,0,1)
        # Multi-Head Attention
        # print("p")
        # print(phone.shape)
        # print(mel_encoded.shape)
        phone = phone.permute(1,0,2)
        attn_output, _ = self.multihead_attention(phone, mel_encoded, mel_encoded)  # [B, T, mel_dim]

        # Global Encoder
        global_features = self.global_encoder(global_mel_spec)  # [B, global_dim]
        # global_features = global_features.unsqueeze(1).expand(-1, attn_output.size(1), -1)  # [B, T, global_dim]
        # global_features = self.global_encoder(attn_output.mean(dim=0))

        # Length Regulator
        global_features = global_features.permute(2,0,1)
        attn_output = attn_output.permute(0,1,2)
        
        # print(attn_output.shape)
        # print(global_features.shape)
        # 合并Attention输出和全局特征
        combined_output = torch.cat((attn_output, global_features), dim=0)  # [B, T*target_length, mel_dim+global_dim]
        # print(combined_output.shape)

        combined_output = combined_output.permute(1,0,2)
        # print(combined_output.shape)
        # print(f"MRTE combined_output shape: {combined_output.shape}")  

        #TODO c
        regulated_output = self.length_regulator(combined_output, target_length)  # [ T*target_length, B,mel_dim]

        return regulated_output

if __name__=='__main__':
    # Example usage
    # Assuming mel-spectrogram with
    mel_dim = 80
    global_mel_dim = 80
    hidden_size = 512
    mrte = MRTE(mel_dim=mel_dim,global_mel_dim=global_mel_dim,hidden_size=hidden_size, n_heads=2)
    # Create a batch of test Mel spectrograms (batch_size, channels, time)
    test_mels = torch.randn(4, mel_dim, 120)  #  B D T Example with 4 items in a batch and 120 time-steps
    # print(test_mels.shape)
    # Assume the target length for each item after the length regulator is fixed at 100 for this test
    # regulated_lengths = torch.full((4,), 100, dtype=torch.long)  # Example target lengths

    duration_length = [[1,2,3,4]] * 4
    duration_tokens = torch.tensor(duration_length).to(
        dtype=torch.int32)
    print(duration_tokens.shape)
    # print(regulated_lengths.shape)
    # Forward pass through the MRTE module
    phone = torch.rand(4,100,512) #B T D
    mrte_output = mrte(phone,test_mels, test_mels, duration_tokens)

    print(f"MRTE output shape: {mrte_output.shape}")  # Expected shape: (target_length, batch_size, hidden_size)
    # print(mrte_output)