# Importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from new_modules.content_encoder2 import FastSpeechContentEncoder
from new_modules.mrte2 import MRTE2
from new_modules.mel_decoder import MelDecoder
# from new_modules.vq_prosody_encoder import VQProsodyEncoder
from modules.vqpe import VQProsodyEncoder
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


from plm import PLMModel
from adm import ADM
from modules.convnet import ConvNet
import yaml
from utils.utils import instantiate_class
from new_modules.mrte2 import LengthRegulator
from transformer.Models import Encoder,Decoder
import hparams as hp
from transformer.Layers import Linear, PostNet
from .modules import  CBHG


class GANDiscriminator(nn.Module):
    # Placeholder for the actual GAN Discriminator implementation
    pass


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask

class VQGANTTS(nn.Module):
    def __init__(self,
                 content_encoder: FastSpeechContentEncoder,
                 mrte:MRTE2,
                 vqpe: VQProsodyEncoder,
                 mel_decoder: ConvNet
    ):
        super(VQGANTTS, self).__init__()
        self.content_encoder = content_encoder # ContentEncoder()
        self.mel_decoder = mel_decoder # MelDecoder(first_channel=512 + 512, last_channel = 80) #vq and mrte dim
        self.length_regulator = LengthRegulator()
        self.mrte = mrte
        self.vqpe = vqpe
        self.repeat_times = (512 + 384 - 1) // 384
        self.up_conv1d = nn.Conv1d(384 * self.repeat_times, 512, kernel_size=1)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=512, num_heads=2)


    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)
    
   
    def forward(self, duration_tokens, text, ref_audio, ref_audios):

        # Content Encoder forward pass
        # self.content_encoder.forward(src_seq, src_seq)
        content_features = self.content_encoder(text)
        
        ref_audio = ref_audio.permute(0,2,1)
        ref_audios = ref_audios.permute(0,2,1)

        # Forward pass through the MRTE module
        mrte_features = self.mrte(content_features, ref_audio, ref_audios, duration_tokens)


        content_features = content_features.permute(1,0,2)
        #mrte_features = mrte_features.permute(2,0,1)

        #print("c",content_features.shape)
        #print("m",mrte_features.shape)

        # attension
        attn_output, _ = self.multihead_attention(content_features, mrte_features, mrte_features)  # [B, T, mel_dim]

        # concat
        attn_output = attn_output.permute(0,1,2)

        combined_output = content_features + attn_output   # [B, T*target_length, mel_dim+global_dim]

        combined_output = combined_output.permute(1,0,2)


        print("m",mrte_features.shape)
        #上采样
        mrte_features = self.length_regulator(combined_output, duration_tokens)  # [ T*target_length, B,mel_dim]

        print("m",mrte_features.shape)
        # ref_audio = ref_audio.permute(0,2,1)
        ref_audio = ref_audio.permute(0,2,1)
        print("r",ref_audio.shape)
        prosody_features,loss, _,_  = self.vqpe(ref_audio)

        # prosody_features,loss, _,  = self.vqpe(ref_audio)


        print("p",prosody_features.shape)
        # 使用repeat函数沿特征维度重复vq_output
        #vq_output_repeated = prosody_features.repeat(1, self.repeat_times, 1)
        #print("v",vq_output_repeated.shape)

        # 使用一个1x1卷积来降维到512
        
        #prosody_features = self.up_conv1d(vq_output_repeated)
        #prosody_features =prosody_features.permute(0,2,1)
        print("p",prosody_features.shape)
        x = torch.cat([mrte_features,prosody_features],dim=-1)

        x = x.permute(0,2,1) #B D T
        mel_output = self.mel_decoder(x)
        mel_output = mel_output.permute(0,2,1)
        
        return mel_output,loss


    def discriminate(self, mel):
        # GAN Discriminator forward pass
        return self.gan_discriminator(mel)
    

    @classmethod
    def from_hparams(self, config_path: str) -> 'VQGANTTS':

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

            G_config = init = config['model']['G']

            mrte = instantiate_class(
                args=(), init=G_config['init_args']['mrte'])
            vqpe = instantiate_class(
                args=(), init=G_config['init_args']['vqpe'])

            content_encoder = instantiate_class(
                args=(), init=G_config['init_args']['content_encoder'])
            mel_decoder = instantiate_class(
                args=(), init=G_config['init_args']['mel_decoder'])


            G_config['init_args']['mrte'] = mrte
            G_config['init_args']['vqpe'] = vqpe
            G_config['init_args']['content_encoder'] = content_encoder
            G_config['init_args']['mel_decoder'] = mel_decoder

            G = instantiate_class(args=(), init=G_config)

            return G
        
    @classmethod
    def from_pretrained(self, ckpt: str, config: str) -> "VQGANTTS":

        G = VQGANTTS.from_hparams(config)

        state_dict = {}
        for k, v in torch.load(ckpt)['state_dict'].items():
            if k.startswith('G.'):
                state_dict[k[2:]] = v

        G.load_state_dict(state_dict, strict=True)
        return G

if __name__=='__main__':    # Example of usage
    text_input = torch.randint(0, 50, (122,)).unsqueeze(0)  # Random text input sequence
    ref_audio = torch.randn(1, 120, 80)  # Random reference audio in mel-spectrogram format

    ref_audios = torch.randn(1,666,80)  # Random reference audio in mel-spectrogram format

    # print("ttt")
    # print(text_input.shape)
    # Create the VQ-GAN TTS model
    vq_gan_tts_model = VQGANTTS()

    duration_length = [[120]] * 1
    duration_tokens = torch.tensor(duration_length).to(
            dtype=torch.int32)

    # Perform a forward pass through the model
    tts_output,loss = vq_gan_tts_model(duration_tokens, text_input, ref_audio, ref_audios)

    # Discriminator step (for training the GAN)
    # discriminated_output = vq_gan_tts_model.discriminate(tts_output)

    # print(tts_output.shape)  # Output from the TTS model
    # print(loss)
    # print(discriminated_output.shape)  # Output from the Discriminator

