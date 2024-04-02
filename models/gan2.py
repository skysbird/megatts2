# Importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from new_modules.content_encoder2 import ContentEncoder2
from new_modules.mrte import MRTE
# from new_modules.vq_prosody_encoder import VQProsodyEncoder
from new_modules.mel_decoder import MelDecoder
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


from plm import PLMModel
from adm import ADM
from modules.convnet import ConvNet
from modules.vqpe import VQProsodyEncoder
import yaml
from utils.utils import instantiate_class
from new_modules.mrte2 import LengthRegulator

class GANDiscriminator(nn.Module):
    # Placeholder for the actual GAN Discriminator implementation
    pass


class VQGANTTS(nn.Module):
    def __init__(self,
                 content_encoder: ContentEncoder2,
                 mel_decoder: ConvNet
    ):
        super(VQGANTTS, self).__init__()
        self.content_encoder = content_encoder # ContentEncoder()
        self.mel_decoder = mel_decoder # MelDecoder(first_channel=512 + 512, last_channel = 80) #vq and mrte dim
        self.length_regulator = LengthRegulator()

    
    def forward(self, 
                phonemes:torch.Tensor, #(B, T)
                duration_tokens: torch.Tensor #(B,)
                ):
        # Content Encoder forward pass
        content_features = self.content_encoder(phonemes)
        
        #上采样
        regulated_output = self.length_regulator(content_features, duration_tokens)  # [ T*target_length, B,mel_dim]

        x = regulated_output.permute(0,2,1)

        mel_output = self.mel_decoder(x)

        mel_output = mel_output.permute(0,2,1)
        
       
        return mel_output

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

