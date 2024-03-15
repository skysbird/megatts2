# Importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from new_modules.content_encoder import ContentEncoder
from new_modules.mrte import MRTE
from new_modules.vq_prosody_encoder import VQProsodyEncoder
from new_modules.mel_decoder import MelDecoder
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


from plm import PLMModel
from adm import ADM



class GANDiscriminator(nn.Module):
    # Placeholder for the actual GAN Discriminator implementation
    pass

class VQGANTTS(nn.Module):
    def __init__(self):
        super(VQGANTTS, self).__init__()
        self.content_encoder = ContentEncoder()
        self.mrte = MRTE( 80,80,512,2)
        self.vq_prosody_encoder = VQProsodyEncoder()
        self.mel_decoder = MelDecoder(first_channel=512 + 512, last_channel = 80) #vq and mrte dim
        self.plm = PLMModel()
        self.adm = ADM()
        # self.gan_discriminator = GANDiscriminator()
    
    def forward(self, duration_tokens, text, ref_audio, ref_audios):
        # Content Encoder forward pass
        content_features = self.content_encoder(text)
        print(content_features.shape)
        
        # mel = rearrange(mel, 'B T D -> B D T')

        ref_audio = ref_audio.permute(0,2,1)
        ref_audios = ref_audios.permute(0,2,1)

        # Forward pass through the MRTE module
        mrte_features = self.mrte(content_features, ref_audio, ref_audios, duration_tokens)

        # VQ Prosody Encoder forward pass
        # XXX ? need check
        loss,prosody_features,perp = self.vq_prosody_encoder(ref_audio)

        # Mel Decoder forward pass
        prosody_features = prosody_features.permute(0,2,1)
        x = torch.cat([mrte_features,prosody_features],dim=-1)

        x = x.permute(0,2,1)
        mel_output = self.mel_decoder(x)
        mel_output = mel_output.permute(0,2,1)
        
        # #TODO
        # sequence_length, batch_size, d_model = 100, 2, 2048
        # # src = torch.rand(sequence_length, batch_size, d_model)
        # memory = torch.rand(sequence_length, batch_size, d_model)
        
        # PLM forward pass
        # prosody_output = self.plm(prosody_features,memory)

        # # ADM forward pass
        # duration_output = self.adm(content_features)

        # Combine all outputs for final processing or return as needed
        # This is a placeholder step - actual implementation will depend on how these outputs are used together
        # combined_output = torch.cat((mel_output, prosody_output, duration_output), dim=-1)

        return mel_output,loss

    def discriminate(self, mel):
        # GAN Discriminator forward pass
        return self.gan_discriminator(mel)

if __name__=='__main__':    # Example of usage
    text_input = torch.randint(0, 50, (122,)).unsqueeze(0)  # Random text input sequence
    ref_audio = torch.randn(1, 120, 80)  # Random reference audio in mel-spectrogram format

    ref_audios = torch.randn(1,666,80)  # Random reference audio in mel-spectrogram format

    print("ttt")
    print(text_input.shape)
    # Create the VQ-GAN TTS model
    vq_gan_tts_model = VQGANTTS()

    duration_length = [[120]] * 1
    duration_tokens = torch.tensor(duration_length).to(
            dtype=torch.int32)

    # Perform a forward pass through the model
    tts_output,loss = vq_gan_tts_model(duration_tokens, text_input, ref_audio, ref_audios)

    # Discriminator step (for training the GAN)
    # discriminated_output = vq_gan_tts_model.discriminate(tts_output)

    print(tts_output.shape)  # Output from the TTS model
    print(loss)
    # print(discriminated_output.shape)  # Output from the Discriminator

