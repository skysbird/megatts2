import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from modules.transformer import TransformerEncoder, TransformerEncoderLayer

from new_modules.embedding import SinePositionalEmbedding
from models.gan import VQGANTTS
from models.adm import ADM
from models.plm import PLMModel
from modules.mrte import LengthRegulator
import librosa
from modules.tokenizer import (
    HIFIGAN_SR,
    HIFIGAN_HOP_LENGTH,
)
from speechbrain.pretrained import HIFIGAN

from montreal_forced_aligner.models import G2PModel, ModelManager

from montreal_forced_aligner.g2p.generator import (
    PyniniConsoleGenerator
)
from modules.datamodule import TokensCollector
from modules.tokenizer import extract_mel_spec, TextTokenizer, HIFIGAN_SR, HIFIGAN_HOP_LENGTH

import torch.nn.functional as F
import torchaudio
from einops import rearrange
import glob



language = "english_us_mfa"

# If you haven't downloaded the model
manager = ModelManager()
manager.download_model("g2p", language)

def make_g2p(text):
    g2p_model_path = G2PModel.get_pretrained_path(language)
    
    g2p = PyniniConsoleGenerator(
                g2p_model_path=g2p_model_path,
                num_pronunciations=1
            )
    g2p.setup()
    
    word =text.lower()
    pronunciations = g2p.rewriter(word)
    [print(p) for p in pronunciations]
    return pronunciations[0].split()

class Mega2(nn.Module):
    def __init__(
        self,
        g_ckpt: str,
        g_config: str,
        plm_ckpt: str,
        plm_config: str,
        adm_ckpt: str,
        adm_config: str,
        symbol_table: str
    ):
        super(Mega2, self).__init__()

        self.generator = VQGANTTS.from_pretrained(g_ckpt, g_config)
        self.generator.eval()
        self.plm = PLMModel.from_pretrained(plm_ckpt, plm_config)
        self.plm.eval()
        self.adm = ADM.from_pretrained(adm_ckpt, adm_config)
        self.adm.eval()

        self.ttc = TokensCollector(symbol_table)

        self.lr = LengthRegulator(
            HIFIGAN_HOP_LENGTH, 16000, (HIFIGAN_HOP_LENGTH / HIFIGAN_SR * 1000))

        self.hifi_gan = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz")
        self.hifi_gan.eval()

    def forward(
            self,
            wavs_dir: str,
            text: str,
            ref_wav: str
    ):
        mels_prompt = None

        #wav = '/data/sky/data/wavs/121/121_121726_000004_000003.wav'
        if ref_wav:
            print("ref"+ref_wav)
            y = librosa.load(ref_wav, sr=HIFIGAN_SR)[0]
            y = librosa.util.normalize(y)
            #y = librosa.effects.trim(y, top_db=20)[0]
            y = torch.from_numpy(y)

            mel_spec = extract_mel_spec(y).transpose(0, 1)

            mels_prompt = mel_spec
        #mels_prompt = None

        
        # Make mrte mels
        wavs = glob.glob(f'{wavs_dir}/*.wav')
        mels = torch.empty(0)
        for wav in wavs:
            y = librosa.load(wav, sr=HIFIGAN_SR)[0]
            y = librosa.util.normalize(y)
            # y = librosa.effects.trim(y, top_db=20)[0]
            y = torch.from_numpy(y)

            mel_spec = extract_mel_spec(y).transpose(0, 1)
            mels = torch.cat([mels, mel_spec], dim=0)

            if mels_prompt is None:
                mels_prompt = mel_spec
                print("p_"+wav)
            print(wav)

        mels = mels.unsqueeze(0)

        # G2P
        ps = make_g2p(text)
        print(ps)
        
        
        phone_tokens = self.ttc.phone2token(
            ps)
        print(phone_tokens)
        phone_tokens = phone_tokens.unsqueeze(0)
   
        print(phone_tokens.shape)
        with torch.no_grad():
            mels = mels.permute(0,2,1)
            mels_prompt = mels_prompt.unsqueeze(0).permute(0,2,1)
            phone_tokens = self.generator.content_encoder(phone_tokens)
            print(phone_tokens.shape)
            tc_latent = self.generator.mrte.tc_latent(phone_tokens, mels_prompt, mels, None)
            print("t1",tc_latent)
            dt = self.adm.infer(tc_latent)[..., 0]
            tc_latent_expand = self.lr(tc_latent, dt)

            # tc_latent = self.generator.mrte(dt, phone_tokens, mels)
            
            tc_latent = F.max_pool1d(tc_latent_expand.transpose(
                1, 2), 8, ceil_mode=True).transpose(1, 2)

            print("ttt",tc_latent)

            p_codes = self.plm.infer(tc_latent)

            print("pppp",p_codes)
            zq = self.generator.vq_prosody_encoder.vq.decode(p_codes.unsqueeze(0))
            zq = rearrange(
                zq, "B D T -> B T D").unsqueeze(2).contiguous().expand(-1, -1, 8, -1)
            zq = rearrange(zq, "B T S D -> B (T S) D")
            x = torch.cat(
                [tc_latent_expand, zq[:, :tc_latent_expand.shape[1], :]], dim=-1)
            print(x.shape)
            x = rearrange(x, 'B T D -> B D T')
            x = self.generator.mel_decoder(x)

            audio = self.hifi_gan.decode_batch(x.cpu())
 
            torchaudio.save('test.wav', audio[0], HIFIGAN_SR)
