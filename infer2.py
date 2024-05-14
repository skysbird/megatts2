from modules.tokenizer import (
    HIFIGAN_SR,
    HIFIGAN_HOP_LENGTH,
)
import torchaudio
from models.mega2 import Mega2

if __name__ == '__main__':
    megatts = Mega2(
        g_ckpt='/data/sky/gan.ckpt',
        g_config='configs/config_gan.yaml',
        plm_ckpt='/data/sky/plm_27420.ckpt',
        plm_config='configs/config_plm.yaml',
        adm_ckpt='/data/sky/adm.ckpt',
        adm_config='configs/config_adm.yaml',
        dp_ckpt='/data/sky/dp_p.ckpt',
        dp_config='configs/config_adm_dp.yaml',

        symbol_table='/data/sky/data/ds/unique_text_tokens.k2symbols'
    )

    megatts.eval()

#    text = 'Also, a popular contrivance whereby'
#    text = 'Do you think this day is a good day?'
#    text = 'I wish I could be more like you.'
#    text = "There could be little art in this last and final round of fencing."
#    text = 'And lay me down in thy cold bed and leave my shining lot.' 
#    text = 'Number ten, fresh nelly is waiting on you, good night husband.'
    text = 'He said nothing about his riches to his eldest daughters, for he knew very well it would at once make them want to return to town; but he told Beauty his secret, and she then said, that while he was away, two gentlemen had been on a visit to their cottage, who had fallen in love with her two sisters.'

    audio = megatts.infer(
        '/data/sky/data/wavs/986/',
        text,      
#        '/data/sky/data/wavs/4145/4145_104606_000049_000000.wav'
        '/data/sky/my.wav'
#        '/data/sky/data/wavs/16/16_122827_000001_000000.wav'
#        '/data/sky/megatts2/xtts_aishell2libri_34_ch_prompts.wav'
#        '/data/sky/megatts2/wavs/librispeech_908-157963-0027_gt.wav'
    )



    torchaudio.save(f'test.wav', audio[0], HIFIGAN_SR)

    # text = 'Also, a popular contrivance whereby'
    # megatts(
    #     '/data/sky/data/wavs/121/',
    #     text
    # )
    
    # megatts.forward2(
    #    '/data/sky/data/wavs/121/',
    #    'Painful to hear.'
    # )
