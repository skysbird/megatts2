from models.mega2 import Mega2

if __name__ == '__main__':
    megatts = Mega2(
        g_ckpt='/data/sky/gan.ckpt',
        g_config='configs/config_gan.yaml',
        plm_ckpt='/data/sky/plm_m.ckpt',
        plm_config='configs/config_plm.yaml',
        adm_ckpt='/data/sky/adm.ckpt',
        adm_config='configs/config_adm.yaml',
        symbol_table='/data/sky/data/ds/unique_text_tokens.k2symbols'
    )

    megatts.eval()

#    text = 'Also, a popular contrivance whereby'
#    text = 'Do you think this day is a good day?'
    text = 'I wish I could be more like you.'

    megatts.infer(
        '/data/sky/data/wavs/986/',
        text,      
        '/data/sky/data/wavs/986/986_129388_000009_000000.wav'
#        '/data/sky/my.wav'
#        '/data/sky/data/wavs/16/16_122827_000001_000000.wav'
    )

    # text = 'Also, a popular contrivance whereby'
    # megatts(
    #     '/data/sky/data/wavs/121/',
    #     text
    # )
    
    # megatts.forward2(
    #    '/data/sky/data/wavs/121/',
    #    'Painful to hear.'
    # )
