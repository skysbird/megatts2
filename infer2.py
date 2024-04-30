from models.mega2 import Mega2

if __name__ == '__main__':
    megatts = Mega2(
        g_ckpt='/data/sky/gan.ckpt',
        g_config='configs/config_gan.yaml',
        plm_ckpt='/data/sky/plm.ckpt',
        plm_config='configs/config_plm_main.yaml',
        adm_ckpt='/data/sky/adm1.ckpt',
        adm_config='configs/config_adm_main.yaml',
        symbol_table='/data/sky/data/ds/unique_text_tokens.k2symbols'
    )

    megatts.eval()

    #text = 'Also, a popular contrivance whereby'
    text = 'Do you think this day is a good day?'

    megatts.infer(
        '/data/sky/data/wavs/986/',
        text,      
        '/data/sky/my.wav'
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
