from models.megatts2 import Megatts

if __name__ == '__main__':
    megatts = Megatts(
        g_ckpt='generator.ckpt',
        g_config='configs/config_gan.yaml',
        plm_ckpt='plm.ckpt',
        plm_config='configs/config_plm.yaml',
        adm_ckpt='adm.ckpt',
        adm_config='configs/config_adm.yaml',
        symbol_table='/data/sky/data/ds/unique_text_tokens.k2symbols'
    )

    megatts.eval()

    # text = 'Also, a popular contrivance whereby'
    text = 'Hello this is a test'

    megatts(
        '/data/sky/data/wavs/121/',
        text
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
