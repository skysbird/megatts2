import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from new_modules.embedding import SinePositionalEmbedding
from utils.utils import make_attn_mask
from modules.transformer import TransformerEncoder, TransformerEncoderLayer

from utils.utils import instantiate_class
import yaml 

class ADM(nn.Module):
    def __init__(self, 
                    d_model=512, 
                    nhead=8, 
                    num_layers=8,
                    dt_emb_dim = 256, #dt emb_dim
                    tc_emb_dim = 256,
                    tc_latent_dim = 512,
                    dropout = 0.1
                 ):
        
        super().__init__()
        self.nhead = nhead
        d_model = dt_emb_dim + tc_emb_dim


        self.duration_embedding = nn.Linear(1, dt_emb_dim, bias=False)
        self.tc_emb = nn.Linear(tc_latent_dim, tc_emb_dim, bias=False)

        self.pos_encoder = SinePositionalEmbedding(d_model)
        # self.decoder_layers = TransformerDecoderLayer(self.dd_model, nhead, dim_feedforward, batch_first=True)
        # self.transformer_decoder = TransformerDecoder(self.decoder_layers, num_layers)
        self.transformer_decoder = TransformerEncoder(
            TransformerEncoderLayer(
                dim=d_model,
                ff_dim=dt_emb_dim * 4,
                n_heads=nhead,
                dropout=dropout,
                conv_ff=False,
            ),
            num_layers=num_layers,
        )

        self.output_layer = nn.Linear(d_model, 1, bias=False)  # Output layer for duration prediction

    # def forward(self, src, memory):
    #     src = self.pos_encoder(src)
    #     output = self.transformer_decoder(src, memory)
    #     return self.output_layer(output)
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones((sz, sz)), diagonal=1).fill_(float('-inf'))
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(
            self,
            tc_latents: torch.Tensor,  # (B, T, D)
            duration_tokens: torch.Tensor,  # (B, T)
            lens: torch.Tensor,  # (B,)
    ):

       
        #要看一下这个tc_latents到底包不包含mrte的所有输入？mrte里面是存在一个GE的，会拉长整体的序列T维度的长度
        duration_embeddings = self.duration_embedding(duration_tokens[:, :-1])
        tc_embeddings = self.tc_emb(tc_latents)

        #print(duration_embeddings.shape)
        #print(tc_latents.shape)
        x_emb = torch.cat([tc_embeddings, duration_embeddings], dim=-1)
        #print(x_emb.shape)
        x_pos = self.pos_encoder(x_emb)

        # 生成掩码
        # batch_size, seq_len = duration_tokens.size(0), duration_tokens.size(1)
        # tgt_mask = self.generate_square_subsequent_mask(seq_len).to(duration_tokens.device)

        # 只有序列中非padding部分参与计算，所以需要一个key_padding_mask
        # key_padding_mask = torch.arange(seq_len).expand(batch_size, seq_len) >= lens.unsqueeze(1)
        
        # print(tgt_mask.shape)
        # print(key_padding_mask.shape)
        # print(key_padding_mask)

        # dummy_memory = torch.zeros((seq_len, batch_size, self.dd_model), device=tc_latents.device)

        # output = self.transformer_decoder(tgt=x_pos, memory=dummy_memory, tgt_mask=tgt_mask, tgt_key_padding_mask=key_padding_mask)

        # print(x_pos.shape)
        # print(lens.shape)
        x = self.transformer_decoder(x_pos, lens, causal=True)


        duration_tokens_predict = self.output_layer(x)[..., 0]

        target = duration_tokens[:, 1:, 0]

        return duration_tokens_predict, target 
    
    @classmethod
    def from_pretrained(self, ckpt: str, config: str) -> "MegaADM":

        with open(config, "r") as f:
            config = yaml.safe_load(f)

            adm_config = config['model']['adm']
            adm = instantiate_class(args=(), init=adm_config)

        state_dict = {}
        for k, v in torch.load(ckpt)['state_dict'].items():
            if k.startswith('adm.'):
                state_dict[k[4:]] = v

        adm.load_state_dict(state_dict, strict=True)
        return adm

if __name__=='__main__':
    # Example usage
    # src has shape [seq_len, batch_size, d_model], for example:
    seq_len, batch_size, d_model = 10, 1, 2048
    # src = torch.randint(seq_len, batch_size, d_model)
    # memory = torch.randint(seq_len, batch_size, d_model)

    tc_latents = torch.randn(batch_size, seq_len, d_model//2)  # 假设已经嵌入了一半的维度
    duration_tokens = torch.randint(0, 256, (batch_size, seq_len))
    lens = torch.tensor([10])  # 每个序列的实际长度

    adm_model = ADM()
    p,t = adm_model(tc_latents, duration_tokens, lens)
    print(p.shape)  # Should be [seq_len, batch_size, 1] where last dimension is the predicted duration
    print(t.shape)
