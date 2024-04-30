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

        self.pos_encoder = SinePositionalEmbedding(d_model,dropout=dropout)
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
            norm = nn.ReLU()
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
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
        duration_embeddings = self.duration_embedding(duration_tokens[:, :-1])
        tc_embeddings = self.tc_emb(tc_latents)

        #print(duration_embeddings.shape)
        #print(tc_latents.shape)
        x_emb = torch.cat([tc_embeddings, duration_embeddings], dim=-1)
        #print(x_emb.shape)
        x_pos = self.pos_encoder(x_emb)

        x = self.transformer_decoder(x_pos, lens, causal=True)

        x = self.norm(x)

        duration_tokens_predict = self.output_layer(x)[..., 0]

        duration_tokens_predict = self.relu(duration_tokens_predict)

        target = duration_tokens[:, 1:, 0]

        return duration_tokens_predict, target 
    
    def infer(
        self,
        tc_latents: torch.Tensor,  # (B, T, D)
    ):
        T = tc_latents.shape[1]
        p_code = torch.Tensor([0]).to(
            tc_latents.device).unsqueeze(0).unsqueeze(1)
        for t in range(T):
            dt_emb = self.duration_embedding(p_code)
            tc_emb = self.tc_emb(tc_latents[:, 0:t+1, :])

            x_emb = torch.cat([tc_emb, dt_emb], dim=-1)
            x_pos = self.pos_encoder(x_emb)

            x = self.transformer_decoder(x_pos)

            x = self.norm(x)

            dt_predict = self.output_layer(x)[:, -1:, :]
        
            dt_predict = self.relu(dt_predict)

            p_code = torch.cat([p_code, dt_predict], dim=1)

        return (p_code[:, 1:, :] + 0.5).to(torch.int32).clamp(1, 128)

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
