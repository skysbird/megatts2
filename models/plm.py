import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from modules.transformer import TransformerEncoder, TransformerEncoderLayer

from new_modules.embedding import SinePositionalEmbedding
from utils.utils import instantiate_class
import yaml 

class PLMModel(nn.Module):
    def __init__(self, 
                 d_model=2048, 
                 n_heads=16, 
                 n_layers=24, 
                 vq_bins = 1024,
                 vq_dim = 512,
                 tc_latent_dim: int = 512,
                 dropout=0.1):
        super(PLMModel, self).__init__()
       
        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        # self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        td_model = vq_dim + tc_latent_dim

        self.plm = TransformerEncoder(
            TransformerEncoderLayer(
                dim=td_model,
                ff_dim=td_model * 4,
                n_heads=n_heads,
                dropout=dropout,
                conv_ff=False,
            ),
            num_layers=n_layers,
        )

        self.positional_encoding =  SinePositionalEmbedding(
            dim_model=td_model,
        )

        self.pc_embedding = nn.Embedding(vq_bins + 2, vq_dim)

        self.output_layer = nn.Linear(td_model, td_model, bias=False)

    def forward(
            self,
            tc_latent: torch.Tensor,  # (B, T, D)
            p_codes: torch.Tensor,  # (B, T)
            lens: torch.Tensor,  # (B,)
    ):
        pc_emb = self.pc_embedding(p_codes[:, :-1])
        x_emb = torch.cat([tc_latent, pc_emb], dim=-1)
        x_pos = self.positional_encoding(x_emb)

        x = self.plm(x_pos, lens, causal=True)
        logits = self.output_layer(x)

        target = p_codes[:, 1:]

        #print(logits.shape)
        #print(target.shape)
        return logits, target
    
    @classmethod
    def from_pretrained(cls, ckpt: str, config: str) -> "PLM":

        with open(config, "r") as f:
            config = yaml.safe_load(f)

            plm_config = config['model']['plm']
            plm = instantiate_class(args=(), init=plm_config)

        state_dict = {}
        for k, v in torch.load(ckpt)['state_dict'].items():
            if k.startswith('plm.'):
                state_dict[k[4:]] = v

        plm.load_state_dict(state_dict, strict=True)
        return plm

if __name__=='__main__':
    # Assuming the inputs are:
    # src with shape [sequence_length, batch_size, d_model]
    # memory (encoder outputs) with shape [sequence_length, batch_size, d_model]
    sequence_length, batch_size, d_model = 100, 2, 2048
    src = torch.rand(sequence_length, batch_size, d_model)
    memory = torch.rand(sequence_length, batch_size, d_model)

    # Create the PLM model
    plm_model = PLMModel()

    # Forward pass through the PLM model
    output = plm_model(src, memory)

    print(output.shape)  # Expected shape: [sequence_length, batch_size, d_model]
