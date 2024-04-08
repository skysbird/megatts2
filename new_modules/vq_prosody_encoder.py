import torch
import torch.nn as nn
import torch.nn.functional as F
from .mrte2 import LayerNormChannels
from einops import rearrange
from modules.convnet import ConvNetDouble

class VectorQuantiser(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, num_embed, embed_dim, beta, distance='cos', 
                 anchor='probrandom', first_batch=False, contras_loss=False):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.dimension = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))

    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        #z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z = rearrange(z, "b d n -> b n d")
        print(z.shape)
        print(self.embed_dim)
        z = z.contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        print("dis",self.distance)
        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:,-1]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
  
        # preserve gradients
        z_q = z + (z_q - z).detach()


        commit_loss = F.mse_loss(z_q.detach(), z)
        print("loss",loss)

        print("com",commit_loss)

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b n d -> b d n').contiguous()
        # count
        #import pdb
        #pdb.set_trace()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1,:]]
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss +=  contra_loss

        return z_q, loss, (perplexity, min_encodings, encoding_indices)

class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features

class VQProsodyEncoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels,
                 kernel_size, 
                 vq_commitment_cost = 0.25,
                 vq_dim = 256,
                 vq_bins = 1024,
                 vq_decay = 0.99,
                 vq_distance='cos',
                 vq_anchor='closest',
                 vq_first_batch=False,
                 vq_contras_loss=True
                ):
        super(VQProsodyEncoder, self).__init__()
        num_layers = 5
        self.input_channels = in_channels

        self.num_layers = num_layers
        self.conv1d_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=in_channels if i == 0 else hidden_channels,
                          out_channels=hidden_channels,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2),
                LayerNormChannels(hidden_channels),
                nn.GELU()
            ) for i in range(num_layers)
        ])

        self.pool = nn.MaxPool1d(kernel_size=8, stride=8, padding=0,ceil_mode=True)


        self.last_conv1d_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_channels,
                          out_channels= vq_dim if (i == num_layers - 1) else hidden_channels,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2),
                LayerNormChannels( vq_dim if (i == num_layers - 1) else hidden_channels),
                nn.GELU()
            ) for i in range(num_layers)
        ])


#  n_layers: int = 3,
#             n_stacks: int = 5,
#             n_blocks: int = 2,
        #use facebook
        self.convnet = ConvNetDouble(
            in_channels=in_channels,
            out_channels=vq_dim,
            hidden_size=hidden_channels,
            n_layers=num_layers,
            n_stacks=5,
            n_blocks=2,
            middle_layer=nn.MaxPool1d(8, ceil_mode=True),
            kernel_size=kernel_size,
            activation='ReLU',
        )

        # self.conv1d = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        # self.vq = VectorQuantizer(hidden_channels, num_embeddings, embedding_dim, commitment_cost)
            # def __init__(self, num_embed, embed_dim, beta, distance='cos', 
            #      anchor='probrandom', first_batch=False, contras_loss=False):

        self.vq = VectorQuantiser(
            num_embed=vq_bins,
            embed_dim=vq_dim,
            beta=vq_commitment_cost,
            distance=vq_distance,
            anchor=vq_anchor,
            first_batch=vq_first_batch,
            contras_loss=vq_contras_loss
        )
        

    def forward(self, mel_spec):
        # Assuming mel_spec is of shape (batch_size, channels, time)
        #x = self.conv1d(mel_spec)  # Apply Conv1D
        mel_len = mel_spec.size(2)
        print("ml",mel_spec.shape)
        mel_spec = mel_spec[:, :self.input_channels,:]

        # x = mel_spec
        # for i in range(self.num_layers):
        #     x = self.conv1d_blocks[i](x)
        
        # x = self.pool(x) 

        # for i in range(self.num_layers):
        #     x = self.last_conv1d_blocks[i](x)

        #old vq
        x = self.convnet(mel)


        quantize, loss, (perplexity, encodings, encoding_indices) = self.vq(x)
        
        vq_loss = F.mse_loss(x.detach(), quantize)

        print("q",quantize.shape)
        quantize = rearrange(quantize, "B D T -> B T D").unsqueeze(2).contiguous().expand(-1, -1, 8 , -1)
        print("q",quantize.shape)
        quantize = rearrange(quantize, "B T S D -> B (T S) D")[:, :mel_len, :]
        print("q",quantize.shape)
        #quantize = quantize.permute(0,2,1)

        #prosody_features,loss,vq_loss, _

        return quantize, loss, vq_loss, encoding_indices

# Define hyperparameters
# in_channels = 80  # Number of mel bins
# hidden_channels = 384
# kernel_size = 5
# num_embeddings = 1024
# embedding_dim = 256
# commit
