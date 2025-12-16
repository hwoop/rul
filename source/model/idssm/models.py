import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class MonotonicLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MonotonicLinear, self).__init__()
        
        self.weight_raw = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight_raw)
        nn.init.zeros_(self.bias)
        
    def forward(self, input):
        return F.linear(input, F.softplus(self.weight_raw), self.bias)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, h):
        Batch, N, _ = h.size()
        Wh = self.W(h)
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        Wh_concat = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.leakyrelu(self.a(Wh_concat)).squeeze(-1)
        alpha = F.softmax(e, dim=2)
        h_prime = torch.bmm(alpha, Wh)
        
        return h_prime, alpha
    
    
class IDSSM(nn.Module):
    def __init__(self, num_sensors, latent_dim=8):
        super(IDSSM, self).__init__()
        self.gat = GATLayer(in_dim=1, out_dim=latent_dim)
        self.pooling = nn.Sequential(
            nn.Linear(latent_dim, 1), 
            nn.Softmax(dim=1)
        )
        self.mnn = nn.Sequential(
            MonotonicLinear(1, 16),
            nn.Tanh(),
            MonotonicLinear(16, latent_dim)
        )
        
    def encode(self, x_sensors):
        h = x_sensors.unsqueeze(-1)
        h_prime, attn_weights = self.gat(h) 
        w = self.pooling(h_prime)
        z = torch.sum(h_prime * w, dim=1)
        return z, attn_weights 
    
    def decode(self, state_x):
        return self.mnn(state_x)
    
    def forward(self, x_sensors, state_x):
        z_enc, _ = self.encode(x_sensors) 
        z_dec = self.decode(state_x)
        return z_enc, z_dec