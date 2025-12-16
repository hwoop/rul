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
        
        nn.init.xavier_uniform_(self.W.weight, gain=3.0)
        nn.init.xavier_uniform_(self.a.weight, gain=3.0)
        
    def forward(self, h):
        Batch, N, _ = h.size()
        Wh = self.W(h)
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        Wh_concat = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.leakyrelu(self.a(Wh_concat)).squeeze(-1)
        alpha = F.softmax(e, dim=2)
        h_prime = torch.bmm(alpha, Wh)
        
        # print(e.min(), e.max(), e.mean())
        
        return h_prime, alpha
    
    
class IDSSM(nn.Module):
    def __init__(self, num_sensors, latent_dim=8, sensor_emb_dim=4): # [변경] sensor_emb_dim 추가
        super(IDSSM, self).__init__()
        self.num_sensors = num_sensors
        
        self.sensor_embedding = nn.Embedding(num_sensors, sensor_emb_dim)
        
        self.gat = GATLayer(in_dim=1 + sensor_emb_dim, out_dim=latent_dim)
        
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
        batch_size, num_nodes = x_sensors.size()
        device = x_sensors.device
        
        # [변경] 센서 인덱스 생성 (0 ~ num_sensors-1)
        sensor_idx = torch.arange(num_nodes, device=device) # (N,)
        
        # [변경] 센서 임베딩 가져오기 및 배치 크기에 맞게 확장
        # (N, emb_dim) -> (1, N, emb_dim) -> (Batch, N, emb_dim)
        sensor_emb = self.sensor_embedding(sensor_idx).unsqueeze(0).expand(batch_size, -1, -1)
        
        # [변경] 입력 데이터(값)와 임베딩 결합
        h = x_sensors.unsqueeze(-1) # (Batch, N, 1)
        h_combined = torch.cat([h, sensor_emb], dim=-1) # (Batch, N, 1 + emb_dim)
        
        # GAT에는 결합된 feature를 입력으로 줌
        h_prime, attn_weights = self.gat(h_combined) 
        
        w = self.pooling(h_prime)
        z = torch.sum(h_prime * w, dim=1)
        return z, attn_weights 
    
    def decode(self, state_x):
        return self.mnn(state_x)
    
    def forward(self, x_sensors, state_x):
        z_enc, _ = self.encode(x_sensors) 
        z_dec = self.decode(state_x)
        return z_enc, z_dec