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
    def __init__(self, in_dim, out_dim, num_heads=4, temperature=0.2):
        super(GATLayer, self).__init__()
        
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.temperature = temperature
        
        self.W = nn.Parameter(torch.Tensor(num_heads, in_dim, out_dim))
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * out_dim, 1))
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        # [유지] 강제 초기화 (분산 확대)
        nn.init.xavier_uniform_(self.W, gain=1.4)
        nn.init.xavier_uniform_(self.a, gain=1.4)
        
    def forward(self, h):
        # h: (Batch, N, In)
        Batch, N, _ = h.size()
        
        # 1. 선형 변환 (Parallel execution for all heads)
        # (Batch, N, In) @ (Heads, In, Out) -> (Batch, Heads, N, Out)
        Wh = torch.einsum('bni,hio->bhno', h, self.W)
        
        # 2. Attention Score 계산 준비
        # Wh_i: (Batch, Heads, N, 1, Out)
        # Wh_j: (Batch, Heads, 1, N, Out)
        Wh_i = Wh.unsqueeze(3).repeat(1, 1, 1, N, 1)
        Wh_j = Wh.unsqueeze(2).repeat(1, 1, N, 1, 1)
        
        # (Batch, Heads, N, N, 2*Out)
        Wh_concat = torch.cat([Wh_i, Wh_j], dim=-1)
        
        # 3. Attention Score (e) 계산
        # (Batch, Heads, N, N, 2*Out) @ (Heads, 2*Out, 1) -> (Batch, Heads, N, N)
        # a_sq: (Heads, 2*Out)
        a_sq = self.a.squeeze(-1)
        e = torch.einsum('bhnmf,hf->bhnm', Wh_concat, a_sq)
        e = self.leakyrelu(e)
        
        # 4. Temperature Scaling & Softmax
        # Temperature를 적용하여 Attention 분포를 더 뾰족하게(Peaky) 만듦
        alpha = F.softmax(e / self.temperature, dim=3) # dim 3 is Target Neighbors
        
        # 5. Aggregation
        # (Batch, Heads, N, N) @ (Batch, Heads, N, Out) -> (Batch, Heads, N, Out)
        h_prime_heads = torch.matmul(alpha, Wh)
        
        # 6. Multi-Head 결합 (Averaging)
        # 차원 유지를 위해 Mean 사용 (Batch, N, Out)
        h_prime = h_prime_heads.mean(dim=1)
        
        # 시각화를 위해 Attention Weight도 평균 (Batch, N, N)
        alpha_avg = alpha.mean(dim=1)
        
        # 디버깅용: Logit 값 범위 확인 (필요시 주석 해제)
        # print(f"Logit range: {e.min().item():.2f} ~ {e.max().item():.2f}")
        
        return h_prime, alpha_avg
    
    
class IDSSM(nn.Module):
    def __init__(self, num_sensors, latent_dim=8, sensor_emb_dim=4, num_heads=4):
        super(IDSSM, self).__init__()
        self.num_sensors = num_sensors
        
        self.sensor_embedding = nn.Embedding(num_sensors, sensor_emb_dim)
        
        self.gat = GATLayer(
            in_dim=1 + sensor_emb_dim, 
            out_dim=latent_dim,
            num_heads=num_heads,
            temperature=0.2 # 0.2 ~ 0.5 사이 값 권장
        )
        
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
        
        # 센서 인덱스 생성
        sensor_idx = torch.arange(num_nodes, device=device)
        
        # 센서 임베딩
        sensor_emb = self.sensor_embedding(sensor_idx).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 입력 데이터 결합
        h = x_sensors.unsqueeze(-1)
        h_combined = torch.cat([h, sensor_emb], dim=-1)
        
        # GAT 통과
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