"""
ID-SSM (Interaction-aware Deep State-Space Model)

컨셉 문서 기반 구현:
1. GAT 기반 센서 퓨전 모듈: 센서 간 상관관계 학습 → 융합 관측 벡터 z_t
2. MNN 기반 단조 관측 모듈: 잠재 건강 상태 x_t와 z_t 사이의 단조 비선형 매핑

핵심 수식:
- State Transition: x_k = x_{k-1} + η·Δt_k + ω_{k-1}
- Deep Measurement: z_k = H(x_k, θ_mnn) + v_k
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicLinear(nn.Module):
    """
    단조성 제약을 가진 선형 레이어
    
    컨셉 문서 (Section 3.2):
    "함수 H가 x에 대해 단조 증가하도록 강제하기 위해, 
     모든 가중치 W^(l)_mnn가 양수여야 합니다.
     W^(l)_mnn = exp(Ŵ^(l))"
    
    변경사항: softplus → exp (컨셉 문서와 일치)
    """
    def __init__(self, in_features, out_features):
        super(MonotonicLinear, self).__init__()
        
        # 실제 학습되는 파라미터 (Ŵ)
        self.weight_raw = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Xavier 초기화 (exp 적용 전이므로 작은 값으로 시작)
        nn.init.uniform_(self.weight_raw, -0.5, 0.5)
        nn.init.zeros_(self.bias)
        
    def forward(self, input):
        # W_mnn = exp(Ŵ) - 컨셉 문서의 단조성 제약
        positive_weight = torch.exp(self.weight_raw)
        return F.linear(input, positive_weight, self.bias)
    
    def get_positive_weights(self):
        """양수 가중치 반환 (디버깅/시각화용)"""
        return torch.exp(self.weight_raw)


class GATLayer(nn.Module):
    """
    Graph Attention Layer for Sensor Fusion
    
    컨셉 문서 (Section 2.2):
    e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
    α_ij = softmax(e_ij)
    h_i' = σ(Σ α_ij · W · h_j)
    """
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(GATLayer, self).__init__()
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # 초기화
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)
        
    def forward(self, h):
        """
        Args:
            h: (Batch, N_sensors, in_dim) - 센서별 특징
        Returns:
            h_prime: (Batch, N_sensors, out_dim) - 업데이트된 특징
            alpha: (Batch, N_sensors, N_sensors) - Attention weights
        """
        Batch, N, _ = h.size()
        
        # Linear projection: Wh
        Wh = self.W(h)  # (Batch, N, out_dim)
        
        # Attention score 계산: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # (Batch, N, N, out_dim)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # (Batch, N, N, out_dim)
        Wh_concat = torch.cat([Wh_i, Wh_j], dim=-1)  # (Batch, N, N, 2*out_dim)
        
        e = self.leakyrelu(self.a(Wh_concat)).squeeze(-1)  # (Batch, N, N)
        
        # Softmax normalization: α_ij = softmax(e_ij)
        alpha = F.softmax(e, dim=2)  # (Batch, N, N)
        alpha = self.dropout(alpha)
        
        # Aggregation: h_i' = σ(Σ α_ij · W · h_j)
        h_prime = torch.bmm(alpha, Wh)  # (Batch, N, out_dim)
        
        return h_prime, alpha


class IDSSM(nn.Module):
    """
    ID-SSM: Interaction-aware Deep State-Space Model
    
    컨셉 문서 전체 구조:
    1. GAT Encoder: 센서 데이터 → 융합 특징 벡터 z
    2. MNN Decoder: 잠재 상태 x → 예측 특징 벡터 ẑ
    
    학습 목표: z ≈ ẑ (Reconstruction)
    """
    def __init__(self, num_sensors, latent_dim=8, hidden_dim=16):
        super(IDSSM, self).__init__()
        
        self.num_sensors = num_sensors
        self.latent_dim = latent_dim
        
        # ============================================================
        # Module 1: GAT-based Sensor Fusion (Section 2)
        # ============================================================
        self.gat = GATLayer(in_dim=1, out_dim=latent_dim)
        
        # Attention-based Readout (Section 2.3)
        # z_k = Σ β_i · h_i' where β_i = Softmax(MLP(h_i'))
        self.pooling = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.Tanh(),
            nn.Linear(latent_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # ============================================================
        # Module 2: Monotonic Neural Network (Section 3)
        # ============================================================
        # H: R^1 → R^d (상태 → 특징 벡터)
        # 모든 가중치가 양수로 제약되어 단조성 보장
        self.mnn = nn.Sequential(
            MonotonicLinear(1, hidden_dim),
            nn.Tanh(),  # 단조 증가 활성화 함수
            MonotonicLinear(hidden_dim, hidden_dim),
            nn.Tanh(),
            MonotonicLinear(hidden_dim, latent_dim)
        )
        
        # ============================================================
        # Trend direction weight (for monotonicity loss)
        # ============================================================
        # 열화 진행에 따른 z의 변화 방향을 학습
        self.w_trend = nn.Parameter(torch.ones(latent_dim))
        
    def encode(self, x_sensors):
        """
        GAT Encoder: 센서 데이터 → 융합 특징 벡터
        
        Args:
            x_sensors: (Batch, num_sensors) - 센서 측정값
        Returns:
            z: (Batch, latent_dim) - 융합 특징 벡터
            attn_weights: (Batch, N, N) - Attention weights
        """
        # 각 센서를 노드로 표현: h_i^(0) = y_k^i
        h = x_sensors.unsqueeze(-1)  # (Batch, N, 1)
        
        # GAT Layer 통과
        h_prime, attn_weights = self.gat(h)  # (Batch, N, latent_dim)
        
        # Attention-based Pooling (Readout)
        # β_i = Softmax(MLP(h_i'))
        beta = self.pooling(h_prime)  # (Batch, N, 1)
        
        # z = Σ β_i · h_i'
        z = torch.sum(h_prime * beta, dim=1)  # (Batch, latent_dim)
        
        return z, attn_weights
    
    def decode(self, state_x):
        """
        MNN Decoder: 잠재 상태 → 예측 특징 벡터
        
        Args:
            state_x: (Batch, 1) - 잠재 건강 상태 (0~1)
        Returns:
            z_pred: (Batch, latent_dim) - 예측 특징 벡터
        """
        return self.mnn(state_x)
    
    def forward(self, x_sensors, state_x):
        """
        Forward pass (학습용)
        
        Args:
            x_sensors: (Batch, num_sensors) - 센서 측정값
            state_x: (Batch, 1) - 정규화된 상태 (time/lifetime)
        Returns:
            z_enc: 인코딩된 특징 (GAT 출력)
            z_dec: 디코딩된 특징 (MNN 출력)
        """
        z_enc, _ = self.encode(x_sensors)
        z_dec = self.decode(state_x)
        return z_enc, z_dec
    
    def inverse_mapping(self, z_target, x_init=0.5, num_iters=100, lr=0.1):
        """
        MNN 역매핑: 관측된 z로부터 상태 x 추정
        
        컨셉 문서 (Section 4.2):
        x_new = argmin_x || z_new - H(x) ||²
        
        Args:
            z_target: (Batch, latent_dim) - 목표 특징 벡터
            x_init: 초기 상태값
            num_iters: 최적화 반복 횟수
            lr: 학습률
        Returns:
            x_estimated: (Batch,) - 추정된 상태
        """
        batch_size = z_target.shape[0]
        
        # 최적화 변수 초기화
        x = torch.full((batch_size, 1), x_init, 
                       dtype=z_target.dtype, device=z_target.device)
        x.requires_grad_(True)
        
        optimizer = torch.optim.Adam([x], lr=lr)
        
        for _ in range(num_iters):
            optimizer.zero_grad()
            z_pred = self.decode(x)
            loss = torch.sum((z_pred - z_target) ** 2, dim=1).mean()
            loss.backward()
            optimizer.step()
            
            # 상태 범위 제약: [0, 1]
            with torch.no_grad():
                x.clamp_(0.0, 1.0)
        
        return x.squeeze(-1).detach()
    
    def inverse_mapping_fast(self, z_target, x_candidates=None):
        """
        빠른 역매핑 (Grid Search 기반)
        
        Args:
            z_target: (Batch, latent_dim) - 목표 특징 벡터
            x_candidates: 후보 상태값 배열 (기본: 0~1을 100등분)
        Returns:
            x_estimated: (Batch,) - 추정된 상태
        """
        if x_candidates is None:
            x_candidates = torch.linspace(0.0, 1.0, 100, device=z_target.device)
        
        batch_size = z_target.shape[0]
        num_candidates = len(x_candidates)
        
        # 모든 후보에 대해 z_pred 계산
        x_expanded = x_candidates.view(-1, 1)  # (num_candidates, 1)
        with torch.no_grad():
            z_preds = self.decode(x_expanded)  # (num_candidates, latent_dim)
        
        # 각 배치 샘플에 대해 최소 거리 후보 찾기
        # z_target: (batch, latent_dim), z_preds: (candidates, latent_dim)
        z_target_expanded = z_target.unsqueeze(1)  # (batch, 1, latent_dim)
        z_preds_expanded = z_preds.unsqueeze(0)  # (1, candidates, latent_dim)
        
        distances = torch.sum((z_target_expanded - z_preds_expanded) ** 2, dim=2)  # (batch, candidates)
        best_indices = torch.argmin(distances, dim=1)  # (batch,)
        
        x_estimated = x_candidates[best_indices]
        return x_estimated


class IDSSMLoss(nn.Module):
    """
    ID-SSM 통합 Loss Function
    
    컨셉 문서 (Section 4.1):
    L_total = L_recon + λ₁·L_mono + λ₂·L_state
    
    1. L_recon: 측정 방정식 오차 (z_enc ≈ z_dec)
    2. L_mono: 단조성 정규화 (시간에 따른 z 방향 일관성)
    3. L_state: 상태 정규화 (선택적)
    """
    def __init__(self, lambda_mono=0.1, lambda_state=0.01):
        super(IDSSMLoss, self).__init__()
        self.lambda_mono = lambda_mono
        self.lambda_state = lambda_state
        self.mse = nn.MSELoss()
        
    def forward(self, z_enc, z_dec, z_enc_seq=None, w_trend=None, state_x=None):
        """
        Args:
            z_enc: (Batch, latent_dim) - GAT 인코딩 결과
            z_dec: (Batch, latent_dim) - MNN 디코딩 결과
            z_enc_seq: (Batch, Seq, latent_dim) - 시퀀스 데이터 (단조성 계산용, optional)
            w_trend: (latent_dim,) - 트렌드 방향 가중치 (optional)
            state_x: (Batch, 1) - 상태값 (상태 정규화용, optional)
        Returns:
            total_loss: 총 손실
            loss_dict: 개별 손실값 딕셔너리
        """
        # ============================================================
        # 1. Reconstruction Loss (L_recon)
        # ============================================================
        # || z_enc - z_dec ||²
        loss_recon = self.mse(z_enc, z_dec)
        
        # ============================================================
        # 2. Monotonicity Regularization (L_mono)
        # ============================================================
        loss_mono = torch.tensor(0.0, device=z_enc.device)
        
        if z_enc_seq is not None and w_trend is not None:
            # z_enc_seq: (Batch, Seq, latent_dim)
            # 시간 순서대로 z·w_trend가 증가해야 함
            
            # 각 시점의 trend projection
            z_projected = torch.matmul(z_enc_seq, w_trend)  # (Batch, Seq)
            
            # 연속 시점 간의 차이
            z_diff = z_projected[:, 1:] - z_projected[:, :-1]  # (Batch, Seq-1)
            
            # 감소하는 경우(역전)에 페널티
            # L_mono = Σ ReLU(-(z_k·w - z_{k-1}·w))
            loss_mono = torch.mean(F.relu(-z_diff))
        
        # ============================================================
        # 3. State Regularization (L_state) - Optional
        # ============================================================
        loss_state = torch.tensor(0.0, device=z_enc.device)
        
        if state_x is not None:
            # 상태값이 [0, 1] 범위를 벗어나는 것에 페널티
            # 학습 중에는 state_x가 time/lifetime이므로 이미 [0,1] 범위
            # 추가적인 정규화가 필요한 경우 여기에 구현
            pass
        
        # ============================================================
        # Total Loss
        # ============================================================
        total_loss = loss_recon + self.lambda_mono * loss_mono + self.lambda_state * loss_state
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': loss_recon.item(),
            'mono': loss_mono.item(),
            'state': loss_state.item()
        }
        
        return total_loss, loss_dict


class SequenceDataset(torch.utils.data.Dataset):
    """
    시퀀스 기반 데이터셋 (단조성 Loss 계산용)
    
    각 유닛의 시계열 데이터를 시퀀스로 구성
    """
    def __init__(self, train_df, features, seq_length=10):
        self.features = features
        self.seq_length = seq_length
        self.sequences = []
        self.states = []
        
        for unit_id in train_df['unit_nr'].unique():
            unit_data = train_df[train_df['unit_nr'] == unit_id]
            X = unit_data[features].values
            state_x = unit_data['state_x'].values
            
            # 슬라이딩 윈도우로 시퀀스 생성
            for i in range(len(X) - seq_length + 1):
                self.sequences.append(X[i:i+seq_length])
                self.states.append(state_x[i:i+seq_length])
        
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.states = torch.FloatTensor(np.array(self.states))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.states[idx]
