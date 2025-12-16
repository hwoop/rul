"""
ID-SSM Particle Filter for RUL Prediction

컨셉 문서 기반 구현:
- 상태 전이: x_k = x_{k-1} + η·Δt_k + ω_{k-1} (기존 MSDFM과 동일)
- 측정 업데이트: MNN 역매핑을 통한 상태 추정 통합

핵심 변경사항:
1. update(): GAT 인코딩 결과 z와 MNN 예측 z_pred 비교
2. estimate_state_from_observation(): MNN 역매핑 기반 상태 추정
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParticleFilterRUL:
    """
    ID-SSM용 파티클 필터
    
    컨셉 문서 (Section 4.2 RUL 예측):
    1. GAT를 통해 z_new 획득
    2. MNN 역매핑으로 x_new 추정: x_new = argmin_x || z_new - H(x) ||²
    3. 추정된 x_new를 Inverse Gaussian에 대입하여 RUL 분포 계산
    """
    
    def __init__(self, num_particles, drift_mean, drift_std, measurement_noise_std):
        """
        Args:
            num_particles: 파티클 수
            drift_mean: 열화율 평균 (μ_η)
            drift_std: 열화율 표준편차 (σ_η)
            measurement_noise_std: 측정 노이즈 표준편차
        """
        self.N = num_particles
        self.drift_mean = drift_mean
        self.drift_std = drift_std
        self.R_std = measurement_noise_std
        
        # 파티클: [state_x, degradation_rate_eta]
        self.particles = np.zeros((self.N, 2))
        self.weights = np.ones(self.N) / self.N
        
        # 재현성을 위한 RNG
        self.rng = np.random.default_rng(2024)
        
        # MNN 역매핑용 상태 후보 (미리 계산)
        self.x_candidates = np.linspace(0.0, 1.0, 200)
        
    def initialize(self):
        """
        파티클 초기화
        
        - state_x: 0 근처에서 시작 (건강한 상태)
        - eta: N(μ_η, σ²_η)에서 샘플링
        """
        # 상태 초기화: 약간의 분산을 가진 0 근처
        self.particles[:, 0] = np.random.uniform(0, 0.05, self.N)
        
        # 열화율 초기화: 사전 분포에서 샘플링
        self.particles[:, 1] = np.random.normal(
            self.drift_mean, 
            self.drift_std, 
            self.N
        )
        # 음수 열화율 방지
        self.particles[:, 1] = np.maximum(self.particles[:, 1], 1e-6)
        
        # 가중치 균등 초기화
        self.weights = np.ones(self.N) / self.N
        
    def predict(self, dt=1.0, sigma_B=0.001):
        """
        상태 전이 예측 단계
        
        x_k = x_{k-1} + η_{k-1}·Δt + ω_{k-1}
        ω ~ N(0, σ²_B·Δt)
        
        Args:
            dt: 시간 간격
            sigma_B: Brownian motion 표준편차
        """
        eta = self.particles[:, 1]
        x_prev = self.particles[:, 0]
        
        # 상태 전이 노이즈
        process_noise = np.random.normal(0, sigma_B * np.sqrt(dt), self.N)
        
        # 열화율에도 약간의 변동 추가 (적응성)
        eta_noise = np.random.normal(0, 1e-5, self.N)
        
        # 상태 업데이트
        x_new = x_prev + eta * dt + process_noise
        self.particles[:, 0] = np.clip(x_new, 0.0, 1.5)  # 약간의 오버슈트 허용
        
        # 열화율 업데이트
        self.particles[:, 1] = np.maximum(eta + eta_noise, 1e-6)
        
    def update_with_mnn(self, z_obs, model, use_inverse_mapping=True):
        """
        MNN 기반 측정 업데이트 (컨셉 문서 일치)
        
        컨셉 문서 (Section 4.2):
        "MNN의 역함수 또는 수치해석적 방법을 통해 z_new에 대응하는 x_new를 찾습니다"
        
        Args:
            z_obs: (latent_dim,) - GAT로 인코딩된 관측 특징 벡터
            model: IDSSM 모델 (MNN 디코더 포함)
            use_inverse_mapping: True면 역매핑 사용, False면 기존 방식
        """
        if use_inverse_mapping:
            self._update_with_inverse_mapping(z_obs, model)
        else:
            self._update_with_likelihood(z_obs, model)
    
    def _update_with_inverse_mapping(self, z_obs, model):
        """
        MNN 역매핑 기반 업데이트
        
        1. z_obs로부터 가장 likely한 상태 x_obs 추정
        2. 각 파티클의 상태와 x_obs 사이의 거리로 가중치 업데이트
        """
        z_obs_tensor = torch.FloatTensor(z_obs).unsqueeze(0)
        
        with torch.no_grad():
            # MNN 역매핑으로 관측된 상태 추정
            x_obs_estimated = model.inverse_mapping_fast(z_obs_tensor).item()
        
        # 각 파티클 상태와 추정된 관측 상태 사이의 차이로 가중치 계산
        x_particles = self.particles[:, 0]
        
        # Gaussian likelihood: p(x_obs | x_particle)
        state_diff = x_particles - x_obs_estimated
        sigma_state = self.R_std * 0.1  # 상태 공간에서의 불확실성
        
        log_likelihood = -0.5 * (state_diff ** 2) / (sigma_state ** 2)
        log_likelihood -= log_likelihood.max()  # 수치 안정성
        
        likelihood = np.exp(log_likelihood)
        
        # 가중치 업데이트
        self.weights = self.weights * likelihood
        weight_sum = np.sum(self.weights)
        
        if weight_sum > 1e-300:
            self.weights /= weight_sum
        else:
            # 가중치 붕괴 시 균등 분포로 리셋
            self.weights = np.ones(self.N) / self.N
            
    def _update_with_likelihood(self, z_obs, model):
        """
        기존 방식: 파티클 상태에서 z_pred 계산 후 z_obs와 비교
        
        p(z_obs | x_particle) ∝ exp(-||z_obs - H(x_particle)||² / 2σ²)
        """
        with torch.no_grad():
            x_tensor = torch.FloatTensor(self.particles[:, 0]).unsqueeze(1)
            z_pred = model.decode(x_tensor).numpy()
        
        # 잔차 계산
        diff = z_pred - z_obs[None, :]
        dist2 = np.sum(diff ** 2, axis=1)
        
        # Log-likelihood
        sigma2 = self.R_std ** 2
        log_likelihood = -0.5 * dist2 / sigma2
        log_likelihood -= log_likelihood.max()
        
        likelihood = np.exp(log_likelihood)
        
        # 가중치 업데이트
        self.weights = self.weights * likelihood
        weight_sum = np.sum(self.weights)
        
        if weight_sum > 1e-300:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.N) / self.N
    
    def update(self, z_obs, model):
        """
        측정 업데이트 (후방 호환성 유지)
        
        기본적으로 역매핑 방식 사용
        """
        self.update_with_mnn(z_obs, model, use_inverse_mapping=True)
    
    def neff(self):
        """
        유효 샘플 크기 (Effective Sample Size)
        
        N_eff = 1 / Σ(w_i²)
        """
        return 1.0 / np.sum(np.square(self.weights))
    
    def resample(self):
        """
        Systematic Resampling
        """
        indices = self._systematic_resample(self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N
        
    def _systematic_resample(self, weights):
        """
        Systematic resampling 알고리즘
        """
        N = len(weights)
        positions = (self.rng.random() + np.arange(N)) / N
        
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # 수치 오차 방지
        
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                if j >= N:
                    j = N - 1
        
        return indexes
    
    def fuzzy_resample(self):
        """
        Fuzzy Resampling (논문 [27] 참조)
        
        중복된 파티클에 노이즈를 추가하여 다양성 유지
        """
        N_eff = self.neff()
        
        if N_eff >= self.N * 0.5:
            return  # 리샘플링 불필요
        
        # Systematic resampling
        indices = self._systematic_resample(self.weights)
        new_particles = self.particles[indices].copy()
        
        # Fuzzing: 열화율에 노이즈 추가
        var_eta = np.var(self.particles[:, 1])
        sigma_fuzz = np.sqrt(var_eta / self.N) if var_eta > 0 else 1e-8
        
        # 중복된 파티클 찾기
        unique_idx, counts = np.unique(indices, return_counts=True)
        
        for idx, count in zip(unique_idx, counts):
            if count > 1:
                mask = (indices == idx)
                noise = np.random.normal(0, sigma_fuzz, count)
                new_particles[mask, 1] += noise
        
        # 양수 열화율 보장
        new_particles[:, 1] = np.maximum(new_particles[:, 1], 1e-6)
        
        self.particles = new_particles
        self.weights = np.ones(self.N) / self.N
    
    def estimate_state(self):
        """
        현재 상태 추정 (가중 평균)
        
        Returns:
            x_hat: 추정 상태
            eta_hat: 추정 열화율
        """
        x_hat = np.sum(self.weights * self.particles[:, 0])
        eta_hat = np.sum(self.weights * self.particles[:, 1])
        return x_hat, eta_hat
    
    def estimate_state_median(self):
        """
        현재 상태 추정 (중앙값 - 더 robust)
        
        Returns:
            x_hat: 추정 상태
            eta_hat: 추정 열화율
        """
        # 가중 중앙값 근사: 리샘플링 후 중앙값
        indices = self._systematic_resample(self.weights)
        resampled = self.particles[indices]
        
        x_hat = np.median(resampled[:, 0])
        eta_hat = np.median(resampled[:, 1])
        return x_hat, eta_hat
    
    def estimate_rul(self, current_time=None):
        """
        RUL 추정
        
        컨셉 문서 (Section 4.2):
        "구해진 x_new를 기존 논문의 식 (21) (Inverse Gaussian Distribution)에 
         대입하여 RUL의 확률 분포를 계산합니다."
        
        Mean RUL = (D - x̂) / η̂  (D = 1.0, 고장 임계값)
        
        Args:
            current_time: 현재 시간 (사용하지 않지만 호환성 유지)
        Returns:
            pred_rul: 예측 RUL
        """
        D = 1.0  # 고장 임계값
        
        # 중앙값 기반 추정 (robust)
        x_hat, eta_hat = self.estimate_state_median()
        
        # 경계 조건 처리
        if x_hat >= D:
            return 0.0
        
        if eta_hat <= 1e-9:
            return 1000.0  # 매우 느린 열화
        
        # Mean RUL (First Passage Time)
        mean_rul = (D - x_hat) / eta_hat
        
        return max(0.0, mean_rul)
    
    def estimate_rul_distribution(self):
        """
        RUL 분포 추정 (평균 및 표준편차)
        
        Returns:
            mean_rul: RUL 평균
            std_rul: RUL 표준편차
        """
        D = 1.0
        
        # 각 파티클의 RUL 계산
        x = np.clip(self.particles[:, 0], 0.0, D - 1e-6)
        eta = np.maximum(self.particles[:, 1], 1e-9)
        
        rul_particles = (D - x) / eta
        rul_particles = np.maximum(rul_particles, 0.0)
        
        # 가중 평균 및 표준편차
        mean_rul = np.sum(self.weights * rul_particles)
        var_rul = np.sum(self.weights * (rul_particles - mean_rul) ** 2)
        std_rul = np.sqrt(var_rul)
        
        return mean_rul, std_rul
    
    def get_diagnostics(self):
        """
        진단 정보 반환 (디버깅용)
        """
        x_hat, eta_hat = self.estimate_state()
        
        return {
            'N_eff': self.neff(),
            'N_eff_ratio': self.neff() / self.N,
            'x_mean': x_hat,
            'x_std': np.sqrt(np.sum(self.weights * (self.particles[:, 0] - x_hat) ** 2)),
            'x_median': np.median(self.particles[:, 0]),
            'eta_mean': eta_hat,
            'eta_std': np.sqrt(np.sum(self.weights * (self.particles[:, 1] - eta_hat) ** 2)),
            'eta_median': np.median(self.particles[:, 1]),
            'weight_max': np.max(self.weights),
            'weight_entropy': -np.sum(self.weights * np.log(self.weights + 1e-300)),
        }
