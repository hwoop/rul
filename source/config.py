"""
MSDFM Configuration - Aligned with Paper Parameters

Reference: "Remaining useful life prediction based on a multi-sensor data fusion model"
Li et al., Reliability Engineering and System Safety 208 (2021) 107249
"""

class Config:
    # ================================================================
    # PARTICLE FILTER PARAMETERS
    # ================================================================
    # 논문에서 명시적으로 언급하지 않았으나, 안정적인 추정을 위해 
    # 충분한 파티클 수 필요 (21개 센서 기준)
    NUM_PARTICLES = 3000  # 증가 (기존 1000)
    
    # ================================================================
    # MODEL CONFIGURATION
    # ================================================================
    # Section 3.1: "The failure threshold is correspondingly defined as D = 1"
    FAILURE_THRESHOLD = 1.0
    
    # Time interval (C-MAPSS는 cycle 단위)
    DT = 1.0
    
    # ================================================================
    # SENSOR CONFIGURATION (NASA C-MAPSS)
    # ================================================================
    TOTAL_SENSORS = 21
    INDEX_COLS = ['unit_nr', 'time_cycles']
    SETTING_COLS = ['setting_1', 'setting_2', 'setting_3']
    SENSOR_COLS = ['s_{}'.format(i) for i in range(1, 22)]
    
    # ================================================================
    # SMOOTHING PARAMETERS
    # ================================================================
    # Reference [28]: Cleveland & Devlin (1988) LOWESS
    # 논문에서 구체적 값 미언급, 일반적인 범위 사용
    SMOOTHING_FRAC = 0.15  # 적절한 스무딩 수준
    
    # ================================================================
    # PARTICLE FILTER STABILITY
    # ================================================================
    # 논문에서는 covariance inflation을 사용하지 않음
    # 수치 안정성을 위한 최소한의 정규화만 적용
    COVARIANCE_REGULARIZATION = 1e-6  # 최소 정규화 (기존 100.0에서 대폭 감소)
    
    # Effective Sample Size threshold for resampling
    # 일반적인 PF 구현에서 N/2 사용
    ESS_THRESHOLD_RATIO = 0.5  # N_eff < N * 0.5 일 때 resampling
    
    # ================================================================
    # PARAMETER ESTIMATION
    # ================================================================
    # Measurement function: φ(x) = x^c
    # c_p 탐색 범위 (Section 4.2 참고)
    C_PARAM_BOUNDS = (0.5, 4.0)  # 논문 Fig.5 참고: 대부분 1~4 범위
    
    # ================================================================
    # PSGS ALGORITHM (Section 3.3.2)
    # ================================================================
    # 센서 선택에 사용할 훈련 유닛 수
    # 전체 유닛 사용 시 계산 비용이 높으므로 subset 사용
    PSGS_SELECTION_UNITS = 30  # 기존 20에서 증가
