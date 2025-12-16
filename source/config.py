# config.py 수정

class Config:
    # ================================================================
    # PARTICLE FILTER PARAMETERS (Increased for stability)
    # ================================================================
    NUM_PARTICLES = 5000  # Increased from 1000
    
    # Dataset Config
    DATASET_PATH = 'data/train_FD001.txt'
    TEST_PATH = 'data/test_FD001.txt'
    RUL_PATH = 'data/RUL_FD001.txt'
    
    # Model Config
    FAILURE_THRESHOLD = 1.0
    DT = 1.0
    
    # Sensor Config (NASA C-MAPSS has 21 sensors)
    TOTAL_SENSORS = 21
    INDEX_COLS = ['unit_nr', 'time_cycles']
    SETTING_COLS = ['setting_1', 'setting_2', 'setting_3']
    SENSOR_COLS = ['s_{}'.format(i) for i in range(1, 22)]
    
    # ================================================================
    # OPTIMIZATION PARAMETERS (Increased for robustness)
    # ================================================================
    SMOOTHING_FRAC = 0.2  # Increased from 0.1 for more smoothing
    
    # ================================================================
    # PARTICLE FILTER STABILITY (New)
    # ================================================================
    INITIAL_COVARIANCE_INFLATION = 100.0  # For first update
    REGULAR_COVARIANCE_INFLATION = 10.0   # For subsequent updates
    MIN_EFFECTIVE_SAMPLE_SIZE_RATIO = 0.05  # 5% of NUM_PARTICLES