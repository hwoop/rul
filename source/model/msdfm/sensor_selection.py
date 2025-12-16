"""
PSGS Algorithm for Informative Sensor Selection

Reference: "Remaining useful life prediction based on a multi-sensor data fusion model"
Li et al., Reliability Engineering and System Safety 208 (2021) 107249

Key Implementation Details:
- Section 3.3.2: PSGS (Prioritized Sensor Group Selection) algorithm
- Figure 3: Flowchart of the PSGS algorithm
- Eq. 22: WARE (Weighted Absolute Relative Error) metric
- Eq. 23: ARE (Absolute Relative Error) calculation
"""

import numpy as np
import pandas as pd
from .particle_filter import ParticleFilter
from config import Config


def calculate_are(lifetime, pred_rul, current_time):
    """
    Calculate Absolute Relative Error - Eq. 23
    
    Er_{n,k} = |t_{n,K_n} - l̂_{n,k} - t_{n,k}| / t_{n,K_n} × 100%
    
    where:
    - t_{n,K_n}: actual lifetime of unit n
    - l̂_{n,k}: predicted RUL at time t_{n,k}
    - t_{n,k}: current time
    
    Args:
        lifetime: t_{n,K_n} - total lifetime
        pred_rul: l̂_{n,k} - predicted RUL
        current_time: t_{n,k} - current time
        
    Returns:
        ARE in percentage
    """
    true_rul = lifetime - current_time  # Actual remaining life
    error = abs(true_rul - pred_rul)
    are = (error / lifetime) * 100.0  # Note: divided by LIFETIME, not true_rul
    return are


def calculate_ware(lifetime, pred_ruls, time_indices):
    """
    Calculate Weighted Absolute Relative Error for a single unit - Eq. 22
    
    WARE_n = (1/K_n) × Σ_k (t_{n,k} × Er_{n,k} / t_{n,K_n})
    
    The time weighting (t_{n,k}/t_{n,K_n}) gives more importance to 
    predictions made later in the unit's life.
    
    Args:
        lifetime: t_{n,K_n} - total lifetime of the unit
        pred_ruls: array of predicted RULs at each time step
        time_indices: array of time indices (t_{n,k})
        
    Returns:
        WARE score for this unit
    """
    K_n = len(pred_ruls)
    if K_n == 0:
        return 0.0
    
    weighted_sum = 0.0
    for k in range(K_n):
        t_nk = time_indices[k]
        pred_rul = pred_ruls[k]
        
        # Calculate ARE (Eq. 23)
        are = calculate_are(lifetime, pred_rul, t_nk)
        
        # Apply time weighting: t_{n,k} / t_{n,K_n}
        weight = t_nk / lifetime
        
        weighted_sum += weight * are
    
    # Average over all time steps
    ware = weighted_sum / K_n
    return ware


def run_prediction_for_unit(unit_df, sensors, params):
    """
    Run RUL prediction for a single unit using specified sensors.
    
    Implements the algorithm in Table 1.
    
    Args:
        unit_df: DataFrame containing sensor data for one unit
        sensors: List of sensor names to use
        params: MSDFM_Parameters object
        
    Returns:
        Array of predicted RULs at each time step
    """
    measurements = unit_df[sensors].values
    
    if len(measurements) == 0:
        return np.array([])
    
    # Initialize PF with first measurement
    pf = ParticleFilter(params, sensors, initial_data=measurements[0])
    
    # First RUL estimate
    ruls = [pf.estimate_rul()]
    
    # Process remaining measurements
    for t in range(1, len(measurements)):
        meas = measurements[t]
        pf.predict()
        pf.update(meas)
        pf.fuzzy_resampling()
        ruls.append(pf.estimate_rul())
    
    return np.array(ruls)


def evaluate_sensor_group(train_df, lifetimes, sensors, params):
    """
    Evaluate a sensor group using WARE metric.
    
    Args:
        train_df: Training DataFrame
        lifetimes: Array of unit lifetimes
        sensors: List of sensors to evaluate
        params: MSDFM_Parameters object
        
    Returns:
        Average WARE score across evaluation units
    """
    units = train_df['unit_nr'].unique()
    
    # Use subset of units for efficiency (as mentioned in paper)
    n_eval_units = min(Config.PSGS_SELECTION_UNITS, len(units))
    eval_units = units[:n_eval_units]
    
    # Build subset params for this sensor group
    try:
        full_sensors = params.sensor_list
        indices = [full_sensors.index(s) for s in sensors]
        subset_cov = params.Cov_matrix[np.ix_(indices, indices)]
    except (ValueError, IndexError) as e:
        print(f"  Warning: Could not subset covariance matrix: {e}")
        subset_cov = np.eye(len(sensors)) * 0.01
    
    # Create subset parameters object
    class SubsetParams:
        pass
    
    subset_params = SubsetParams()
    subset_params.mu_eta = params.mu_eta
    subset_params.sigma_eta = params.sigma_eta
    subset_params.sigma_B = params.sigma_B
    subset_params.sensor_params = params.sensor_params
    subset_params.Cov_matrix = subset_cov
    subset_params.sensor_list = sensors
    
    # Calculate WARE for each unit
    ware_scores = []
    
    for unit in eval_units:
        unit_df = train_df[train_df['unit_nr'] == unit]
        unit_idx = unit - 1  # Assuming 1-indexed units
        
        if unit_idx >= len(lifetimes):
            continue
            
        lifetime = lifetimes[unit_idx]
        time_indices = unit_df['time_cycles'].values
        
        # Get predictions
        pred_ruls = run_prediction_for_unit(unit_df, sensors, subset_params)
        
        if len(pred_ruls) == 0:
            continue
        
        # Calculate WARE for this unit
        ware = calculate_ware(lifetime, pred_ruls, time_indices)
        ware_scores.append(ware)
    
    if len(ware_scores) == 0:
        return float('inf')
    
    return np.mean(ware_scores)


def psgs_algorithm(train_df, lifetimes, params):
    """
    Prioritized Sensor Group Selection (PSGS) Algorithm - Section 3.3.2, Figure 3
    
    Process:
    1. Evaluate each sensor individually using WARE metric
    2. Sort sensors in ascending order of WARE (lower = better)
    3. Build sensor groups by adding one sensor at a time following priority order
    4. Select the group with lowest WARE
    
    Args:
        train_df: Training DataFrame with sensor measurements
        lifetimes: Array of unit lifetimes
        params: MSDFM_Parameters object with estimated parameters
        
    Returns:
        best_group: List of selected sensor names
        ranked_sensors: Sensors sorted by individual performance
        group_scores: WARE scores for each sensor group
        individual_scores: Dict of individual sensor WARE scores
    """
    all_sensors = params.sensor_list
    
    print("="*60)
    print("PSGS Algorithm - Sensor Selection")
    print("="*60)
    
    # ====================================================================
    # STEP 1: Individual Sensor Evaluation (Figure 3, left branch)
    # ====================================================================
    print("\nStep 1: Evaluating individual sensors...")
    
    individual_scores = {}
    
    for i, sensor in enumerate(all_sensors):
        ware = evaluate_sensor_group(train_df, lifetimes, [sensor], params)
        individual_scores[sensor] = ware
        
        if (i + 1) % 5 == 0 or (i + 1) == len(all_sensors):
            print(f"  Progress: {i+1}/{len(all_sensors)} sensors evaluated")
    
    # ====================================================================
    # STEP 2: Sort sensors by WARE (ascending - lower is better)
    # ====================================================================
    ranked_sensors = sorted(individual_scores.keys(), key=lambda s: individual_scores[s])
    
    print(f"\nSensor Ranking (by individual WARE):")
    for i, sensor in enumerate(ranked_sensors[:5]):
        print(f"  {i+1}. {sensor}: WARE = {individual_scores[sensor]:.4f}")
    if len(ranked_sensors) > 5:
        print(f"  ... and {len(ranked_sensors) - 5} more sensors")
    
    # ====================================================================
    # STEP 3: Evaluate sensor groups (Figure 3, right branch)
    # ====================================================================
    print("\nStep 2: Evaluating sensor groups...")
    
    best_ware = float('inf')
    best_group = []
    group_scores = []
    current_group = []
    
    for i, sensor in enumerate(ranked_sensors):
        current_group.append(sensor)
        
        # Evaluate current group
        ware = evaluate_sensor_group(train_df, lifetimes, current_group, params)
        group_scores.append(ware)
        
        print(f"  Group size {len(current_group)}: WARE = {ware:.4f}")
        
        # Track best group
        if ware < best_ware:
            best_ware = ware
            best_group = list(current_group)
    
    # ====================================================================
    # STEP 4: Return optimal sensor group
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"Optimal Sensor Group: {len(best_group)} sensors")
    print(f"Selected sensors: {best_group}")
    print(f"Best WARE: {best_ware:.4f}")
    print(f"{'='*60}")
    
    return best_group, ranked_sensors, group_scores, individual_scores
