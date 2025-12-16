# sensor_selection.py
import numpy as np
import pandas as pd
from particle_filter import ParticleFilter
from config import Config

def calculate_ware(y_true, y_pred, time_indices):
    rul_true = y_true - time_indices
    mask = rul_true > 0
    if np.sum(mask) == 0: return 0.0
    are = np.abs(rul_true[mask] - y_pred[mask]) / rul_true[mask]
    return np.mean(are)

def run_prediction_for_unit(unit_df, sensors, params):
    pf = ParticleFilter(params, sensors)
    measurements = unit_df[sensors].values
    if len(measurements) == 0:
        return np.array([])
    
    pf = ParticleFilter(params, sensors, initial_data=measurements[0])
    ruls = [pf.estimate_rul()]
    
    for t in range(1, len(measurements)):
        meas = measurements[t]
        pf.predict()
        pf.update(meas)
        pf.fuzzy_resampling()
        ruls.append(pf.estimate_rul())
    
    return np.array(ruls)

def psgs_algorithm(train_df, lifetimes, params):
    # === [FIX]: Iterate over sensors present in the model params ===
    all_sensors = params.sensor_list
    units = train_df['unit_nr'].unique()
    selection_units = units[:20] 
    
    # 1. Individual Performance
    sensor_scores = {}
    print("PSGS Step 1: Evaluating individual sensors...")
    
    for sensor in all_sensors:
        temp_params = params 
        unit_wares = []
        for u in selection_units:
            u_df = train_df[train_df['unit_nr'] == u]
            L_n = lifetimes[u-1] 
            preds = run_prediction_for_unit(u_df, [sensor], temp_params)
            ware = calculate_ware(L_n, preds, u_df['time_cycles'].values)
            unit_wares.append(ware)
        sensor_scores[sensor] = np.mean(unit_wares)
        
    ranked_sensors = sorted(sensor_scores, key=sensor_scores.get)
    print(f"Ranked Sensors: {ranked_sensors}")
    
    # 2. Group Selection
    print("PSGS Step 2: Evaluating sensor groups...")
    best_ware = float('inf')
    best_group = []
    group_scores = []
    current_group = []
    
    for sensor in ranked_sensors:
        current_group.append(sensor)
        
        # === [FIX]: Slice using params.sensor_list ===
        full_sensors = params.sensor_list
        indices = [full_sensors.index(s) for s in current_group]
        subset_cov = params.Cov_matrix[np.ix_(indices, indices)]
        
        group_params = type('obj', (object,), {
            'mu_eta': params.mu_eta, 'sigma_eta': params.sigma_eta, 'sigma_B': params.sigma_B,
            'sensor_params': params.sensor_params,
            'Cov_matrix': subset_cov,
            'sensor_list': current_group # For the subset params
        })
        
        unit_wares = []
        for u in selection_units:
            u_df = train_df[train_df['unit_nr'] == u]
            L_n = lifetimes[u-1]
            preds = run_prediction_for_unit(u_df, current_group, group_params)
            ware = calculate_ware(L_n, preds, u_df['time_cycles'].values)
            unit_wares.append(ware)
            
        avg_ware = np.mean(unit_wares)
        group_scores.append(avg_ware)
        print(f"Group size {len(current_group)}: WARE = {avg_ware:.4f}")
        
        if avg_ware < best_ware:
            best_ware = avg_ware
            best_group = list(current_group)
            
    return best_group, ranked_sensors, group_scores, sensor_scores