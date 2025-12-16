import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import random
import os
import datetime
import time

import utils
import visualize
from model import idssm, msdfm

# ---------------------------------------------------------
# 0. 시드 설정 및 데이터 로드 함수
# ---------------------------------------------------------
def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_process_data(train_path, test_path, rul_path):
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    train = pd.read_csv(train_path, sep='\s+', header=None, names=col_names)
    test = pd.read_csv(test_path, sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv(rul_path, sep='\s+', header=None, names=['RUL'])

    # Calculate Max Cycles (Lifetimes) for Training Data
    train_max_cycle = train.groupby('unit_nr')['time_cycles'].max().reset_index()
    train_max_cycle.columns = ['unit_nr', 'max_cycle']
    train = train.merge(train_max_cycle, on='unit_nr', how='left')
    
    # Normalized State (0~1) for IDSSM
    train['state_x'] = train['time_cycles'] / train['max_cycle']

    # Drop constant/uninformative sensors (based on typical C-MAPSS analysis)
    # drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_sensors = []
    features = [c for c in train.columns if c.startswith('s_') and c not in drop_sensors]

    # MinMax Scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])

    # Drift Statistics for Particle Filters
    # MSDFM needs raw lifetimes, IDSSM needs inverse rate stats
    lifetimes = train_max_cycle['max_cycle'].values
    drift_stats = (1.0 / train_max_cycle['max_cycle']).agg(['mean', 'std']).to_dict()
    
    return train, test, y_test, features, lifetimes, drift_stats

# ---------------------------------------------------------
# 1. MSDFM Pipeline (Baseline)
# ---------------------------------------------------------
def run_msdfm_step(train_df, test_df, y_test, features, lifetimes, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("\n" + "="*50)
    print("Running MSDFM Pipeline")
    print("="*50)
    
    # 1.1 Parameter Estimation
    print("[MSDFM] Estimating Model Parameters...")
    params = msdfm.MSDFM_Parameters() 
    params.estimate_state_params(lifetimes)
    params.estimate_measurement_params(train_df, features)
    
    # 1.2 Sensor Selection (PSGS)
    print("[MSDFM] Running PSGS for Sensor Selection...")
    best_sensors, ranked, group_scores, individual_scores = msdfm.psgs_algorithm(train_df, lifetimes, params)
    print(f"[MSDFM] Optimal Sensor Group ({len(best_sensors)}): {best_sensors}")
    
    # PSGS 결과 시각화
    visualize.plot_psgs_ware(
        ranked_sensors=ranked,
        group_scores=group_scores,
        sensor_scores=individual_scores,
        title_suffix="FD001",  # 데이터셋 이름 등을 접미사로 추가
        save_dir=save_dir,
        show_plot=False
    )
    
    # Update params to use only best sensors (subset covariance)
    indices = [params.sensor_list.index(s) for s in best_sensors]
    params.sensor_list = best_sensors
    params.Cov_matrix = params.Cov_matrix[np.ix_(indices, indices)]
    
    # 1.3 Test Evaluation
    print("[MSDFM] Evaluating on Test Set...")
    pred_ruls = []
    true_ruls = y_test['RUL'].values
    test_units = sorted(test_df['unit_nr'].unique())
    all_results = []
    
    start_time = time.time()
    
    for i, unit_id in enumerate(test_units):
        unit_data = test_df[test_df['unit_nr'] == unit_id]
        X_seq = unit_data[best_sensors].values # Use best sensors
        time_cycles = unit_data['time_cycles'].values
        
        if len(X_seq) == 0: continue
            
        final_true_rul = true_ruls[i]
        total_life = time_cycles[-1] + final_true_rul
        
        # Initialize PF with the first measurement
        pf = msdfm.particle_filter.ParticleFilter(params, best_sensors, initial_data=X_seq[0])
        pred_rul_t = pf.estimate_rul() # Initial estimate
        
        # Time Step Iteration
        for t in range(1, len(X_seq)):
            meas = X_seq[t]
            pf.predict()
            pf.update(meas)
            pf.fuzzy_resampling()
            
            pred_rul_t = pf.estimate_rul()
            
            # Record result for metrics
            current_age = time_cycles[t]
            are = np.abs(total_life - pred_rul_t - current_age) / total_life * 100.0
            
            all_results.append({
                'unit_nr': unit_id,
                'life_percent': (current_age / total_life) * 100,
                'ARE': are,
                't_nk': current_age,
                't_nKn': total_life,
                'Model': 'MSDFM'
            })
            
        pred_ruls.append(pred_rul_t)
        if (i+1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_units)} units...")
            
    elapsed = time.time() - start_time
    rmse = np.sqrt(mean_squared_error(true_ruls, pred_ruls))
    
    # Calculate WARE
    results_df = pd.DataFrame(all_results)
    ware_list = []
    for unit_id, df_u in results_df.groupby('unit_nr'):
        t_nKn = df_u['t_nKn'].iloc[0]
        term = (df_u['t_nk'] * df_u['ARE'] / t_nKn).mean()
        ware_list.append(term)
    overall_ware = np.mean(ware_list)
    
    print(f"[MSDFM] Done in {elapsed:.2f}s | RMSE: {rmse:.4f} | WARE: {overall_ware:.2f}%")
    
    # Save results
    results_df.to_csv(os.path.join(save_dir, 'msdfm_results.csv'), index=False)
    utils.save_percentile_stats(results_df, save_dir)
    utils.plot_lifetime_performance(results_df, save_dir) 
    utils.plot_rul_comparison(true_ruls, pred_ruls, rmse, save_dir)
    
    return rmse, overall_ware, results_df

# ---------------------------------------------------------
# 2. IDSSM Pipeline (Proposed)
# ---------------------------------------------------------
def run_idssm_step(train_df, test_df, y_test, features, drift_stats, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("\n" + "="*50)
    print("Running IDSSM Pipeline")
    print("="*50)
    
    # Data Prep for PyTorch
    X_train = torch.FloatTensor(train_df[features].values)
    state_train = torch.FloatTensor(train_df['state_x'].values).unsqueeze(-1)
    dataset = torch.utils.data.TensorDataset(X_train, state_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # 2.1 Model Training
    model = idssm.models.IDSSM(num_sensors=len(features), latent_dim=16)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    loss_history = []

    print("[IDSSM] Training GAT-MNN Network...")
    model.train()
    for epoch in range(50): 
        epoch_loss = 0
        for bx, by in dataloader:
            optimizer.zero_grad()
            z_enc, z_dec = model(bx, by)
            loss = criterion(z_enc, z_dec)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    # 2.2 Noise Estimation
    model.eval()
    with torch.no_grad():
        z_enc_full, z_dec_full = model(X_train, state_train)
        residuals = z_enc_full - z_dec_full
    err_norm = torch.norm(residuals, dim=1)
    meas_noise_std = err_norm.median().item()
    meas_noise_std_robust = max(meas_noise_std, 1e-3) * 5.0
    print(f"[IDSSM] Estimated Noise Std: {meas_noise_std_robust:.6f}")

    # 2.3 Test Evaluation
    print("[IDSSM] Conducting Lifetime Performance Analysis...")
    all_results = [] 
    pred_ruls = []   
    true_ruls = y_test['RUL'].values 
    test_units = sorted(test_df['unit_nr'].unique())
    final_unit_predictions = []
    
    # For GAT Attention Visualization
    last_attention_weights = None
    
    start_time = time.time()

    for i, unit_id in enumerate(test_units):
        unit_data = test_df[test_df['unit_nr'] == unit_id]
        X_seq = unit_data[features].values
        time_cycles = unit_data['time_cycles'].values
        
        final_true_rul = true_ruls[i]
        total_life = time_cycles[-1] + final_true_rul
        
        pf = idssm.ParticleFilterRUL(5000, drift_stats['mean'], drift_stats['std'], meas_noise_std_robust)
        pf.initialize()
        
        for t in range(len(X_seq)):
            pf.predict()

            x_obs = torch.FloatTensor(X_seq[t]).unsqueeze(0)
            with torch.no_grad():
                # model.encode returns (z, attention_weights)
                z_out, attn = model.encode(x_obs)
                z_obs = z_out.numpy().flatten()
                
                # Capture attention from the last step of the last unit (or any representative step)
                if i == len(test_units) - 1 and t == len(X_seq) - 1:
                    last_attention_weights = attn

            pf.update(z_obs, model)
            if pf.neff() < pf.N / 2: pf.resample()
            
            pred_rul_t = pf.estimate_rul(time_cycles[t])
            pred_rul_t = min(pred_rul_t, 145.0) # Cap for stability

            current_age = time_cycles[t]
            are = np.abs(total_life - pred_rul_t - current_age) / total_life * 100.0
                
            all_results.append({
                'unit_nr': unit_id,
                'life_percent': (current_age / total_life) * 100,
                'ARE': are,
                't_nk': current_age,
                't_nKn': total_life,
                'Model': 'IDSSM'
            })
            
            if t == len(X_seq) - 1:
                pred_ruls.append(pred_rul_t)
                final_unit_predictions.append((unit_id, final_true_rul, pred_rul_t))
                
            if unit_id % 10 == 0 and (t == 0 or t == len(X_seq) - 1):
                visualize.plot_gat_attention_heatmap(
                    attention_weights=attn,
                    sensor_names=features, 
                    save_dir=f'{save_dir}/gat_attention_heatmap', 
                    filename=f'unit{unit_id}_timestep{t+1}',
                    show_plot=False
                )

        if (i+1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_units)} units...")
            
    elapsed = time.time() - start_time

    # Results Aggregation
    results_df = pd.DataFrame(all_results)
    ware_list = []
    for unit_id, df_u in results_df.groupby('unit_nr'):
        t_nKn = df_u['t_nKn'].iloc[0]
        term = (df_u['t_nk'] * df_u['ARE'] / t_nKn).mean()
        ware_list.append(term)

    overall_ware = np.mean(ware_list)
    rmse = np.sqrt(mean_squared_error(true_ruls, pred_ruls))

    print(f"[IDSSM] Done in {elapsed:.2f}s | RMSE: {rmse:.4f} | WARE: {overall_ware:.2f}%")

    # Saving artifacts
    results_df.to_csv(os.path.join(save_dir, 'idssm_results.csv'), index=False)
    utils.plot_training_loss(loss_history, save_dir)
    utils.save_unit_predictions(final_unit_predictions, rmse, overall_ware, save_dir)
    utils.save_percentile_stats(results_df, save_dir)
    utils.plot_lifetime_performance(results_df, save_dir) 
    utils.plot_rul_comparison(true_ruls, pred_ruls, rmse, save_dir)
    
    # --- Visualization Integration: GAT Attention Heatmap ---
    if last_attention_weights is not None:
        print(f"[IDSSM] Saving GAT Attention Heatmap...")
        visualize.plot_gat_attention_heatmap(
            last_attention_weights, 
            sensor_names=features, 
            save_dir=save_dir, 
            show_plot=False
        )
    
    return rmse, overall_ware, results_df

# ---------------------------------------------------------
# 3. Main Execution
# ---------------------------------------------------------
def run_full_experiment(train_path, test_path, rul_path, save_dir):
            
    set_seed(2024)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print(f"Experiment Results will be saved to: {save_dir}")

    # 1. Load Data
    print("Loading and Processing Data...")
    train, test, y_test, feats, lifetimes, drift = load_and_process_data(train_path, test_path, rul_path)
    
    rmse_b, ware_b, res_idssm = run_idssm_step(train, test, y_test, feats, drift, f'{save_dir}/idssm')
    rmse_a, ware_a, res_msdfm = run_msdfm_step(train, test, y_test, feats, lifetimes, f'{save_dir}/msdfm')

    # 4. Final Comparison & Visualization
    print("\n" + "="*50)
    print(" FINAL COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Model':<10} | {'RMSE':<10} | {'WARE (%)':<10}")
    print("-" * 36)
    print(f"{'MSDFM':<10} | {rmse_a:<10.4f} | {ware_a:<10.2f}")
    print(f"{'IDSSM':<10} | {rmse_b:<10.4f} | {ware_b:<10.2f}")
    print("="*50)
    
    # Save comparison text
    with open(os.path.join(save_dir, 'final_comparison.txt'), 'w') as f:
        f.write("FINAL COMPARISON SUMMARY\n")
        f.write(f"Model  | RMSE       | WARE (%)\n")
        f.write(f"MSDFM  | {rmse_a:.4f}     | {ware_a:.2f}\n")
        f.write(f"IDSSM  | {rmse_b:.4f}     | {ware_b:.2f}\n")

    # --- Visualization Integration: Comparative Plots ---
    if res_msdfm is not None and res_idssm is not None:
        print("\n[Visualization] Generating Comparative Plots (MSDFM vs IDSSM)...")
        try:
            # Combined ARE Plot (Mean & Variance)
            # visualize.plot_combined_are_comparison(
            #     msdfm_results_df=res_msdfm,
            #     idssm_results_df=res_idssm,
            #     save_dir=save_dir,
            #     show_plot=False
            # )
            
            # Individual Plots (optional, but good for detailed view)
            visualize.plot_mean_are_comparison(
                results_dfs=[res_msdfm, res_idssm],
                labels=["MSDFM", "IDSSM"],
                save_dir=f'{save_dir}/all',
                show_plot=False
            )
            visualize.plot_variance_are_comparison(
                results_dfs=[res_msdfm, res_idssm],
                labels=["MSDFM", "IDSSM"],
                save_dir=f'{save_dir}/all',
                show_plot=False
            )
            visualize.plot_mean_are_comparison(
                results_dfs=[res_idssm],
                labels=["IDSSM"],
                save_dir=f'{save_dir}/baseline',
                show_plot=False
            )
            visualize.plot_variance_are_comparison(
                results_dfs=[res_idssm],
                labels=["IDSSM"],
                save_dir=f'{save_dir}/baseline',
                show_plot=False
            )
            
            print("[Visualization] All comparative plots saved successfully.")
        except Exception as e:
            print(f"[Visualization] Error generating comparative plots: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[Visualization] Skipping comparative plots because one or both models failed.")

def save_results_dir():
    id = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    save_path = f"results/{id}"
    return save_path

if __name__ == "__main__":
    run_full_experiment(
        train_path="./data/train_FD001.txt",
        test_path="./data/test_FD001.txt",
        rul_path="./data/RUL_FD001.txt",
        save_dir=save_results_dir()
    )