"""
MSDFM vs ID-SSM 비교 실험 파이프라인

MSDFM: 논문 "Remaining useful life prediction based on a multi-sensor data fusion model" 구현
ID-SSM: 컨셉 문서 기반 개선 모델 구현
"""

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
    # 컨셉 문서: "x̃_k = k / T_life로, 선형적인 가상의 상태값"
    train['state_x'] = train['time_cycles'] / train['max_cycle']

    # Drop constant/uninformative sensors (based on typical C-MAPSS analysis)
    drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
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
        title_suffix="FD001",
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
        X_seq = unit_data[best_sensors].values
        time_cycles = unit_data['time_cycles'].values
        
        if len(X_seq) == 0: continue
            
        final_true_rul = true_ruls[i]
        total_life = time_cycles[-1] + final_true_rul
        
        # Initialize PF with the first measurement
        pf = msdfm.particle_filter.ParticleFilter(params, best_sensors, initial_data=X_seq[0])
        pred_rul_t = pf.estimate_rul()
        
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
# 2. ID-SSM Pipeline (Proposed - 컨셉 문서 기반)
# ---------------------------------------------------------
def run_idssm_step(train_df, test_df, y_test, features, drift_stats, save_dir):
    """
    ID-SSM (Interaction-aware Deep State-Space Model) 파이프라인
    
    컨셉 문서 구현:
    1. GAT 기반 센서 퓨전
    2. MNN 기반 단조 관측 모델
    3. 통합 Loss Function (L_recon + λ₁·L_mono + λ₂·L_state)
    4. MNN 역매핑 기반 RUL 추정
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("\n" + "="*50)
    print("Running ID-SSM Pipeline (Concept Document Implementation)")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[IDSSM] Using device: {device}")
    # ================================================================
    # 2.1 데이터 준비
    # ================================================================
    X_train = torch.FloatTensor(train_df[features].values)
    state_train = torch.FloatTensor(train_df['state_x'].values).unsqueeze(-1)
    
    # 기본 데이터셋 (포인트 단위)
    dataset = torch.utils.data.TensorDataset(X_train, state_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 시퀀스 데이터셋 (단조성 Loss용) - 선택적
    use_sequence_training = True
    seq_length = 10
    
    if use_sequence_training:
        try:
            seq_dataset = idssm.SequenceDataset(train_df, features, seq_length=seq_length)
            seq_dataloader = torch.utils.data.DataLoader(
                seq_dataset, batch_size=32, shuffle=True
            )
            print(f"[ID-SSM] Sequence dataset created: {len(seq_dataset)} sequences")
        except Exception as e:
            print(f"[ID-SSM] Sequence dataset creation failed: {e}")
            use_sequence_training = False

    # ================================================================
    # 2.2 모델 초기화
    # ================================================================
    model = idssm.IDSSM(num_sensors=len(features), latent_dim=8, hidden_dim=16).to(device)
    
    # 컨셉 문서 Loss Function
    criterion = idssm.IDSSMLoss(lambda_mono=0.1, lambda_state=0.01)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    loss_history = []
    loss_components_history = {'recon': [], 'mono': [], 'total': []}

    # ================================================================
    # 2.3 모델 학습
    # ================================================================
    print("[ID-SSM] Training GAT-MNN Network...")
    print(f"  - Loss: L_recon + {criterion.lambda_mono}*L_mono + {criterion.lambda_state}*L_state")
    
    model.train()
    num_epochs = 50
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_loss_recon = 0
        epoch_loss_mono = 0
        num_batches = 0
        
        if use_sequence_training:
            # 시퀀스 기반 학습 (단조성 Loss 포함)
            for seq_x, seq_state in seq_dataloader:
                seq_x, seq_state = seq_x.to(device), seq_state.to(device)
                # seq_x: (batch, seq_len, num_sensors)
                # seq_state: (batch, seq_len)
                
                batch_size, seq_len, _ = seq_x.shape
                
                # 시퀀스의 각 시점에 대해 인코딩
                z_enc_seq = []
                z_dec_seq = []
                
                for t in range(seq_len):
                    z_enc_t, _ = model.encode(seq_x[:, t, :])
                    z_dec_t = model.decode(seq_state[:, t:t+1])
                    z_enc_seq.append(z_enc_t)
                    z_dec_seq.append(z_dec_t)
                
                z_enc_seq = torch.stack(z_enc_seq, dim=1)  # (batch, seq, latent_dim)
                z_dec_seq = torch.stack(z_dec_seq, dim=1)
                
                # 마지막 시점의 Loss (또는 전체 평균)
                z_enc_last = z_enc_seq[:, -1, :]
                z_dec_last = z_dec_seq[:, -1, :]
                
                optimizer.zero_grad()
                
                # 통합 Loss 계산
                loss, loss_dict = criterion(
                    z_enc=z_enc_last,
                    z_dec=z_dec_last,
                    z_enc_seq=z_enc_seq,
                    w_trend=model.w_trend,
                    state_x=seq_state[:, -1:]
                )
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss_dict['total']
                epoch_loss_recon += loss_dict['recon']
                epoch_loss_mono += loss_dict['mono']
                num_batches += 1
        else:
            # 포인트 단위 학습 (단조성 Loss 없음)
            for bx, by in dataloader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                z_enc, z_dec = model(bx, by)
                
                loss, loss_dict = criterion(z_enc, z_dec)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss_dict['total']
                epoch_loss_recon += loss_dict['recon']
                num_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_loss_recon / num_batches
        avg_mono = epoch_loss_mono / num_batches if use_sequence_training else 0
        
        loss_history.append(avg_loss)
        loss_components_history['total'].append(avg_loss)
        loss_components_history['recon'].append(avg_recon)
        loss_components_history['mono'].append(avg_mono)
        
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Total={avg_loss:.6f}, Recon={avg_recon:.6f}, Mono={avg_mono:.6f}")

    # ================================================================
    # 2.4 측정 노이즈 추정
    # ================================================================
    model.eval()
    with torch.no_grad():
        X_train_dev = X_train.to(device)
        state_train_dev = state_train.to(device)
        
        z_enc_full, z_dec_full = model(X_train_dev, state_train_dev)
        residuals = z_enc_full - z_dec_full
    
    err_norm = torch.norm(residuals, dim=1).cpu()
    meas_noise_std = err_norm.median().item()
    meas_noise_std_robust = max(meas_noise_std, 1e-3) * 3.0  # 스케일 조정
    print(f"[ID-SSM] Estimated Measurement Noise Std: {meas_noise_std_robust:.6f}")

    # ================================================================
    # 2.5 테스트 평가 (컨셉 문서 Section 4.2)
    # ================================================================
    print("[ID-SSM] Evaluating on Test Set...")
    print("  - Using MNN inverse mapping for state estimation")
    
    all_results = []
    pred_ruls = []
    true_ruls = y_test['RUL'].values
    test_units = sorted(test_df['unit_nr'].unique())
    final_unit_predictions = []
    
    # Attention weights 저장 (마지막 유닛용)
    last_attention_weights = None
    
    start_time = time.time()

    for i, unit_id in enumerate(test_units):
        unit_data = test_df[test_df['unit_nr'] == unit_id]
        X_seq = unit_data[features].values
        time_cycles = unit_data['time_cycles'].values
        
        final_true_rul = true_ruls[i]
        total_life = time_cycles[-1] + final_true_rul
        
        # 파티클 필터 초기화
        pf = idssm.ParticleFilterRUL(
            num_particles=5000,
            drift_mean=drift_stats['mean'],
            drift_std=drift_stats['std'],
            measurement_noise_std=meas_noise_std_robust
        )
        pf.initialize()
        
        for t in range(len(X_seq)):
            # 상태 전이 예측
            pf.predict(dt=1.0, sigma_B=0.001)
            
            # GAT 인코딩
            x_obs = torch.FloatTensor(X_seq[t]).unsqueeze(0).to(device)
            with torch.no_grad():
                z_out, attn = model.encode(x_obs)
                z_obs = z_out.cpu().numpy().flatten()
                
                # 마지막 유닛의 마지막 시점 attention 저장
                if i == len(test_units) - 1 and t == len(X_seq) - 1:
                    last_attention_weights = attn

            # 컨셉 문서: MNN 역매핑 기반 측정 업데이트
            pf.update_with_mnn(z_obs, model, use_inverse_mapping=True)
            
            # 리샘플링
            if pf.neff() < pf.N / 2:
                pf.fuzzy_resample()
            
            # RUL 추정
            pred_rul_t = pf.estimate_rul()
            pred_rul_t = min(pred_rul_t, 150.0)  # 상한 클리핑
            
            # 결과 기록
            current_age = time_cycles[t]
            are = np.abs(total_life - pred_rul_t - current_age) / total_life * 100.0
            
            all_results.append({
                'unit_nr': unit_id,
                'life_percent': (current_age / total_life) * 100,
                'ARE': are,
                't_nk': current_age,
                't_nKn': total_life,
                'Model': 'ID-SSM'
            })
            
            if t == len(X_seq) - 1:
                pred_ruls.append(pred_rul_t)
                final_unit_predictions.append((unit_id, final_true_rul, pred_rul_t))

        if (i+1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_units)} units...")
            
    elapsed = time.time() - start_time

    # ================================================================
    # 2.6 결과 집계 및 저장
    # ================================================================
    results_df = pd.DataFrame(all_results)
    
    # WARE 계산
    ware_list = []
    for unit_id, df_u in results_df.groupby('unit_nr'):
        t_nKn = df_u['t_nKn'].iloc[0]
        term = (df_u['t_nk'] * df_u['ARE'] / t_nKn).mean()
        ware_list.append(term)

    overall_ware = np.mean(ware_list)
    rmse = np.sqrt(mean_squared_error(true_ruls, pred_ruls))

    print(f"[ID-SSM] Done in {elapsed:.2f}s | RMSE: {rmse:.4f} | WARE: {overall_ware:.2f}%")

    # 결과 저장
    results_df.to_csv(os.path.join(save_dir, 'idssm_results.csv'), index=False)
    utils.plot_training_loss(loss_history, save_dir)
    utils.save_unit_predictions(final_unit_predictions, rmse, overall_ware, save_dir)
    utils.save_percentile_stats(results_df, save_dir)
    utils.plot_lifetime_performance(results_df, save_dir)
    utils.plot_rul_comparison(true_ruls, pred_ruls, rmse, save_dir)
    
    # Loss 컴포넌트 히스토리 저장
    _save_loss_components(loss_components_history, save_dir)
    
    # GAT Attention Heatmap 저장
    if last_attention_weights is not None:
        print(f"[ID-SSM] Saving GAT Attention Heatmap...")
        visualize.plot_gat_attention_heatmap(
            last_attention_weights,
            sensor_names=features,
            save_dir=save_dir,
            show_plot=False
        )
    
    return rmse, overall_ware, results_df


def _save_loss_components(loss_history, save_dir):
    """Loss 컴포넌트 히스토리 저장"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(loss_history['total']) + 1)
    
    ax.plot(epochs, loss_history['total'], 'b-', label='Total Loss', linewidth=2)
    ax.plot(epochs, loss_history['recon'], 'g--', label='Reconstruction Loss', linewidth=2)
    if any(loss_history['mono']):
        ax.plot(epochs, loss_history['mono'], 'r:', label='Monotonicity Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('ID-SSM Training Loss Components', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    save_path = os.path.join(save_dir, 'loss_components.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Loss components plot saved to: {save_path}")


# ---------------------------------------------------------
# 3. Main Execution
# ---------------------------------------------------------
def run_full_experiment(train_path, test_path, rul_path, save_dir):
    set_seed(2024)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Experiment Results will be saved to: {save_dir}")

    # 1. Load Data
    print("Loading and Processing Data...")
    train, test, y_test, feats, lifetimes, drift = load_and_process_data(train_path, test_path, rul_path)
    
    # Initialize result containers
    res_msdfm = None
    res_idssm = None

    # 3. Run ID-SSM
    try:
        rmse_b, ware_b, res_idssm = run_idssm_step(train, test, y_test, feats, drift, f'{save_dir}/idssm')
    except Exception as e:
        print(f"[ID-SSM] Failed: {e}")
        import traceback
        traceback.print_exc()
        rmse_b, ware_b = float('nan'), float('nan')
        
    # 2. Run MSDFM
    try:
        rmse_a, ware_a, res_msdfm = run_msdfm_step(train, test, y_test, feats, lifetimes, f'{save_dir}/msdfm')
    except Exception as e:
        print(f"[MSDFM] Failed: {e}")
        import traceback
        traceback.print_exc()
        rmse_a, ware_a = float('nan'), float('nan')
        

    # 4. Final Comparison & Visualization
    print("\n" + "="*50)
    print(" FINAL COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Model':<10} | {'RMSE':<10} | {'WARE (%)':<10}")
    print("-" * 36)
    print(f"{'MSDFM':<10} | {rmse_a:<10.4f} | {ware_a:<10.2f}")
    print(f"{'ID-SSM':<10} | {rmse_b:<10.4f} | {ware_b:<10.2f}")
    print("="*50)
    
    # Save comparison text
    with open(os.path.join(save_dir, 'final_comparison.txt'), 'w') as f:
        f.write("FINAL COMPARISON SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Model      | RMSE       | WARE (%)\n")
        f.write("-"*36 + "\n")
        f.write(f"MSDFM      | {rmse_a:.4f}     | {ware_a:.2f}\n")
        f.write(f"ID-SSM     | {rmse_b:.4f}     | {ware_b:.2f}\n")
        f.write("="*50 + "\n")

    # Comparative Plots
    if res_msdfm is not None and res_idssm is not None:
        print("\n[Visualization] Generating Comparative Plots...")
        try:
            visualize.plot_mean_are_comparison(
                results_dfs=[res_msdfm, res_idssm],
                labels=["MSDFM", "ID-SSM"],
                save_dir=f'{save_dir}/comparison',
                show_plot=False
            )
            visualize.plot_variance_are_comparison(
                results_dfs=[res_msdfm, res_idssm],
                labels=["MSDFM", "ID-SSM"],
                save_dir=f'{save_dir}/comparison',
                show_plot=False
            )
            print("[Visualization] All comparative plots saved successfully.")
        except Exception as e:
            print(f"[Visualization] Error generating comparative plots: {e}")
            import traceback
            traceback.print_exc()


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
