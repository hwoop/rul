import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import random
import os
import datetime

import utils
from model import idssm, msdfm

# ---------------------------------------------------------
# 0. 시드 설정 및 데이터 로드 함수 (변경 없음)
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

    train_max_cycle = train.groupby('unit_nr')['time_cycles'].max().reset_index()
    train_max_cycle.columns = ['unit_nr', 'max_cycle']
    train = train.merge(train_max_cycle, on='unit_nr', how='left')
    train['state_x'] = train['time_cycles'] / train['max_cycle']

    drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    features = [c for c in train.columns if c.startswith('s_') and c not in drop_sensors]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])

    drift_stats = (1.0 / train_max_cycle['max_cycle']).agg(['mean', 'std']).to_dict()
    
    return train, test, y_test, features, drift_stats

# ---------------------------------------------------------
# 2. Main Pipeline Function
# ---------------------------------------------------------
def run_fd001_pipeline(train_path, test_path, rul_path, D_state=1.0, save_dir="results"):
    
    set_seed(2024)
    print(f"Results will be saved to: {save_dir}")


    # ==========================================
    # Model A: Baseline (MSDFM)
    # ==========================================       
    print("Running Baseline: MSDFM...")
    msdfm = msdfm.MSDFM_Parameters()
    msdfm.estimate_state_params(lifetimes)
    msdfm.estimate_measurement_params(train_raw, feats)
    
    

    # 1. 데이터 로드
    print("Loading Data...")
    train_df, test_df, y_test, feats, drift_stats = load_and_process_data(train_path, test_path, rul_path)

    X_train = torch.FloatTensor(train_df[feats].values)
    state_train = torch.FloatTensor(train_df['state_x'].values).unsqueeze(-1)
    dataset = torch.utils.data.TensorDataset(X_train, state_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. 모델 학습
    model = IDSSM(num_sensors=len(feats), latent_dim=8)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    # [수정] Loss 기록용 리스트
    loss_history = []

    print("Stage 1: Training Measurement Model...")
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
        
        # [수정] Epoch Loss 저장
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    # Measurement Noise 추정
    model.eval()
    with torch.no_grad():
        z_enc_full, z_dec_full = model(X_train, state_train)
        residuals = z_enc_full - z_dec_full

    err_norm = torch.norm(residuals, dim=1)
    meas_noise_std = err_norm.median().item()
    meas_noise_std = max(meas_noise_std, 1e-3)
    
    NOISE_SCALE_FACTOR = 5.0 
    meas_noise_std_robust = meas_noise_std * NOISE_SCALE_FACTOR
    print(f"Robust PF Noise Std: {meas_noise_std_robust:.6f}")

    # 3. 추론 및 분석 (Stage 2)
    print("\nStage 2: Conducting Lifetime Performance Analysis...")

    all_results = [] 
    pred_ruls = []   
    true_ruls = y_test['RUL'].values 
    test_units = sorted(test_df['unit_nr'].unique())
    
    # [수정] Unit별 최종 결과 저장용 리스트 (Unit ID, Actual RUL, Pred RUL)
    final_unit_predictions = []

    for i, unit_id in enumerate(test_units):
        unit_data = test_df[test_df['unit_nr'] == unit_id]
        X_seq = unit_data[feats].values
        time_cycles = unit_data['time_cycles'].values
        
        final_true_rul = true_ruls[i]
        max_cycle_observed = time_cycles[-1]
        total_life = max_cycle_observed + final_true_rul
        
        pf = ParticleFilterRUL(5000, drift_stats['mean'], drift_stats['std'], meas_noise_std_robust)
        pf.initialize()
        
        for t in range(len(X_seq)):
            pf.predict()

            x_obs = torch.FloatTensor(X_seq[t]).unsqueeze(0)
            with torch.no_grad():
                z_out, _ = model.forward_encoder(x_obs)
                z_obs = z_out.numpy().flatten()

            pf.update(z_obs, model)

            if pf.neff() < pf.N / 2:
                pf.resample()
            
            pred_rul_t = pf.estimate_rul(time_cycles[t])
            pred_rul_t = min(pred_rul_t, 145.0) 

            current_age = time_cycles[t]
            life_percent = (current_age / total_life) * 100 
            are = np.abs(total_life - pred_rul_t - current_age) / total_life * 100.0
                
            all_results.append({
                'unit_nr': unit_id,
                'life_percent': life_percent,
                'ARE': are,
                't_nk': current_age,
                't_nKn': total_life
            })
            
            # 마지막 시점
            if t == len(X_seq) - 1:
                pred_ruls.append(pred_rul_t)
                # [수정] 리스트에 추가
                final_unit_predictions.append((unit_id, final_true_rul, pred_rul_t))
                print(f"Unit {unit_id:<5} (Actual RUL : {int(final_true_rul):<3}, Pred RUL : {pred_rul_t:.2f})")

        if (i+1) % 20 == 0:
            print(f"Processed {i+1}/{len(test_units)} units...")

    # 결과 집계
    results_df = pd.DataFrame(all_results)
    
    ware_list = []
    for unit_id, df_u in results_df.groupby('unit_nr'):
        Kn = len(df_u)
        t_nKn = df_u['t_nKn'].iloc[0]
        term = (df_u['t_nk'] * df_u['ARE'] / t_nKn).mean()
        ware_list.append(term)

    overall_ware = np.mean(ware_list)
    rmse = np.sqrt(mean_squared_error(true_ruls, pred_ruls))

    print("\n" + "="*50)
    print(f" Final Performance Summary")
    print("="*50)
    print(f" Test RMSE : {rmse:.4f}")
    print(f" Mean ARE (WARE) : {overall_ware:.2f}%")
    print("="*50)

    # 4. 결과 저장 및 시각화 (Utils 호출)
    print("\nSaving Results...")
    results_df.to_csv(os.path.join(save_dir, 'all_results.csv'), index=False)

    # 기존 그래프
    utils.plot_lifetime_performance(results_df, save_dir)
    utils.plot_rul_comparison(true_ruls, pred_ruls, rmse, save_dir)

    # [추가된 기능 호출]
    # 1. Loss 그래프
    utils.plot_training_loss(loss_history, save_dir)
    
    # 2. Unit별 RUL txt 저장
    utils.save_unit_predictions(final_unit_predictions, rmse, overall_ware, save_dir)
    
    # 3. Percentile 통계 txt 저장
    utils.save_percentile_stats(results_df, save_dir)

    return results_df

def save_results():
    id = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    save_path = f"results/{id}"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    return save_path


if __name__ == "__main__":
    results = run_fd001_pipeline(
        train_path="./data/train_FD001.txt",
        test_path="./data/test_FD001.txt",
        rul_path="./data/RUL_FD001.txt",
        D_state=1.0,
        save_dir=save_results()
    )