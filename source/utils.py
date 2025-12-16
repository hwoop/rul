import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def make_save_dir(base_dir="results"):
    """결과 저장을 위한 디렉토리 생성"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

# ---------------------------------------------------------
# [추가 기능 1] 학습 Loss 그래프 저장
# ---------------------------------------------------------
def plot_training_loss(loss_history, save_dir):
    """
    Epoch별 Loss 변화를 그래프로 저장
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b', label='Training Loss')
    
    plt.title('Training Loss per Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    save_path = os.path.join(save_dir, 'training_loss.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Training loss plot saved to: {save_path}")

# ---------------------------------------------------------
# [추가 기능 2] Unit별 RUL 예측 결과 텍스트 저장
# ---------------------------------------------------------
def save_unit_predictions(prediction_list, rmse, ware, save_dir):
    """
    Final Summary와 Unit별 예측 결과(unit_id, true, pred)를 .txt 파일로 저장
    """
    save_path = os.path.join(save_dir, 'unit_predictions.txt')
    
    with open(save_path, 'w') as f:
        # 1. Final Summary 상단 작성
        f.write("=" * 50 + "\n")
        f.write(" Final Performance Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f" Test RMSE : {rmse:.4f}\n")
        f.write(f" Mean ARE (WARE) : {ware:.2f}%\n")
        f.write("=" * 50 + "\n\n")

        # 2. Unit별 결과 테이블 작성
        f.write(f"{'Unit':<6} | {'Actual RUL':<12} | {'Pred RUL':<12}\n")
        f.write("-" * 36 + "\n")
        
        for unit_id, true_rul, pred_rul in prediction_list:
            f.write(f"{unit_id:<6} | {true_rul:<12.2f} | {pred_rul:<12.2f}\n")
            
    print(f"[Saved] Unit predictions (with summary) saved to: {save_path}")

# ---------------------------------------------------------
# [추가 기능 3] Percentile별 통계 결과 텍스트 저장
# ---------------------------------------------------------
def save_percentile_stats(results_df, save_dir):
    """
    Percentile별 Mean ARE와 Variance Ratio를 계산하여 .txt 파일로 저장
    """
    # 통계 계산 로직 (Plot 함수와 동일한 로직 적용)
    bins = np.arange(0, 101, 10)
    labels = [i for i in range(10, 101, 10)]
    
    df = results_df.copy()
    df['percentile_bin'] = pd.cut(df['life_percent'], bins=bins, labels=labels)

    # 구간별 통계 산출
    bin_stats = df.groupby('percentile_bin', observed=False)['ARE'].agg(['mean', 'var']).reset_index()
    bin_stats.columns = ['Percentile', 'Mean_ARE_Percent', 'Var_ARE_Percent']

    # 10~90 구간만 필터링
    bin_stats = bin_stats[bin_stats['Percentile'].astype(int).between(10, 90)]

    # 단위 변환
    bin_stats['Var_ARE_Ratio'] = bin_stats['Var_ARE_Percent'] / 10000.0

    # 텍스트 파일 저장
    save_path = os.path.join(save_dir, 'percentile_stats.txt')
    
    with open(save_path, 'w') as f:
        f.write(f"{'Percentile':<12} | {'Mean_ARE_Percent':<18} | {'Var_ARE_Ratio':<15}\n")
        f.write("-" * 50 + "\n")
        
        for _, row in bin_stats.iterrows():
            f.write(f"{int(row['Percentile']):<12} | {row['Mean_ARE_Percent']:<18.6f} | {row['Var_ARE_Ratio']:<15.6f}\n")
            
    print(f"[Saved] Percentile stats saved to: {save_path}")

# ---------------------------------------------------------
# 기존 시각화 함수들
# ---------------------------------------------------------
def plot_lifetime_performance(results_df, save_dir):
    # (기존 코드와 동일, 생략하지 않고 그대로 유지하세요)
    bins = np.arange(0, 101, 10)
    labels = [i for i in range(10, 101, 10)]
    
    df = results_df.copy()
    df['percentile_bin'] = pd.cut(df['life_percent'], bins=bins, labels=labels)

    bin_stats = df.groupby('percentile_bin', observed=False)['ARE'].agg(['mean', 'var']).reset_index()
    bin_stats.columns = ['Percentile', 'Mean_ARE_Percent', 'Var_ARE_Percent']
    bin_stats = bin_stats[bin_stats['Percentile'].astype(int).between(10, 90)]
    bin_stats['Var_ARE_Ratio'] = bin_stats['Var_ARE_Percent'] / 10000.0

    fig, ax1 = plt.subplots(figsize=(10, 6))
    line1 = ax1.plot(bin_stats['Percentile'], bin_stats['Mean_ARE_Percent'], 
                     marker='o', color='blue', label='Mean ARE (%)', linewidth=2)
    ax1.set_xlabel('Life Percentile (%)', fontsize=12)
    ax1.set_ylabel('Mean ARE (%)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    line2 = ax2.plot(bin_stats['Percentile'], bin_stats['Var_ARE_Ratio'], 
                     marker='s', color='red', linestyle='--', label='Variance of ARE (Ratio)', linewidth=2)
    ax2.set_ylabel('Variance of ARE (Ratio)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')

    lines = line1 + line2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='upper center')

    plt.title('Performance over Lifetime (Unit Corrected)', fontsize=14)
    save_path = os.path.join(save_dir, 'performance_over_lifetime_corrected.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Lifetime plot saved to: {save_path}")


def plot_rul_comparison(true_ruls, pred_ruls, rmse, save_dir):
    # (기존 코드와 동일)
    plt.figure(figsize=(14, 6))
    unit_indices = np.arange(1, len(true_ruls) + 1)
    plt.plot(unit_indices, true_ruls, 'o-', label='Actual RUL', color='black', alpha=0.6, markersize=5)
    plt.plot(unit_indices, pred_ruls, 'x-', label='Predicted RUL', color='red', alpha=0.8, markersize=6)
    plt.fill_between(unit_indices, true_ruls, pred_ruls, color='gray', alpha=0.2, label='Prediction Error')

    plt.title(f'RUL Prediction Comparison (Test RMSE: {rmse:.2f})', fontsize=14)
    plt.xlabel('Test Unit Number', fontsize=12)
    plt.ylabel('Remaining Useful Life (Cycles)', fontsize=12)
    step = max(1, len(true_ruls) // 20)
    plt.xticks(np.arange(0, len(true_ruls) + 1, step)) 
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = os.path.join(save_dir, 'rul_comparison_all_units.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Comparison plot saved to: {save_path}")