"""
visualize.py - MSDFM/IDSSM 모델 성능 비교 및 GAT Attention 시각화 모듈

제공 함수:
1. plot_mean_are_comparison: MSDFM, IDSSM 모델의 Mean ARE(%) 비교
2. plot_variance_are_comparison: MSDFM, IDSSM 모델의 Variance ARE(%) 비교
3. plot_gat_attention_heatmap: GAT Attention weights heatmap 시각화
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def _compute_are_statistics(results_df, bin_range=(10, 90)):
    """
    ARE 통계(Mean, Variance)를 Percentile 구간별로 계산

    Parameters:
    -----------
    results_df : pd.DataFrame
        'life_percent'와 'ARE' 컬럼을 포함하는 결과 데이터프레임
    bin_range : tuple
        분석할 percentile 범위 (기본값: 10~90)

    Returns:
    --------
    pd.DataFrame
        Percentile별 Mean ARE, Variance ARE 통계
    """
    bins = np.arange(0, 101, 10)
    labels = [i for i in range(10, 101, 10)]

    df = results_df.copy()
    df['percentile_bin'] = pd.cut(df['life_percent'], bins=bins, labels=labels)

    bin_stats = df.groupby('percentile_bin', observed=False)['ARE'].agg(['mean', 'var']).reset_index()
    bin_stats.columns = ['Percentile', 'Mean_ARE', 'Var_ARE']

    # 지정 범위 필터링
    bin_stats = bin_stats[
        bin_stats['Percentile'].astype(int).between(bin_range[0], bin_range[1])
    ]

    return bin_stats


def plot_mean_are_comparison(msdfm_results_df, idssm_results_df, save_dir=None,
                              figsize=(12, 7), show_plot=True):
    """
    MSDFM과 IDSSM 모델의 Mean ARE(%) 비교 시각화

    Parameters:
    -----------
    msdfm_results_df : pd.DataFrame
        MSDFM 모델의 결과 데이터프레임 (life_percent, ARE 컬럼 필요)
    idssm_results_df : pd.DataFrame
        IDSSM 모델의 결과 데이터프레임 (life_percent, ARE 컬럼 필요)
    save_dir : str, optional
        저장 디렉토리 경로. None이면 저장하지 않음
    figsize : tuple
        그래프 크기 (기본값: (12, 7))
    show_plot : bool
        플롯 표시 여부 (기본값: True)

    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure와 axis 객체
    """
    # ARE 통계 계산
    msdfm_stats = _compute_are_statistics(msdfm_results_df)
    idssm_stats = _compute_are_statistics(idssm_results_df)

    # 그래프 생성
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(msdfm_stats))
    width = 0.35

    # 막대 그래프
    bars1 = ax.bar(x - width/2, msdfm_stats['Mean_ARE'].values, width,
                   label='MSDFM', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, idssm_stats['Mean_ARE'].values, width,
                   label='IDSSM', color='#e74c3c', alpha=0.8, edgecolor='black')

    # 선 그래프 오버레이
    ax.plot(x - width/2, msdfm_stats['Mean_ARE'].values, 'o-',
            color='#2980b9', linewidth=2, markersize=8)
    ax.plot(x + width/2, idssm_stats['Mean_ARE'].values, 's-',
            color='#c0392b', linewidth=2, markersize=8)

    # 축 설정
    ax.set_xlabel('Life Percentile (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean ARE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Mean ARE Comparison: MSDFM vs IDSSM', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(msdfm_stats['Percentile'].astype(int).values)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    # 막대 위에 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # 저장
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'mean_are_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Mean ARE comparison plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_variance_are_comparison(msdfm_results_df, idssm_results_df, save_dir=None,
                                  figsize=(12, 7), show_plot=True, use_ratio=True):
    """
    MSDFM과 IDSSM 모델의 Variance ARE(%) 비교 시각화

    Parameters:
    -----------
    msdfm_results_df : pd.DataFrame
        MSDFM 모델의 결과 데이터프레임 (life_percent, ARE 컬럼 필요)
    idssm_results_df : pd.DataFrame
        IDSSM 모델의 결과 데이터프레임 (life_percent, ARE 컬럼 필요)
    save_dir : str, optional
        저장 디렉토리 경로. None이면 저장하지 않음
    figsize : tuple
        그래프 크기 (기본값: (12, 7))
    show_plot : bool
        플롯 표시 여부 (기본값: True)
    use_ratio : bool
        Variance를 ratio (/ 10000)로 표시할지 여부 (기본값: True)

    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure와 axis 객체
    """
    # ARE 통계 계산
    msdfm_stats = _compute_are_statistics(msdfm_results_df)
    idssm_stats = _compute_are_statistics(idssm_results_df)

    # Variance 값 처리
    if use_ratio:
        msdfm_var = msdfm_stats['Var_ARE'].values / 10000.0
        idssm_var = idssm_stats['Var_ARE'].values / 10000.0
        ylabel = 'Variance of ARE (Ratio)'
    else:
        msdfm_var = msdfm_stats['Var_ARE'].values
        idssm_var = idssm_stats['Var_ARE'].values
        ylabel = 'Variance of ARE (%²)'

    # 그래프 생성
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(msdfm_stats))
    width = 0.35

    # 막대 그래프
    bars1 = ax.bar(x - width/2, msdfm_var, width,
                   label='MSDFM', color='#9b59b6', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, idssm_var, width,
                   label='IDSSM', color='#f39c12', alpha=0.8, edgecolor='black')

    # 선 그래프 오버레이
    ax.plot(x - width/2, msdfm_var, 'o-',
            color='#8e44ad', linewidth=2, markersize=8)
    ax.plot(x + width/2, idssm_var, 's-',
            color='#d68910', linewidth=2, markersize=8)

    # 축 설정
    ax.set_xlabel('Life Percentile (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title('Variance of ARE Comparison: MSDFM vs IDSSM', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(msdfm_stats['Percentile'].astype(int).values)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    # 막대 위에 값 표시
    fmt = '{:.4f}' if use_ratio else '{:.2f}'
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, rotation=45)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, rotation=45)

    plt.tight_layout()

    # 저장
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'variance_are_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Variance ARE comparison plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_gat_attention_heatmap(attention_weights, sensor_names=None, save_dir=None,
                                figsize=(12, 10), show_plot=True, cmap='viridis',
                                title_suffix=""):
    """
    GAT (Graph Attention Network)의 Attention weights heatmap 시각화

    Parameters:
    -----------
    attention_weights : np.ndarray or torch.Tensor
        Attention weights 행렬. Shape: (N, N) 또는 (Batch, N, N)
        N은 센서(노드) 수
    sensor_names : list, optional
        센서 이름 리스트. None이면 자동 생성 (Sensor 1, Sensor 2, ...)
    save_dir : str, optional
        저장 디렉토리 경로. None이면 저장하지 않음
    figsize : tuple
        그래프 크기 (기본값: (12, 10))
    show_plot : bool
        플롯 표시 여부 (기본값: True)
    cmap : str
        Colormap 이름 (기본값: 'viridis')
    title_suffix : str
        제목에 추가할 접미사 (예: 샘플 번호)

    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure와 axis 객체
    """
    # Tensor를 numpy로 변환
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    # Batch 차원 처리 (평균 또는 첫 번째 샘플 사용)
    if attention_weights.ndim == 3:
        # 배치 평균
        attn_matrix = attention_weights.mean(axis=0)
    else:
        attn_matrix = attention_weights

    n_sensors = attn_matrix.shape[0]

    # 센서 이름 생성
    if sensor_names is None:
        sensor_names = [f'S{i+1}' for i in range(n_sensors)]

    # 그래프 생성
    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    sns.heatmap(attn_matrix,
                xticklabels=sensor_names,
                yticklabels=sensor_names,
                cmap=cmap,
                annot=True if n_sensors <= 16 else False,
                fmt='.3f' if n_sensors <= 16 else None,
                annot_kws={'size': 8} if n_sensors <= 16 else None,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Attention Weight', 'shrink': 0.8},
                ax=ax)

    # 축 설정
    title = 'GAT Attention Heatmap'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Target Sensor (j)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Source Sensor (i)', fontsize=12, fontweight='bold')

    # 틱 레이블 회전
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # 저장
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = 'gat_attention_heatmap'
        if title_suffix:
            filename += f'_{title_suffix.replace(" ", "_")}'
        save_path = os.path.join(save_dir, f'{filename}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] GAT attention heatmap saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_gat_attention_multi_sample(attention_weights_list, sensor_names=None,
                                     save_dir=None, figsize=(16, 12),
                                     show_plot=True, cmap='viridis',
                                     sample_labels=None, ncols=2):
    """
    여러 샘플의 GAT Attention heatmap을 subplot으로 시각화

    Parameters:
    -----------
    attention_weights_list : list
        여러 샘플의 attention weights 리스트
    sensor_names : list, optional
        센서 이름 리스트
    save_dir : str, optional
        저장 디렉토리 경로
    figsize : tuple
        전체 figure 크기
    show_plot : bool
        플롯 표시 여부
    cmap : str
        Colormap 이름
    sample_labels : list, optional
        각 샘플의 레이블 리스트
    ncols : int
        subplot 열 수

    Returns:
    --------
    tuple
        (fig, axes) matplotlib figure와 axes 배열
    """
    n_samples = len(attention_weights_list)
    nrows = (n_samples + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    if sample_labels is None:
        sample_labels = [f'Sample {i+1}' for i in range(n_samples)]

    for idx, (attn, label) in enumerate(zip(attention_weights_list, sample_labels)):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        # Tensor 변환
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu().numpy()

        if attn.ndim == 3:
            attn = attn[0]  # 첫 번째 배치 사용

        n_sensors = attn.shape[0]
        if sensor_names is None:
            names = [f'S{i+1}' for i in range(n_sensors)]
        else:
            names = sensor_names

        # Heatmap
        sns.heatmap(attn,
                    xticklabels=names,
                    yticklabels=names,
                    cmap=cmap,
                    annot=False,
                    square=True,
                    linewidths=0.3,
                    cbar=True,
                    cbar_kws={'shrink': 0.6},
                    ax=ax)

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Target', fontsize=9)
        ax.set_ylabel('Source', fontsize=9)
        ax.tick_params(axis='both', labelsize=7)

    # 빈 subplot 제거
    for idx in range(n_samples, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')

    plt.suptitle('GAT Attention Heatmaps - Multiple Samples',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 저장
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'gat_attention_multi_sample.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Multi-sample attention heatmap saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, axes


def extract_gat_attention(model, x_sensors):
    """
    IDSSM 모델에서 GAT attention weights 추출

    Parameters:
    -----------
    model : IDSSM
        학습된 IDSSM 모델
    x_sensors : torch.Tensor
        입력 센서 데이터. Shape: (batch_size, num_sensors)

    Returns:
    --------
    tuple
        (z_encoded, attention_weights)
        - z_encoded: 인코딩된 잠재 벡터
        - attention_weights: GAT attention 가중치 행렬
    """
    model.eval()
    with torch.no_grad():
        z_encoded, attention_weights = model.encode(x_sensors)
    return z_encoded, attention_weights


# =========================================================
# 통합 비교 시각화 함수
# =========================================================
def plot_combined_are_comparison(msdfm_results_df, idssm_results_df, save_dir=None,
                                  figsize=(14, 6), show_plot=True):
    """
    Mean ARE와 Variance ARE를 하나의 figure에 subplot으로 표시

    Parameters:
    -----------
    msdfm_results_df : pd.DataFrame
        MSDFM 모델 결과
    idssm_results_df : pd.DataFrame
        IDSSM 모델 결과
    save_dir : str, optional
        저장 디렉토리
    figsize : tuple
        figure 크기
    show_plot : bool
        플롯 표시 여부

    Returns:
    --------
    tuple
        (fig, axes)
    """
    msdfm_stats = _compute_are_statistics(msdfm_results_df)
    idssm_stats = _compute_are_statistics(idssm_results_df)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    x = np.arange(len(msdfm_stats))
    width = 0.35

    # === Left: Mean ARE ===
    ax1 = axes[0]
    ax1.bar(x - width/2, msdfm_stats['Mean_ARE'].values, width,
            label='MSDFM', color='#3498db', alpha=0.8, edgecolor='black')
    ax1.bar(x + width/2, idssm_stats['Mean_ARE'].values, width,
            label='IDSSM', color='#e74c3c', alpha=0.8, edgecolor='black')

    ax1.plot(x - width/2, msdfm_stats['Mean_ARE'].values, 'o-', color='#2980b9', linewidth=2)
    ax1.plot(x + width/2, idssm_stats['Mean_ARE'].values, 's-', color='#c0392b', linewidth=2)

    ax1.set_xlabel('Life Percentile (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean ARE (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Mean ARE Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(msdfm_stats['Percentile'].astype(int).values)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')

    # === Right: Variance ARE ===
    ax2 = axes[1]
    msdfm_var = msdfm_stats['Var_ARE'].values / 10000.0
    idssm_var = idssm_stats['Var_ARE'].values / 10000.0

    ax2.bar(x - width/2, msdfm_var, width,
            label='MSDFM', color='#9b59b6', alpha=0.8, edgecolor='black')
    ax2.bar(x + width/2, idssm_var, width,
            label='IDSSM', color='#f39c12', alpha=0.8, edgecolor='black')

    ax2.plot(x - width/2, msdfm_var, 'o-', color='#8e44ad', linewidth=2)
    ax2.plot(x + width/2, idssm_var, 's-', color='#d68910', linewidth=2)

    ax2.set_xlabel('Life Percentile (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Variance of ARE (Ratio)', fontsize=11, fontweight='bold')
    ax2.set_title('Variance of ARE Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(msdfm_stats['Percentile'].astype(int).values)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

    plt.suptitle('MSDFM vs IDSSM Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 저장
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'combined_are_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Combined ARE comparison plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, axes


# =========================================================
# 사용 예제
# =========================================================
if __name__ == "__main__":
    # 예제 실행을 위한 더미 데이터 생성
    print("=" * 60)
    print("visualize.py - 사용 예제")
    print("=" * 60)

    # 1. 더미 ARE 결과 데이터 생성
    np.random.seed(42)
    n_samples = 1000

    # MSDFM 더미 데이터
    msdfm_data = {
        'life_percent': np.random.uniform(5, 95, n_samples),
        'ARE': np.random.exponential(10, n_samples) + 5
    }
    msdfm_df = pd.DataFrame(msdfm_data)

    # IDSSM 더미 데이터 (성능이 더 좋다고 가정)
    idssm_data = {
        'life_percent': np.random.uniform(5, 95, n_samples),
        'ARE': np.random.exponential(7, n_samples) + 3
    }
    idssm_df = pd.DataFrame(idssm_data)

    print("\n[예제 1] Mean ARE 비교 플롯")
    plot_mean_are_comparison(msdfm_df, idssm_df, show_plot=False)

    print("\n[예제 2] Variance ARE 비교 플롯")
    plot_variance_are_comparison(msdfm_df, idssm_df, show_plot=False)

    print("\n[예제 3] GAT Attention Heatmap")
    # 더미 attention weights (16개 센서)
    dummy_attention = np.random.dirichlet(np.ones(16), size=16)
    sensor_names = [f's_{i}' for i in range(2, 18)]  # s_2 ~ s_17 (s_1, s_5, s_10 등 제외 가정)
    plot_gat_attention_heatmap(dummy_attention, sensor_names=sensor_names, show_plot=False)

    print("\n[예제 4] 통합 ARE 비교 플롯")
    plot_combined_are_comparison(msdfm_df, idssm_df, show_plot=False)

    print("\n" + "=" * 60)
    print("모든 예제 실행 완료!")
    print("=" * 60)
