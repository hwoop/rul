"""
visualize.py - MSDFM/ID-SSM 모델 성능 비교 및 GAT Attention 시각화 모듈

제공 함수:
1. plot_mean_are_comparison: MSDFM, ID-SSM 모델의 Mean ARE(%) 비교
2. plot_variance_are_comparison: MSDFM, ID-SSM 모델의 Variance ARE(%) 비교
3. plot_gat_attention_heatmap: GAT Attention weights heatmap 시각화
"""

import os
import re
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


def plot_mean_are_comparison(results_dfs, labels, save_dir=None, figsize=(12, 7), show_plot=True):
    """
    여러 모델의 Mean ARE(%)를 논문(MSDFM Paper) 수치와 비교하여 Line Plot으로 시각화.

    Parameters:
    -----------
    results_dfs : list of pd.DataFrame
        비교할 모델들의 결과 데이터프레임 리스트 (각 df는 life_percent, ARE 컬럼 필요)
    labels : list of str
        각 모델의 라벨 리스트 (results_dfs와 길이가 같아야 함)
    save_dir : str, optional
        저장 디렉토리 경로. None이면 저장하지 않음
    figsize : tuple
        그래프 크기
    show_plot : bool
        플롯 표시 여부
    """
    # 1. 논문 하드코딩 데이터 (Paper)
    paper_percentiles = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    paper_mean_are = np.array([16.3, 17.1, 14.6, 8.8, 6.3, 4.2, 3.5, 2.5, 2.0])

    fig, ax = plt.subplots(figsize=figsize)

    # 2. Paper Data Plot (기본 비교군: 회색 점선)
    ax.plot(paper_percentiles, paper_mean_are, 'o--', color='gray', label='MSDFM(Paper)',
            linewidth=2, alpha=0.6, markersize=6)
    
    # Paper 값 주석 (약간 아래쪽)
    for x, y in zip(paper_percentiles, paper_mean_are):
        ax.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0, -15), 
                    ha='center', fontsize=8, color='gray')

    # 3. 모델별 데이터 Plot
    # 색상 팔레트 생성 (모델 개수에 맞춰 자동 할당)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dfs)))

    for i, (df, label) in enumerate(zip(results_dfs, labels)):
        # 통계 계산 (_compute_are_statistics는 기존 visualize.py 함수 활용)
        stats = _compute_are_statistics(df, bin_range=(10, 90))
        
        # 결측 구간(bin)이 있을 수 있으므로 Paper Percentile 기준으로 병합
        base_df = pd.DataFrame({'Percentile': paper_percentiles})
        stats['Percentile'] = stats['Percentile'].astype(int)
        merged = pd.merge(base_df, stats, on='Percentile', how='left')
        
        # Line Plot
        ax.plot(merged['Percentile'], merged['Mean_ARE'], 's-', label=label,
                color=colors[i], linewidth=2.5, markersize=8)

        # 값 주석 (모델 값)
        for x, y in zip(merged['Percentile'], merged['Mean_ARE']):
            if not np.isnan(y):
                ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 8), 
                            ha='center', fontsize=9, fontweight='bold', color=colors[i])

    # 축 및 스타일 설정
    ax.set_xlabel('Life Percentile (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean ARE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Mean ARE Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(paper_percentiles)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # 저장
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'mean_are_comparison_multi.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Mean ARE comparison plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_variance_are_comparison(results_dfs, labels, save_dir=None, figsize=(12, 7), show_plot=True):
    """
    여러 모델의 Variance of ARE를 논문(MSDFM Paper) 수치와 비교 시각화.
    Y축은 10^-3 스케일로 표시 (값에 1000을 곱함).

    Parameters:
    -----------
    results_dfs : list of pd.DataFrame
        비교할 모델들의 결과 데이터프레임 리스트
    labels : list of str
        각 모델의 라벨 리스트
    ...
    """
    # 1. 논문 하드코딩 데이터 (Paper)
    paper_percentiles = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    # 논문의 Variance 값
    paper_var_raw = np.array([0.0127, 0.014, 0.0125, 0.0046, 0.0024, 0.002, 0.0023, 0.0006, 0.0007])
    
    # 스케일링: 10^-3 단위로 표현하기 위해 1000을 곱함 (0.016 -> 16.0)
    paper_var_scaled = paper_var_raw * 1000.0

    fig, ax = plt.subplots(figsize=figsize)

    # 2. Paper Data Plot (기본 비교군: 회색 점선)
    ax.plot(paper_percentiles, paper_var_scaled, 'o--', color='gray', label='MSDFM(Paper)',
            linewidth=2, alpha=0.6, markersize=6)
    
    # Paper 값 주석
    for x, y in zip(paper_percentiles, paper_var_scaled):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, -15), 
                    ha='center', fontsize=8, color='gray')

    # 3. 모델별 데이터 Plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dfs)))

    for i, (df, label) in enumerate(zip(results_dfs, labels)):
        stats = _compute_are_statistics(df, bin_range=(10, 90))
        
        base_df = pd.DataFrame({'Percentile': paper_percentiles})
        stats['Percentile'] = stats['Percentile'].astype(int)
        merged = pd.merge(base_df, stats, on='Percentile', how='left')
        
        # 모델 Variance 스케일링
        # df['ARE']는 % 단위이므로, Var_ARE는 %^2 단위임.
        # Ratio^2 단위로 변환(/ 10000) 후, 10^-3 스케일(* 1000) 적용 => / 10.0
        model_var_scaled = merged['Var_ARE'] / 10.0
        
        # Line Plot
        ax.plot(merged['Percentile'], model_var_scaled, 's-', label=label,
                color=colors[i], linewidth=2.5, markersize=8)

        # 값 주석 (모델 값)
        for x, y in zip(merged['Percentile'], model_var_scaled):
            if not np.isnan(y):
                ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 8), 
                            ha='center', fontsize=9, fontweight='bold', color=colors[i])

    # 축 및 스타일 설정
    ax.set_xlabel('Life Percentile (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Variance of ARE ($\times 10^{-3}$)', fontsize=12, fontweight='bold')
    ax.set_title('Variance of ARE Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(paper_percentiles)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # 저장
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'variance_are_comparison_multi.png')
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
    ID-SSM 모델에서 GAT attention weights 추출

    Parameters:
    -----------
    model : IDSSM
        학습된 ID-SSM 모델
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
def plot_combined_are_comparison(
    msdfm_results_df, 
    idssm_results_df, 
    save_dir=None,
    figsize=(14, 6), 
    show_plot=True
):
    """
    Mean ARE와 Variance ARE를 하나의 figure에 subplot으로 표시

    Parameters:
    -----------
    msdfm_results_df : pd.DataFrame
        MSDFM 모델 결과
    idssm_results_df : pd.DataFrame
        ID-SSM 모델 결과
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
    # ax1.bar(x - width/2, msdfm_stats['Mean_ARE'].values, width,
    #         label='MSDFM', color='#3498db', alpha=0.8, edgecolor='black')
    # ax1.bar(x + width/2, idssm_stats['Mean_ARE'].values, width,
    #         label='ID-SSM', color='#e74c3c', alpha=0.8, edgecolor='black')

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

    # ax2.bar(x - width/2, msdfm_var, width,
    #         label='MSDFM', color='#9b59b6', alpha=0.8, edgecolor='black')
    # ax2.bar(x + width/2, idssm_var, width,
    #         label='ID-SSM', color='#f39c12', alpha=0.8, edgecolor='black')

    ax2.plot(x - width/2, msdfm_var, 'o-', color='#8e44ad', linewidth=2)
    ax2.plot(x + width/2, idssm_var, 's-', color='#d68910', linewidth=2)

    ax2.set_xlabel('Life Percentile (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Variance of ARE (Ratio)', fontsize=11, fontweight='bold')
    ax2.set_title('Variance of ARE Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(msdfm_stats['Percentile'].astype(int).values)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

    plt.suptitle('MSDFM vs ID-SSM Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
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


def plot_psgs_ware(ranked_sensors, group_scores, sensor_scores, title_suffix="", 
                   save_dir=None, figsize=(14, 5), show_plot=True):
    """
    PSGS(Priority Sensor Group Selection) 알고리즘 결과 시각화 (논문 Fig 6 & 9 재현)
    (a) 우선순위별 개별 센서 WARE 점수
    (b) 센서 그룹 크기에 따른 WARE 점수 변화 및 최적 그룹

    Parameters:
    -----------
    ranked_sensors : list
        WARE 점수 순으로 정렬된 센서 이름 리스트
    group_scores : list or np.array
        그룹별(상위 n개 센서 조합) WARE 점수 리스트
    sensor_scores : dict
        센서별 개별 WARE 점수 딕셔너리 {sensor_name: score}
    title_suffix : str, optional
        그래프 제목에 붙일 접미사
    save_dir : str, optional
        저장 디렉토리 경로. None이면 저장하지 않음
    figsize : tuple
        그래프 크기
    show_plot : bool
        플롯 표시 여부
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # -------------------------------------------------------
    # (a) Individual Scores (Bar Chart)
    # -------------------------------------------------------
    ordered_scores = [sensor_scores[s] for s in ranked_sensors]
    x_indices = np.arange(1, len(ranked_sensors) + 1)
    
    bars = axes[0].bar(x_indices, ordered_scores, width=0.4, color='tab:blue', alpha=0.8)
    
    axes[0].set_xlabel('Prioritized Order', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('WARE Score (%)', fontsize=11, fontweight='bold')
    title_a = f'Individual Sensor WARE'
    if title_suffix: title_a += f' ({title_suffix})'
    axes[0].set_title(title_a, fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_indices)
    
    # 센서 번호 텍스트 표시 (예: s_2 -> 2)
    for bar, sensor_name in zip(bars, ranked_sensors):
        height = bar.get_height()
        # "s_2" 등에서 숫자만 추출
        s_nums = re.findall(r'\d+', sensor_name)
        s_num = s_nums[0] if s_nums else sensor_name
        
        axes[0].text(bar.get_x() + bar.get_width()/2., height, f'{s_num}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    # -------------------------------------------------------
    # (b) Group Scores (Line Chart)
    # -------------------------------------------------------
    x = np.arange(1, len(group_scores) + 1)
    axes[1].plot(x, group_scores, 'b.-', markersize=8, linewidth=1.5)
    
    # Optimal Point 표시
    min_idx = np.argmin(group_scores)
    min_val = group_scores[min_idx]
    
    axes[1].plot(min_idx + 1, min_val, 'ro', label='Optimal Sensor Group', markersize=8)
    axes[1].annotate(f'Optimal (n={min_idx+1})', 
                     xy=(min_idx+1, min_val), 
                     xytext=(min_idx+1, min_val * 1.15),
                     ha='center', fontsize=10, fontweight='bold', color='red',
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8))
    
    axes[1].set_xlabel('Number of Sensors in Group', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('WARE Score (%)', fontsize=11, fontweight='bold')
    title_b = f'Group WARE Scores'
    if title_suffix: title_b += f' ({title_suffix})'
    axes[1].set_title(title_b, fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    
    # -------------------------------------------------------
    # Save & Show Logic (프로젝트 표준 준수)
    # -------------------------------------------------------
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 파일명 생성 로직
        clean_suffix = re.sub(r'\W+', '_', title_suffix).strip('_')
        filename = "psgs_ware_analysis"
        if clean_suffix:
            filename += f"_{clean_suffix}"
        filename += ".png"
        
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] PSGS WARE plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, axes
