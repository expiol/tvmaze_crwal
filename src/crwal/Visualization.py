from __future__ import annotations

import json
import logging
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .util import read_csv, ensure_dir, setup_logging


def configure_matplotlib() -> None:

    try:
        fonts = ['SimHei', 'Heiti TC', 'WenQuanYi Micro Hei', 'STHeiti', 'Arial Unicode MS']
        for font in fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                break
            except:
                continue
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 使用现代化样式
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass
    
    # 专业配色方案（参考Nature期刊）
    plt.rcParams.update({
        # 图形尺寸和DPI
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'figure.facecolor': 'white',
        
        # 坐标轴样式
        'axes.facecolor': '#f8f9fa',
        'axes.edgecolor': '#dee2e6',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # 网格样式
        'grid.color': '#e9ecef',
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        
        # 字体样式
        'font.size': 11,
        'font.family': 'sans-serif',
        
        # 刻度样式
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.color': '#495057',
        'ytick.color': '#495057',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # 图例样式
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': True,
        'legend.fontsize': 10,
        'legend.edgecolor': '#dee2e6',
        
        # 线条样式
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        
        # 保存选项
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.facecolor': 'white',
    })


# 专业配色方案
COLORS = {
    'primary': '#1f77b4',      # 蓝色
    'success': '#2ecc71',      # 绿色
    'warning': '#f39c12',      # 橙色
    'danger': '#e74c3c',       # 红色
    'info': '#3498db',         # 浅蓝
    'purple': '#9b59b6',       # 紫色
    'teal': '#1abc9c',         # 青色
    'navy': '#34495e',         # 深蓝
    'gradient': ['#667eea', '#764ba2', '#f093fb', '#4facfe'],  # 渐变色
}


configure_matplotlib()


def parse_genres_cell(cell: str) -> List[str]:
    if not isinstance(cell, str) or not cell:
        return []
    try:
        data = json.loads(cell)
        return [str(x) for x in data] if isinstance(data, list) else []
    except:
        return [x.strip() for x in cell.split(",") if x.strip()]


def add_year_columns(df: pd.DataFrame) -> pd.DataFrame:
    def to_year(s: str) -> float:
        if not isinstance(s, str) or not s:
            return np.nan
        try:
            return float(s[:4])
        except:
            return np.nan
    
    df = df.copy()
    df["Year"] = df["First air date"].apply(to_year)
    return df


def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Genres_list"] = df["Genres"].apply(parse_genres_cell)
    df_exploded = df.explode("Genres_list").rename(columns={"Genres_list": "Genre"})
    return df_exploded.dropna(subset=["Genre"])


def fig_top_rated(df: pd.DataFrame, k: int, save_path: str) -> None:
    """Top K评分最高节目（专业横向条形图）"""
    data = df.dropna(subset=["Rating"]).sort_values("Rating", ascending=False).head(k)
    if len(data) == 0:
        return
    
    fig, ax = plt.figure(figsize=(12, max(7, int(k * 0.4)))), plt.gca()
    
    # 使用渐变色（高分到低分）
    colors = plt.cm.RdYlGn(np.linspace(0.5, 0.95, len(data)))
    
    bars = ax.barh(range(len(data)), data["Rating"].values[::-1], 
                   color=colors[::-1], edgecolor='white', linewidth=1.5, alpha=0.9)
    
    # 设置Y轴标签（节目名称）
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["Title"].values[::-1], fontsize=10)
    
    ax.set_xlabel("Rating Score", fontsize=12, fontweight='bold')
    ax.set_title(f"Top {k} Highest Rated TV Shows", fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 10.5)
    
    # 添加评分数值标签
    for i, (bar, rating) in enumerate(zip(bars, data["Rating"].values[::-1])):
        ax.text(rating + 0.15, i, f'{rating:.1f}', 
                va='center', ha='left', fontsize=10, fontweight='bold', color='#2c3e50')
    
    # 添加参考线
    ax.axvline(data["Rating"].mean(), color='#e74c3c', linestyle=':', linewidth=2, 
               alpha=0.6, label=f'Average: {data["Rating"].mean():.2f}')
    ax.legend(loc='lower right', framealpha=0.95)
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def fig_rating_hist(df: pd.DataFrame, bins: int, save_path: str) -> None:
    """评分分布直方图（专业统计图表）"""
    vals = df["Rating"].dropna().values
    if len(vals) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # 直方图
    n, bins_edges, patches = ax.hist(vals, bins=bins, color=COLORS['primary'], 
                                       edgecolor='white', linewidth=1.5, alpha=0.85)
    
    # 为每个柱子设置渐变色
    cm = plt.cm.Blues
    norm = plt.Normalize(vmin=bins_edges.min(), vmax=bins_edges.max())
    for i, (patch, bin_edge) in enumerate(zip(patches, bins_edges[:-1])):
        patch.set_facecolor(cm(norm(bin_edge + (bins_edges[1]-bins_edges[0])/2)))
    
    ax.set_xlabel("Rating Score", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency (Number of Shows)", fontsize=12, fontweight='bold')
    ax.set_title("TV Show Rating Distribution", fontsize=15, fontweight='bold', pad=20)
    
    # 统计线
    mean_val, median_val = vals.mean(), np.median(vals)
    std_val = vals.std()
    
    ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2.5, 
               label=f'Mean: {mean_val:.2f}', alpha=0.8)
    ax.axvline(median_val, color='#f39c12', linestyle='-.', linewidth=2.5, 
               label=f'Median: {median_val:.2f}', alpha=0.8)
    
    # 添加文本注释
    ax.text(0.02, 0.98, f'n = {len(vals)}\nμ = {mean_val:.2f}\nσ = {std_val:.2f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#dee2e6'))
    
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def fig_status_mean_rating(df: pd.DataFrame, save_path: str) -> None:
    """播出状态vs平均评分（专业柱状图）"""
    status_data = df.groupby("Status", dropna=False).agg({
        'Rating': ['mean', 'count']
    }).droplevel(0, axis=1)
    status_data.columns = ['mean', 'count']
    status_data = status_data.sort_values('mean', ascending=False)
    
    if len(status_data) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # 使用现代配色
    color_map = {'Running': COLORS['success'], 'Ended': COLORS['info'], 
                 'In Development': COLORS['warning'], 'To Be Determined': COLORS['purple']}
    colors = [color_map.get(status, COLORS['navy']) for status in status_data.index]
    
    bars = ax.bar(range(len(status_data)), status_data['mean'].values, 
                  color=colors, edgecolor='white', linewidth=2, alpha=0.9, width=0.7)
    
    ax.set_xticks(range(len(status_data)))
    ax.set_xticklabels(status_data.index.astype(str), rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("Average Rating Score", fontsize=12, fontweight='bold')
    ax.set_title("Average Rating by Show Status", fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 10.5)
    
    # 添加数值标签和样本量
    for i, (bar, (idx, row)) in enumerate(zip(bars, status_data.iterrows())):
        height = row['mean']
        count = int(row['count'])
        if not np.isnan(height):
            ax.text(i, height + 0.2, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')
            ax.text(i, 0.3, f'n={count}',
                    ha='center', va='bottom', fontsize=9, color='#7f8c8d', style='italic')
    
    # 添加全局平均线
    global_mean = df['Rating'].mean()
    ax.axhline(global_mean, color='#e74c3c', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Overall Mean: {global_mean:.2f}')
    ax.legend(loc='upper right', framealpha=0.95)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def fig_genre_mean_rating(df: pd.DataFrame, topn: int, save_path: str) -> None:
    """类型平均评分（专业条形图with渐变色）"""
    dfg = explode_genres(df)
    genre_data = dfg.groupby("Genre").agg({
        'Rating': ['mean', 'count']
    }).droplevel(0, axis=1)
    genre_data.columns = ['mean', 'count']
    genre_data = genre_data.dropna().sort_values('mean', ascending=False).head(topn)
    
    if len(genre_data) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 使用渐变色（从红到绿）
    colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(genre_data)))
    
    bars = ax.bar(range(len(genre_data)), genre_data['mean'].values, 
                  color=colors, edgecolor='white', linewidth=2, alpha=0.9, width=0.75)
    
    ax.set_xticks(range(len(genre_data)))
    ax.set_xticklabels(genre_data.index.astype(str), rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Average Rating Score", fontsize=12, fontweight='bold')
    ax.set_title(f"Top {topn} Genres by Average Rating", fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 10.5)
    
    # 添加评分和样本量标签
    for i, (bar, (idx, row)) in enumerate(zip(bars, genre_data.iterrows())):
        height = row['mean']
        count = int(row['count'])
        ax.text(i, height + 0.15, f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')
        ax.text(i, 0.2, f'({count})',
                ha='center', va='bottom', fontsize=8, color='#95a5a6', style='italic')
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def fig_year_counts(df: pd.DataFrame, save_path: str) -> None:
    """节目首播年度趋势（专业时间序列图）"""
    dfy = add_year_columns(df)
    agg = dfy.groupby("Year").size().dropna().sort_index()
    if len(agg) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    # 主线条和区域填充
    ax.plot(agg.index.values, agg.values, marker="o", linewidth=3, 
            markersize=8, color=COLORS['primary'], markerfacecolor='white', 
            markeredgewidth=2.5, markeredgecolor=COLORS['primary'], 
            label='Show Count', alpha=0.9, zorder=3)
    ax.fill_between(agg.index.values, agg.values, alpha=0.2, color=COLORS['primary'])
    
    ax.set_xlabel("Premiere Year", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Shows", fontsize=12, fontweight='bold')
    ax.set_title("TV Show Premiere Trends Over Time", fontsize=15, fontweight='bold', pad=20)
    
    # 标注峰值
    max_year, max_count = agg.idxmax(), agg.max()
    ax.annotate(f'Peak Year\n{int(max_count)} shows in {int(max_year)}',
                xy=(max_year, max_count), xytext=(30, 20),
                textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', fc='#fff3cd', alpha=0.9, edgecolor='#ffc107'),
                arrowprops=dict(arrowstyle='->', lw=2, color='#f39c12'))
    
    # 添加趋势线
    z = np.polyfit(agg.index.values, agg.values, 2)
    p = np.poly1d(z)
    ax.plot(agg.index.values, p(agg.index.values), "--", 
            color=COLORS['danger'], alpha=0.6, linewidth=2, label='Trend (polynomial)')
    
    ax.legend(loc='best', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def fig_network_topn(df: pd.DataFrame, topn: int, save_path: str) -> None:
    """网络频道节目数量（专业渐变柱状图）"""
    cnt = df["Network"].fillna("").replace("", "Unknown").value_counts().head(topn)
    if len(cnt) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(13, 7.5))
    
    # 使用Tableau颜色方案
    colors = plt.cm.tab20c(np.linspace(0, 1, len(cnt)))
    
    bars = ax.bar(range(len(cnt)), cnt.values, color=colors,
                  edgecolor='white', linewidth=2, alpha=0.9, width=0.75)
    
    ax.set_xticks(range(len(cnt)))
    ax.set_xticklabels(cnt.index.astype(str), rotation=40, ha="right", fontsize=11)
    ax.set_ylabel("Number of Shows", fontsize=12, fontweight='bold')
    ax.set_title(f"Top {topn} Networks/Channels by Show Count", fontsize=15, fontweight='bold', pad=20)
    
    # 添加数值标签和百分比
    total = cnt.sum()
    for i, (bar, count) in enumerate(zip(bars, cnt.values)):
        percentage = (count / total) * 100
        ax.text(i, count + max(cnt.values)*0.01, f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')
        ax.text(i, count * 0.5, f'{percentage:.1f}%',
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # 添加平均线
    avg = cnt.mean()
    ax.axhline(avg, color='#e74c3c', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Average: {avg:.1f}')
    ax.legend(loc='upper right', framealpha=0.95)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def fig_genre_boxplot(df: pd.DataFrame, topk_genres: int, save_path: str) -> None:
    """类型评分分布箱线图（专业统计可视化）"""
    dfg = explode_genres(df)
    if len(dfg) == 0:
        return
    
    top_genres = dfg["Genre"].value_counts().head(topk_genres).index.tolist()
    data = [dfg.loc[dfg["Genre"] == g, "Rating"].dropna().values for g in top_genres]
    valid_data = [(g, d) for g, d in zip(top_genres, data) if len(d) > 0]
    
    if not valid_data:
        return
    
    top_genres, data = zip(*valid_data)
    
    fig, ax = plt.subplots(figsize=(13, 8))
    
    # 创建箱线图
    bp = ax.boxplot(data, labels=top_genres, patch_artist=True, showmeans=True,
                    boxprops=dict(facecolor=COLORS['info'], alpha=0.7, linewidth=2, edgecolor='#2c3e50'),
                    medianprops=dict(color='#e74c3c', linewidth=3),
                    meanprops=dict(marker='D', markerfacecolor=COLORS['success'], 
                                  markeredgecolor='white', markersize=8, markeredgewidth=1.5),
                    whiskerprops=dict(linewidth=2, color='#34495e'),
                    capprops=dict(linewidth=2, color='#34495e'),
                    flierprops=dict(marker='o', markerfacecolor=COLORS['warning'], 
                                   markersize=6, alpha=0.6, markeredgecolor='none'))
    
    # 为每个箱子设置不同颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_genres)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax.set_ylabel("Rating Score", fontsize=12, fontweight='bold')
    ax.set_title(f"Rating Distribution by Genre (Top {topk_genres} Most Popular)", 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticklabels(top_genres, rotation=25, ha="right", fontsize=11)
    ax.set_ylim(0, 10.5)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.5, label='IQR (25%-75%)'),
        plt.Line2D([0], [0], color='#e74c3c', linewidth=3, label='Median'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['success'], 
                  markersize=8, label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def run(csv_path: str, save_dir: str, topk: int = 20) -> None:
    logging.info("=" * 60)
    logging.info("TVMaze Data Visualization Started")
    logging.info(f"Input: {csv_path}, Output: {save_dir}")
    
    ensure_dir(save_dir)
    df = read_csv(csv_path)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    
    logging.info(f"Dataset: {len(df)} rows, Avg rating: {df['Rating'].mean():.2f}")
    
    fig_top_rated(df, k=topk, save_path=os.path.join(save_dir, f"top_{topk}_rated.png"))
    fig_rating_hist(df, bins=20, save_path=os.path.join(save_dir, "rating_hist.png"))
    fig_status_mean_rating(df, save_path=os.path.join(save_dir, "status_mean_rating.png"))
    fig_genre_mean_rating(df, topn=10, save_path=os.path.join(save_dir, "genre_mean_top10.png"))
    fig_year_counts(df, save_path=os.path.join(save_dir, "year_counts.png"))
    fig_network_topn(df, topn=15, save_path=os.path.join(save_dir, "network_top15.png"))
    fig_genre_boxplot(df, topk_genres=6, save_path=os.path.join(save_dir, "genre_boxplot_top6.png"))
    
    logging.info(f"SUCCESS: All visualizations saved to {save_dir}")
    logging.info("=" * 60)
