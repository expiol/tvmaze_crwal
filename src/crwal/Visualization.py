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
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass
    
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f5f5f5'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['grid.linewidth'] = 1.2
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 11


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
    data = df.dropna(subset=["Rating"]).sort_values("Rating", ascending=False).head(k)
    if len(data) == 0:
        return
    
    plt.figure(figsize=(10, max(6, int(k * 0.4))))
    plt.barh(data["Title"][::-1], data["Rating"][::-1], 
             color='#4CAF50', edgecolor='#2E7D32', linewidth=0.5)
    plt.xlabel("Rating", fontweight='bold')
    plt.title(f"Top {k} Shows by Rating", fontweight='bold', pad=15)
    plt.xlim(0, 10)
    
    for i, (_, row) in enumerate(data[::-1].iterrows()):
        plt.text(row["Rating"] + 0.1, i, f'{row["Rating"]:.1f}', 
                va='center', fontsize=9, color='#333')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_rating_hist(df: pd.DataFrame, bins: int, save_path: str) -> None:
    vals = df["Rating"].dropna().values
    if len(vals) == 0:
        return
    
    plt.figure(figsize=(9, 6))
    plt.hist(vals, bins=bins, color='#2196F3', edgecolor='#1565C0', 
             linewidth=0.5, alpha=0.85)
    plt.xlabel("Rating", fontweight='bold')
    plt.ylabel("Number of Shows", fontweight='bold')
    plt.title("Rating Distribution", fontweight='bold', pad=15)
    
    mean_val, median_val = vals.mean(), np.median(vals)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {median_val:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_status_mean_rating(df: pd.DataFrame, save_path: str) -> None:
    agg = df.groupby("Status", dropna=False)["Rating"].mean().sort_values(ascending=False)
    if len(agg) == 0:
        return
    
    plt.figure(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(agg)))
    bars = plt.bar(agg.index.astype(str), agg.values, color=colors, 
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    plt.ylabel("Average Rating", fontweight='bold')
    plt.title("Average Rating by Show Status", fontweight='bold', pad=15)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 10)
    
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_genre_mean_rating(df: pd.DataFrame, topn: int, save_path: str) -> None:
    dfg = explode_genres(df)
    agg = dfg.groupby("Genre")["Rating"].mean().dropna().sort_values(ascending=False).head(topn)
    if len(agg) == 0:
        return
    
    plt.figure(figsize=(11, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(agg)))
    bars = plt.bar(agg.index.astype(str), agg.values, color=colors,
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    plt.ylabel("Average Rating", fontweight='bold')
    plt.title(f"Top {topn} Genres by Average Rating", fontweight='bold', pad=15)
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 10)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_year_counts(df: pd.DataFrame, save_path: str) -> None:
    dfy = add_year_columns(df)
    agg = dfy.groupby("Year").size().dropna()
    if len(agg) == 0:
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(agg.index.values, agg.values, marker="o", linewidth=2.5, 
             markersize=6, color='#FF5722', markerfacecolor='#FFC107', 
             markeredgewidth=1.5, markeredgecolor='#E64A19')
    plt.xlabel("Year (First Air Date)", fontweight='bold')
    plt.ylabel("Number of Shows", fontweight='bold')
    plt.title("TV Show Premiere Trends Over Time", fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    if len(agg) > 0:
        max_year, max_count = agg.idxmax(), agg.max()
        plt.annotate(f'Peak: {int(max_count)} shows\nin {int(max_year)}',
                    xy=(max_year, max_count), xytext=(10, 10),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_network_topn(df: pd.DataFrame, topn: int, save_path: str) -> None:
    cnt = df["Network"].fillna("").replace("", "Unknown").value_counts().head(topn)
    if len(cnt) == 0:
        return
    
    plt.figure(figsize=(11, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(cnt)))
    bars = plt.bar(cnt.index.astype(str), cnt.values, color=colors,
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    plt.ylabel("Number of Shows", fontweight='bold')
    plt.title(f"Top {topn} Networks by Show Count", fontweight='bold', pad=15)
    plt.xticks(rotation=35, ha="right")
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def fig_genre_boxplot(df: pd.DataFrame, topk_genres: int, save_path: str) -> None:
    dfg = explode_genres(df)
    if len(dfg) == 0:
        return
    
    top_genres = dfg["Genre"].value_counts().head(topk_genres).index.tolist()
    data = [dfg.loc[dfg["Genre"] == g, "Rating"].dropna().values for g in top_genres]
    valid_data = [(g, d) for g, d in zip(top_genres, data) if len(d) > 0]
    
    if not valid_data:
        return
    
    top_genres, data = zip(*valid_data)
    
    plt.figure(figsize=(12, 7))
    plt.boxplot(data, labels=top_genres, showmeans=True, meanline=True,
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=1.5),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(color='green', linewidth=2, linestyle='--'),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))
    plt.ylabel("Rating", fontweight='bold')
    plt.title(f"Rating Distribution by Genre (Top {topk_genres} Most Common)", 
             fontweight='bold', pad=15)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 10)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
