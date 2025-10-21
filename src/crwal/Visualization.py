# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) & Visualization for the scraped TVMaze dataset.

Usage:
    python -m src.crwal.Visualization --csv data/your_name+id.csv --save-dir data/figures --topk 20
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 兼容包/脚本两种运行方式
try:
    from .util import read_csv, ensure_dir, setup_logging
except ImportError:  # pragma: no cover
    from util import read_csv, ensure_dir, setup_logging


# -----------------------------
# Helpers
# -----------------------------
def parse_genres_cell(cell: str) -> List[str]:
    """
    Genres 列是 JSON 字符串，解析失败时做保守兼容。
    """
    if not isinstance(cell, str) or not cell:
        return []
    try:
        data = json.loads(cell)
        if isinstance(data, list):
            return [str(x) for x in data]
        return []
    except Exception:
        # 兜底：逗号或分号分割
        return [x.strip() for x in cell.split(",") if x.strip()]


def add_year_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 'First air date' 提取年份，便于按年统计。
    日期格式为 YYYY/MM/DD 或空字符串。
    """
    def to_year(s: str) -> float:
        if not isinstance(s, str) or not s:
            return np.nan
        # 前四位为年份
        try:
            return float(s[:4])
        except Exception:
            return np.nan

    df = df.copy()
    df["Year"] = df["First air date"].apply(to_year)
    return df


# -----------------------------
# Plot functions (≥5 figures)
# -----------------------------
def fig_top_rated(df: pd.DataFrame, k: int, save_path: str | None = None) -> None:
    data = df.dropna(subset=["Rating"]).sort_values("Rating", ascending=False).head(k)
    plt.figure(figsize=(10, max(6, int(k * 0.5))))
    plt.barh(y=data["Title"][::-1], width=data["Rating"][::-1])
    plt.xlabel("Rating")
    plt.title(f"Top-{k} Shows by Rating")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def fig_rating_hist(df: pd.DataFrame, bins: int, save_path: str | None = None) -> None:
    vals = df["Rating"].dropna().values
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=bins)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title("Rating Distribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def fig_status_mean_rating(df: pd.DataFrame, save_path: str | None = None) -> None:
    agg = df.groupby("Status", dropna=False)["Rating"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(agg.index.astype(str), agg.values)
    plt.ylabel("Average Rating")
    plt.title("Average Rating by Status")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Genres_list"] = df["Genres"].apply(parse_genres_cell)
    df_exploded = df.explode("Genres_list")
    df_exploded = df_exploded.rename(columns={"Genres_list": "Genre"})
    df_exploded = df_exploded.dropna(subset=["Genre"])
    return df_exploded

def fig_genre_mean_rating(df: pd.DataFrame, topn: int, save_path: str | None = None) -> None:
    dfg = explode_genres(df)
    agg = dfg.groupby("Genre")["Rating"].mean().dropna().sort_values(ascending=False).head(topn)
    plt.figure(figsize=(10, 5))
    plt.bar(agg.index.astype(str), agg.values)
    plt.ylabel("Average Rating")
    plt.title(f"Top-{topn} Genres by Average Rating")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def fig_year_counts(df: pd.DataFrame, save_path: str | None = None) -> None:
    dfy = add_year_columns(df)
    agg = dfy.groupby("Year").size().dropna()
    plt.figure(figsize=(9, 5))
    plt.plot(agg.index.values, agg.values, marker="o")
    plt.xlabel("Year (First Air Date)")
    plt.ylabel("#Shows")
    plt.title("Shows Count by Year")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def fig_network_topn(df: pd.DataFrame, topn: int, save_path: str | None = None) -> None:
    cnt = df["Network"].fillna("").replace("", "Unknown").value_counts().head(topn)
    plt.figure(figsize=(10, 5))
    plt.bar(cnt.index.astype(str), cnt.values)
    plt.ylabel("Count")
    plt.title(f"Top-{topn} Networks by Appearance")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def fig_genre_boxplot(df: pd.DataFrame, topk_genres: int, save_path: str | None = None) -> None:
    """
    选择出现次数最多的前 K 个体裁，绘制评分箱线图，展示分布差异。
    """
    dfg = explode_genres(df)
    top_genres = dfg["Genre"].value_counts().head(topk_genres).index.tolist()
    data = [dfg.loc[dfg["Genre"] == g, "Rating"].dropna().values for g in top_genres]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=top_genres, showmeans=True)
    plt.ylabel("Rating")
    plt.title(f"Rating Distribution by Genre (Top {topk_genres})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()



def run(csv_path: str, save_dir: str, topk: int = 20) -> None:
    setup_logging()
    ensure_dir(save_dir)
    df = read_csv(csv_path)

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    fig_top_rated(df, k=topk, save_path=os.path.join(save_dir, f"top_{topk}_rated.png"))
    fig_rating_hist(df, bins=20, save_path=os.path.join(save_dir, "rating_hist.png"))
    fig_status_mean_rating(df, save_path=os.path.join(save_dir, "status_mean_rating.png"))

    # 4. 体裁均分 Top-10
    fig_genre_mean_rating(df, topn=10, save_path=os.path.join(save_dir, "genre_mean_top10.png"))

    # 5. 年度开播趋势
    fig_year_counts(df, save_path=os.path.join(save_dir, "year_counts.png"))

    # 6. Network Top-15
    fig_network_topn(df, topn=15, save_path=os.path.join(save_dir, "network_top15.png"))

    # 7. 主流体裁的评分箱线图（可选但推荐，用于提升“≥5图”的质量）
    fig_genre_boxplot(df, topk_genres=6, save_path=os.path.join(save_dir, "genre_boxplot_top6.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA & Visualization for TVMaze dataset")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV (your_name+id.csv)")
    parser.add_argument("--save-dir", type=str, default="data/figures", help="Directory to save figures")
    parser.add_argument("--topk", type=int, default=20, help="Top-K shows for ranking plot")
    args = parser.parse_args()

    run(args.csv, args.save_dir, args.topk)


if __name__ == "__main__":
    main()
