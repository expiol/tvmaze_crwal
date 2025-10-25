from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .util import read_csv, ensure_dir, setup_logging


# ========= Nature（NPG）有色配色 =========
# 参考常见的 Nature Publishing Group（NPG）调色板（广泛用于学术可视化）
NPG_CYCLE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"
]
# 语义色（按需挑选）
NPG = {
    "primary":  "#3C5488",  # 深蓝
    "secondary":"#E64B35",  # 红
    "success":  "#00A087",  # 绿
    "warning":  "#F39B7F",  # 橙
    "info":     "#4DBBD5",  # 青
    "purple":   "#8491B4",
    "teal":     "#91D1C2",
    "scarlet":  "#DC0000",
    "brown":    "#7E6148",
    "beige":    "#B09C85",
}


# ========= 期刊风格 Matplotlib（不依赖 seaborn） =========
def configure_matplotlib() -> None:
    """
    期刊风格（Nature 配色）：
    - 统一字体/线宽/刻度风格
    - 使用 NPG_CYCLE 作为默认颜色循环
    - 300dpi 导出，便于报告直接使用
    """
    try:
        # 中文字体回退
        fonts = ['SimHei', 'Heiti TC', 'WenQuanYi Micro Hei', 'STHeiti', 'Arial Unicode MS', 'Arial', 'DejaVu Sans']
        for font in fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams.get('font.sans-serif', [])
                break
            except Exception:
                continue
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    plt.rcParams.update({
        # 图形与导出
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,

        # 字体与字号
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',

        # 坐标轴与网格
        'axes.facecolor': 'white',
        'axes.edgecolor': '#222222',
        'axes.linewidth': 1.0,
        'axes.grid': False,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # 刻度
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,

        # 线条与误差线
        'lines.linewidth': 1.7,
        'lines.markersize': 5.5,

        # 图例
        'legend.frameon': False,

        # 矢量友好
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    # 设置默认颜色循环为 NPG 调色板
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=NPG_CYCLE)


def _set_axes(ax: plt.Axes, xlabel: str = None, ylabel: str = None, title: str = None) -> None:
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='out', length=4, width=1, pad=2)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1)
        ax.spines[spine].set_color('#222222')
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=10)


def _footnote(ax: plt.Axes, text: str) -> None:
    ax.figure.text(0.01, 0.01, text, ha='left', va='bottom', fontsize=8, color="#6e6e6e")


# ========= 数据处理 =========
def parse_genres_cell(cell: str) -> List[str]:
    if not isinstance(cell, str) or not cell:
        return []
    try:
        data = json.loads(cell)
        return [str(x) for x in data] if isinstance(data, list) else []
    except Exception:
        return [x.strip() for x in cell.split(",") if x.strip()]


def add_year_columns(df: pd.DataFrame) -> pd.DataFrame:
    def to_year(s: str) -> float:
        if not isinstance(s, str) or not s:
            return np.nan
        try:
            y = int(s[:4])
            if 1900 <= y <= 2100:
                return float(y)
        except Exception:
            return np.nan
        return np.nan

    df = df.copy()
    # 注意：仅从 First air date 推出 Year，避免把 End date (含占位9999) 误用
    if "First air date" in df.columns:
        df["Year"] = df["First air date"].apply(to_year)
    else:
        df["Year"] = np.nan
    return df


def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Genres" not in df.columns:
        df["Genres"] = ""
    df["Genres_list"] = df["Genres"].apply(parse_genres_cell)
    df_exploded = df.explode("Genres_list").rename(columns={"Genres_list": "Genre"})
    return df_exploded.dropna(subset=["Genre"])


# ========= 统计工具 =========
def _auto_bins(vals: np.ndarray, max_bins: int = 50) -> int:
    if len(vals) < 2:
        return 5
    q75, q25 = np.percentile(vals, [75, 25])
    iqr = max(q75 - q25, 1e-9)
    bin_width = 2 * iqr * (len(vals) ** (-1/3))
    if bin_width <= 0:
        return min(20, len(np.unique(vals)))
    bins = int(np.ceil((vals.max() - vals.min()) / bin_width))
    return max(5, min(bins, max_bins))


def _bootstrap_ci_mean(vals: np.ndarray, n_boot: int = 4000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(vals)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = vals[idx].mean()
    mean = float(vals.mean())
    low = float(np.quantile(boots, alpha/2))
    high = float(np.quantile(boots, 1 - alpha/2))
    return (mean, low, high)


def _permutation_test_diff_means(x: np.ndarray, y: np.ndarray, n_perm: int = 10000, seed: int = 123) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    obs = abs(x.mean() - y.mean())
    rng = np.random.default_rng(seed)
    combined = np.concatenate([x, y])
    n_x = len(x)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        diff = abs(combined[:n_x].mean() - combined[n_x:].mean())
        if diff >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)  # add-one smoothing


# ========= 平台识别（Streaming vs TV） =========
STREAMING_KEYWORDS = [
    "Netflix", "Amazon", "Prime", "Hulu", "Disney", "Apple TV", "AppleTV", "Apple+",
    "HBO Max", "Max", "Paramount", "Paramount+", "Peacock", "Discovery", "Discovery+",
    "AMC+", "Showtime", "Starz", "BBC iPlayer", "ITVX", "Crave", "Stan",
    "Hotstar", "Jiocinema", "Viu", "iQIYI", "Tencent", "Youku"
]


def _platform_from_network(network: str) -> str:
    if not isinstance(network, str) or not network.strip():
        return "Unknown"
    s = network.lower()
    for kw in STREAMING_KEYWORDS:
        if kw.lower() in s:
            return "Streaming"
    return "TV"


# ========= 可视化图表（Nature配色） =========
def fig_top_rated(df: pd.DataFrame, k: int, save_path: str) -> None:
    """Top K 评分最高（横向条形图，NPG色循环）"""
    if "Rating" not in df.columns or "Title" not in df.columns:
        return
    data = df.dropna(subset=["Rating"]).sort_values("Rating", ascending=False).head(k)
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(10.5, max(5.5, int(k * 0.42))))
    titles = data["Title"].astype(str).values[::-1]
    ratings = data["Rating"].astype(float).values[::-1]
    y = np.arange(len(titles))

    colors = [NPG_CYCLE[i % len(NPG_CYCLE)] for i in range(len(titles))]
    ax.barh(y, ratings, color=colors[::-1], edgecolor="#222", linewidth=0.8, height=0.8)
    ax.set_yticks(y, labels=[f"{i+1}. {t}" for i, t in enumerate(titles)])
    ax.set_xlim(0, 10.5)

    for i, r in enumerate(ratings):
        ax.text(min(r + 0.12, 10.35), i, f"{r:.1f}", va='center', ha='left', fontsize=9, color="#111")

    _set_axes(ax, xlabel="Rating", title=f"Top {k} Highest Rated TV Shows")
    _footnote(ax, f"仅包含有评分的条目；首页截断样本可能抬高热门在榜比例。n={len(data)}")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def fig_rating_hist(df: pd.DataFrame, bins: Optional[int], save_path: str) -> None:
    """评分分布直方图（NPG主色 + 均值/中位数）"""
    if "Rating" not in df.columns:
        return
    vals = df["Rating"].dropna().values
    if len(vals) == 0:
        return

    bins_used = bins if isinstance(bins, int) and bins > 0 else _auto_bins(vals)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    ax.hist(vals, bins=bins_used, color=NPG["teal"], edgecolor="#222", linewidth=0.8)
    mean_val, median_val, std_val = vals.mean(), np.median(vals), vals.std()

    ax.axvline(mean_val, color=NPG["primary"], linestyle='--', linewidth=1.8, label=f"Mean {mean_val:.2f}")
    ax.axvline(median_val, color=NPG["secondary"], linestyle='-.', linewidth=1.6, label=f"Median {median_val:.2f}")
    ax.legend(loc='upper left', fontsize=9)

    _set_axes(ax, xlabel="Rating", ylabel="Count", title="Distribution of Ratings")
    _footnote(ax, f"n={len(vals)}；μ={mean_val:.2f}；σ={std_val:.2f}")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def fig_status_mean_rating(df: pd.DataFrame, save_path: str, min_group_n: int = 3) -> None:
    """播出状态 vs 平均评分（NPG色 + 95%CI + 样本量）"""
    if "Status" not in df.columns or "Rating" not in df.columns:
        return
    g = df.groupby("Status", dropna=False)["Rating"]
    rows = []
    for name, s in g:
        arr = s.dropna().values
        if len(arr) < min_group_n:
            continue
        mean, low, high = _bootstrap_ci_mean(arr)
        rows.append((str(name) if name is not None else "Unknown", mean, low, high, len(arr)))
    if not rows:
        return
    tbl = pd.DataFrame(rows, columns=["Status", "Mean", "Low", "High", "n"]).sort_values("Mean", ascending=False)

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    x = np.arange(len(tbl))
    bar_colors = [NPG_CYCLE[i % len(NPG_CYCLE)] for i in range(len(tbl))]

    ax.bar(x, tbl["Mean"].values, color=bar_colors, edgecolor="#222", linewidth=0.8, width=0.75)
    yerr = np.vstack([tbl["Mean"].values - tbl["Low"].values, tbl["High"].values - tbl["Mean"].values])
    ax.errorbar(x, tbl["Mean"].values, yerr=yerr, fmt='none', ecolor="#222", elinewidth=1.2, capsize=3)

    ax.set_xticks(x, labels=tbl["Status"].astype(str).values, rotation=15, ha='right')
    _set_axes(ax, ylabel="Average Rating", title="Average Rating by Status (95% CI)")

    overall = float(df["Rating"].mean())
    ax.axhline(overall, color="#666", linestyle='--', linewidth=1.2, label=f"Overall mean {overall:.2f}")
    ax.legend(loc='lower right', fontsize=9)

    for i, (m, n) in enumerate(zip(tbl["Mean"].values, tbl["n"].values)):
        ax.text(i, m + 0.15, f"n={n}", ha='center', va='bottom', fontsize=8, color="#111")

    # Running vs Ended 置换检验（若同时存在）
    run_vals = df[df["Status"].fillna("").str.contains("Running", case=False, na=False)]["Rating"].dropna().values
    end_vals = df[df["Status"].fillna("").str.contains("Ended", case=False, na=False)]["Rating"].dropna().values
    if len(run_vals) > 0 and len(end_vals) > 0:
        p = _permutation_test_diff_means(run_vals, end_vals)
        _footnote(ax, f"仅显示样本量≥{min_group_n}的状态；Permutation p (Running vs Ended) = {p:.4f}")
    else:
        _footnote(ax, f"仅显示样本量≥{min_group_n}的状态。")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def fig_genre_mean_rating(df: pd.DataFrame, topn: int, save_path: str, min_count: int = 5) -> None:
    """类型平均评分（Top-N + 95%CI，NPG色；过滤稀疏类型）"""
    if "Genres" not in df.columns or "Rating" not in df.columns:
        return
    dfg = explode_genres(df)
    if dfg.empty:
        return

    rows = []
    for gname, s in dfg.groupby("Genre")["Rating"]:
        vals = s.dropna().values
        if len(vals) < min_count:
            continue
        mean, low, high = _bootstrap_ci_mean(vals)
        rows.append((gname, mean, low, high, len(vals)))
    if not rows:
        return
    tbl = pd.DataFrame(rows, columns=["Genre", "Mean", "Low", "High", "n"]).sort_values(["Mean", "n"], ascending=[False, False]).head(topn)

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    x = np.arange(len(tbl))
    bar_colors = [NPG_CYCLE[i % len(NPG_CYCLE)] for i in range(len(tbl))]

    ax.bar(x, tbl["Mean"].values, color=bar_colors, edgecolor="#222", linewidth=0.8, width=0.72)
    yerr = np.vstack([tbl["Mean"].values - tbl["Low"].values, tbl["High"].values - tbl["Mean"].values])
    ax.errorbar(x, tbl["Mean"].values, yerr=yerr, fmt='none', ecolor="#222", elinewidth=1.2, capsize=3)

    ax.set_xticks(x, labels=tbl["Genre"].astype(str).values, rotation=28, ha='right')
    _set_axes(ax, ylabel="Average Rating", title=f"Top {len(tbl)} Genres by Average Rating (min n={min_count})")

    overall = float(df["Rating"].mean())
    ax.axhline(overall, color="#666", linestyle='--', linewidth=1.2, label=f"Overall mean {overall:.2f}")
    ax.legend(loc='lower right', fontsize=9)

    for i, (m, n) in enumerate(zip(tbl["Mean"].values, tbl["n"].values)):
        ax.text(i, m + 0.12, f"n={n}", ha='center', va='bottom', fontsize=8, color="#111")

    _footnote(ax, "误差线为95%自助法置信区间；设最小样本门槛以抑制“冷门类型=高分”的错觉。")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def fig_year_counts(df: pd.DataFrame, save_path: str) -> None:
    """首播年度趋势（NPG主色线+填充）"""
    if "First air date" not in df.columns:
        return
    dfy = add_year_columns(df)
    agg = dfy.groupby("Year").size().dropna().sort_index()
    if len(agg) == 0:
        return

    x = agg.index.values.astype(float)
    y = agg.values.astype(float)

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.plot(x, y, marker='o', color=NPG["primary"], linewidth=1.9, markersize=4.8)
    ax.fill_between(x, y, step='pre', color=NPG["teal"], alpha=0.25)

    if len(agg) >= 3:
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        ax.plot(x, p(x), linestyle='--', color=NPG["secondary"], linewidth=1.3, label='Polynomial trend')
        ax.legend(loc='best', fontsize=9)
    elif len(agg) == 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), linestyle='--', color=NPG["secondary"], linewidth=1.3, label='Linear trend')
        ax.legend(loc='best', fontsize=9)

    max_year, max_count = int(agg.idxmax()), int(agg.max())
    ax.annotate(f"Peak: {max_count} in {max_year}",
                xy=(max_year, max_count), xytext=(12, 12), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', lw=1.2, color=NPG["primary"]),
                fontsize=9, color="#111")

    _set_axes(ax, xlabel="Premiere Year", ylabel="Number of Shows", title="TV Show Premiere Trend Over Time")
    _footnote(ax, "仅根据可用首播年份统计；首页截断样本可能放大近年节目密度。")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def fig_network_topn(df: pd.DataFrame, topn: int, save_path: str) -> None:
    """网络频道节目数量（Top-N；NPG色循环 + 数值标签）"""
    if "Network" not in df.columns:
        return
    cnt = df["Network"].fillna("").replace("", "Unknown").value_counts().head(topn)
    if len(cnt) == 0:
        return

    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    x = np.arange(len(cnt))
    bar_colors = [NPG_CYCLE[i % len(NPG_CYCLE)] for i in range(len(cnt))]

    ax.bar(x, cnt.values, color=bar_colors, edgecolor="#222", linewidth=0.8, width=0.72)
    ax.set_xticks(x, labels=cnt.index.astype(str), rotation=35, ha='right')

    for i, v in enumerate(cnt.values):
        ax.text(i, v + max(cnt.values) * 0.01, f"{int(v)}", ha='center', va='bottom', fontsize=9, color="#111")

    _set_axes(ax, ylabel="Number of Shows", title=f"Top {topn} Networks/Channels by Show Count")
    _footnote(ax, f"仅显示Top-{topn}；合计（Top-{topn}）={int(cnt.sum())}。")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def fig_genre_boxplot(df: pd.DataFrame, topk_genres: int, save_path: str) -> None:
    """类型评分分布箱线图（Top-k 类型；NPG色）"""
    if "Genres" not in df.columns or "Rating" not in df.columns:
        return
    dfg = explode_genres(df)
    if len(dfg) == 0:
        return

    top_genres = dfg["Genre"].value_counts().head(topk_genres).index.tolist()
    data = [dfg.loc[dfg["Genre"] == g, "Rating"].dropna().values for g in top_genres]
    valid = [(g, d) for g, d in zip(top_genres, data) if len(d) > 0]
    if not valid:
        return

    labels, series = zip(*valid)
    fig, ax = plt.subplots(figsize=(11.8, 7.2))

    # 每个箱体给一个NPG颜色
    colors = [NPG_CYCLE[i % len(NPG_CYCLE)] for i in range(len(labels))]
    bp = ax.boxplot(series, labels=labels, patch_artist=True, showmeans=True,
                    boxprops=dict(facecolor='white', edgecolor="#222", linewidth=1.2),
                    medianprops=dict(color="#222", linewidth=1.6),
                    meanprops=dict(marker='D', markerfacecolor=NPG["primary"], markeredgecolor="#222",
                                   markersize=5.5, markeredgewidth=0.8),
                    whiskerprops=dict(linewidth=1.2, color="#222"),
                    capprops=dict(linewidth=1.2, color="#222"),
                    flierprops=dict(marker='o', markerfacecolor=NPG["teal"], markeredgecolor='none', markersize=4, alpha=0.6))
    # 给箱体上色边框（顶部边框）
    for patch, c in zip(bp['boxes'], colors):
        try:
            patch.set_facecolor(c + "20")  # 轻微透明（若环境不支持HEX+alpha则fallback）
        except Exception:
            patch.set_facecolor(c)
            patch.set_alpha(0.25)

    _set_axes(ax, ylabel="Rating", title=f"Rating Distribution by Genre (Top {topk_genres} by count)")
    _footnote(ax, "箱体=IQR，线=中位数，菱形=均值；仅显示出现次数最多的Top-k类型。")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def fig_platform_mean_rating(df: pd.DataFrame, save_path: str, min_group_n: int = 5) -> None:
    """平台差异（Streaming vs TV）：NPG色 + 95%CI + 置换检验p值"""
    if "Network" not in df.columns or "Rating" not in df.columns:
        return
    d = df.copy()
    d["Platform"] = d["Network"].apply(_platform_from_network)

    rows = []
    for name, s in d.groupby("Platform")["Rating"]:
        vals = s.dropna().values
        if len(vals) < min_group_n:
            continue
        mean, low, high = _bootstrap_ci_mean(vals)
        rows.append((name, mean, low, high, len(vals)))
    if not rows:
        return
    tbl = pd.DataFrame(rows, columns=["Platform", "Mean", "Low", "High", "n"]).sort_values("Mean", ascending=False)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    x = np.arange(len(tbl))
    colors = [NPG["success"] if p == "Streaming" else NPG["primary"] for p in tbl["Platform"].values]

    ax.bar(x, tbl["Mean"].values, color=colors, edgecolor="#222", linewidth=0.8, width=0.7)
    yerr = np.vstack([tbl["Mean"].values - tbl["Low"].values, tbl["High"].values - tbl["Mean"].values])
    ax.errorbar(x, tbl["Mean"].values, yerr=yerr, fmt='none', ecolor="#222", elinewidth=1.2, capsize=3)

    ax.set_xticks(x, labels=tbl["Platform"].astype(str).values, rotation=0, ha='center')
    _set_axes(ax, ylabel="Average Rating", title="Average Rating by Platform (95% CI)")

    overall = float(d["Rating"].mean())
    ax.axhline(overall, color="#666", linestyle='--', linewidth=1.1, label=f"Overall mean {overall:.2f}")
    ax.legend(loc='lower right', fontsize=9)

    for i, (m, n) in enumerate(zip(tbl["Mean"].values, tbl["n"].values)):
        ax.text(i, m + 0.12, f"n={n}", ha='center', va='bottom', fontsize=8, color="#111")

    # 置换检验（Streaming vs TV）
    s_vals = d[d["Platform"] == "Streaming"]["Rating"].dropna().values
    t_vals = d[d["Platform"] == "TV"]["Rating"].dropna().values
    if len(s_vals) >= min_group_n and len(t_vals) >= min_group_n:
        p = _permutation_test_diff_means(s_vals, t_vals)
        _footnote(ax, f"Permutation p (Streaming vs TV) = {p:.4f}；仅显示样本量≥{min_group_n}的组。")
    else:
        _footnote(ax, f"仅显示样本量≥{min_group_n}的组。")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ========= 数据清洗（关键：排除占位结束日） =========
_PLACEHOLDER_END_DATES = {"9999-12-31", "9999/12/31"}

def sanitize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 将 End date 为占位值（9999-12-31 或 9999/12/31）置为 NaN，避免被下游分析/图表误用
    - 不改动 First air date（首播年用于 fig_year_counts）
    """
    df = df.copy()
    if "End date" in df.columns:
        mask_placeholder = df["End date"].astype(str).isin(_PLACEHOLDER_END_DATES)
        if mask_placeholder.any():
            df.loc[mask_placeholder, "End date"] = np.nan
            logging.info(f"Sanitized placeholder End date rows: {int(mask_placeholder.sum())}")
    return df


# ========= 主流程 =========
def run(csv_path: str, save_dir: str, topk: int = 20) -> None:
    setup_logging()
    logging.info("=" * 60)
    logging.info("TVMaze Data Visualization (Nature Color) Started")
    logging.info(f"Input: {csv_path}, Output: {save_dir}")

    configure_matplotlib()
    ensure_dir(save_dir)

    df = read_csv(csv_path)

    # ---- 字段健壮性检查与补齐 ----
    expected_cols = ["Rating", "Title", "Genres", "Status", "First air date", "Network", "End date"]
    for c in expected_cols:
        if c not in df.columns:
            logging.warning(f"Column '{c}' not found in dataset; filling with default.")
            df[c] = np.nan

    # 评分统一为数值
    df["Rating"] = pd.to_numeric(df.get("Rating"), errors="coerce")

    # 关键：排除“连载中占位结束日”
    df = sanitize_dates(df)

    logging.info(f"Dataset: {len(df)} rows; rows with rating: {df['Rating'].notna().sum()}, "
                 f"Avg rating: {df['Rating'].mean():.2f}")

    # 图文对比的核心可视化（全部采用 NPG 色）：
    fig_top_rated(df, k=topk, save_path=os.path.join(save_dir, f"top_{topk}_rated.png"))
    fig_rating_hist(df, bins=20, save_path=os.path.join(save_dir, "rating_hist.png"))
    fig_status_mean_rating(df, save_path=os.path.join(save_dir, "status_mean_rating.png"))
    fig_genre_mean_rating(df, topn=10, save_path=os.path.join(save_dir, "genre_mean_top10.png"), min_count=5)
    fig_year_counts(df, save_path=os.path.join(save_dir, "year_counts.png"))
    fig_network_topn(df, topn=15, save_path=os.path.join(save_dir, "network_top15.png"))
    fig_genre_boxplot(df, topk_genres=6, save_path=os.path.join(save_dir, "genre_boxplot_top6.png"))

    fig_platform_mean_rating(df, save_path=os.path.join(save_dir, "platform_mean_rating.png"))

    logging.info(f"SUCCESS: All visualizations saved to {save_dir}")
    logging.info("=" * 60)
