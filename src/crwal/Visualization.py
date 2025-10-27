# viz_for_latex.py
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def configure_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": False,
        "lines.linewidth": 1.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    # 中英兼容的无衬线字体
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "SimHei"] + plt.rcParams.get("font.sans-serif", [])
    plt.rcParams["axes.unicode_minus"] = False


BROADCAST = [
    "abc", "cbs", "nbc", "fox", "the cw", "pbs",
    "bbc one", "bbc two", "bbc three", "bbc four", "itv", "channel 4", "channel 5",
]
CABLE_PREMIUM = [
    "hbo", "showtime", "starz", "amc", "fx", "fxx", "usa network", "syfy", "tnt", "tbs",
    "a&e", "history", "bravo", "comedy central", "mtv", "vh1", "e!", "hallmark", "freeform",
    "nickelodeon", "cartoon network", "adult swim", "discovery channel", "national geographic",
    "nat geo", "hgtv", "food network", "bbc america", "sky one", "sky atlantic", "tv land",
]
STREAMERS = [
    "netflix", "amazon", "prime video", "hulu", "disney+", "apple tv+", "paramount+", "peacock", "max",
    "discovery+", "amc+", "crave", "stan", "itvx", "bbc iplayer", "hotstar", "jiocinema", "viu",
    "iqiyi", "tencent", "youku",
]


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d")
    except Exception:
        return None


def _year_from_date_str(s: Optional[str]) -> Optional[int]:
    dt = _parse_date(s)
    return dt.year if dt else None


def _status_group(raw: Optional[str]) -> str:
    if not isinstance(raw, str):
        return "Other"
    s = raw.strip().lower()
    if "ended" in s:
        return "Ended"
    if any(k in s for k in ["running", "continuing", "returning", "on hiatus", "in production"]):
        return "Running"
    if any(k in s for k in ["to be determined", "tbd", "unknown", "pending"]):
        return "To Be Determined"
    return "Other"


def _platform_type_from_network(net: Optional[str]) -> str:
    if not isinstance(net, str) or not net.strip():
        return "Other"
    s = net.strip().lower()
    if any(k in s for k in STREAMERS):
        return "Streamer"
    if any(k in s for k in BROADCAST):
        return "Broadcast"
    if any(k in s for k in CABLE_PREMIUM):
        return "Cable/Premium"
    return "Other"


def _parse_genres(cell: Optional[str]) -> List[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    c = cell.strip()
    if c.startswith("[") and c.endswith("]"):
        try:
            arr = json.loads(c)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in c.split(",") if x.strip()]


def _freedman_diaconis_bins(vals: np.ndarray, max_bins: int = 50) -> int:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 10
    q75, q25 = np.percentile(v, [75, 25])
    iqr = max(q75 - q25, 1e-9)
    bw = 2 * iqr * (v.size ** (-1/3))
    if bw <= 0:
        return min(20, len(np.unique(v)))
    bins = int(np.ceil((v.max() - v.min()) / bw))
    return max(5, min(bins, max_bins))


def _bootstrap_ci_mean(vals: np.ndarray, n_boot: int = 3000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = v.size
    boots = rng.choice(v, size=(n_boot, n), replace=True).mean(axis=1)
    mean = float(v.mean())
    low = float(np.quantile(boots, alpha / 2))
    high = float(np.quantile(boots, 1 - alpha / 2))
    return mean, low, high


def derive_variables(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ["Title", "First air date", "End date", "Rating", "Genres", "Status", "Network", "Summary"]:
        if c not in d.columns:
            d[c] = np.nan

    d["Rating"] = pd.to_numeric(d["Rating"], errors="coerce")
    d["End date"] = d["End date"].replace({"9999-12-31": np.nan, "9999/12/31": np.nan})

    d["StatusGroup"] = d["Status"].apply(_status_group)
    d["YearFirst"] = d["First air date"].apply(_year_from_date_str)
    d["YearEnd"] = d["End date"].apply(_year_from_date_str)

    def _years_active(row) -> Optional[float]:
        fst = _parse_date(row.get("First air date"))
        if not fst:
            return np.nan
        end_dt = _parse_date(row.get("End date"))
        if end_dt is None and row.get("StatusGroup") != "Ended":
            end_dt = datetime.combine(date.today(), datetime.min.time())
        if end_dt is None:
            return np.nan
        return max(0.0, (end_dt - fst).days / 365.25)

    d["YearsActive"] = d.apply(_years_active, axis=1)
    d["PlatformType"] = d["Network"].apply(_platform_type_from_network)
    d["NumGenres"] = d["Genres"].apply(lambda x: len(_parse_genres(x)))
    return d



def fig_rating_hist(d: pd.DataFrame, out_path: str) -> None:
    vals = d["Rating"].dropna().values
    if vals.size == 0:
        return
    bins = _freedman_diaconis_bins(vals)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(vals, bins=bins, alpha=0.85, edgecolor="#222222", linewidth=0.8)
    mean, median = vals.mean(), np.median(vals)
    ax.axvline(mean, linestyle="--", linewidth=1.8, label=f"Mean {mean:.2f}")
    ax.axvline(median, linestyle="-.", linewidth=1.6, label=f"Median {median:.2f}")
    ax.set_xlabel("Rating (0–10)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ratings")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(out_path)
    plt.close(fig)


def fig_avg_rating_by_genre(
    d: pd.DataFrame,
    out_path: str,
    top_n: int = 12,
    min_n: int = 5,
    show_counts: str = "none",   # "xtick" | "above" | "none"
    sort_by: str = "mean",        # "mean" | "freq"
    save_svg: bool = True,
) -> None:
    """
    画“不同 Genre 的平均评分 + 95%CI”的期刊风格柱状图。
    - show_counts: n 的展示位置；"xtick" 放到 x 轴标签中，"above" 放到柱顶，"none" 不显示
    - sort_by: 输出表的排序方式；"mean" 按均值降序，"freq" 按样本量降序
    """
    # -------- 样式（近似期刊风） --------
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "grid.color": "#808080",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

    # -------- 数据处理 --------
    dd = d.copy()
    dd["__genres"] = dd["Genres"].apply(_parse_genres)
    dd = dd.explode("__genres").rename(columns={"__genres": "Genre"}).dropna(subset=["Genre"])
    if dd.empty:
        return

    freq = dd["Genre"].value_counts()
    cand = freq.head(top_n).index.tolist()

    rows = []
    for g in cand:
        x = dd.loc[dd["Genre"] == g, "Rating"].dropna().values
        if x.size >= min_n:
            m, lo, hi = _bootstrap_ci_mean(x)
            rows.append((g, m, lo, hi, x.size))
    if not rows:
        return

    tbl = pd.DataFrame(rows, columns=["Genre", "Mean", "Low", "High", "n"])
    if sort_by == "freq":
        tbl = tbl.sort_values("n", ascending=False)
    else:
        tbl = tbl.sort_values("Mean", ascending=False).reset_index(drop=True)

    # -------- 调色：低饱和度多色 --------
    def _palette(n, cmap_name="tab20"):
        cmap = plt.get_cmap(cmap_name)
        return [cmap(i) for i in np.linspace(0.05, 0.95, n)]
    colors = _palette(len(tbl))

    # -------- 绘图 --------
    fig_w = max(7.6, 0.55 * len(tbl) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    x = np.arange(len(tbl))
    bar_kwargs = dict(width=0.7, edgecolor="#222222", linewidth=0.8, alpha=0.95)
    for i, (m, c) in enumerate(zip(tbl["Mean"].values, colors)):
        ax.bar(i, m, color=c, **bar_kwargs)

    # 误差线（95% CI）
    y = tbl["Mean"].values
    yerr = np.vstack([y - tbl["Low"].values, tbl["High"].values - y])
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#222222", elinewidth=1.0, capsize=3, zorder=3)

    # x 轴标签与 n 的展示
    if show_counts == "xtick":
        xticklabels = [f"{g}\n(n={n})" for g, n in zip(tbl["Genre"], tbl["n"])]
    else:
        xticklabels = tbl["Genre"].tolist()
    ax.set_xticks(x, labels=xticklabels, rotation=28, ha="right")

    # 柱顶标注 n
    if show_counts == "above":
        for i, (m, n) in enumerate(zip(tbl["Mean"].values, tbl["n"].values)):
            ax.text(i, m + (0.015 * (tbl["High"].max() - tbl["Low"].min())), f"n={n}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Average Rating")
    ax.set_title("Average Rating by Genre (Top by frequency)")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # 合理的 y 轴范围与刻度
    y_min = max(0, (tbl["Low"].min() - 0.1))
    y_max = min(10, (tbl["High"].max() + 0.1)) if np.isfinite(tbl["High"].max()) else tbl["High"].max()
    if np.isfinite(y_min) and np.isfinite(y_max) and (y_max > y_min):
        ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", format="png")
    # 与原参数保持一致（即使未另存 svg，也保留形参）
    plt.close(fig)


def fig_box_rating_by_status(d: pd.DataFrame, out_path: str) -> None:
    required = {"StatusGroup", "Rating"}
    if not required.issubset(d.columns):
        return
    df = d.dropna(subset=["Rating"]).copy()
    if df.empty:
        return

    # 规范化标签并排序
    def _normalize_status(x: str) -> str:
        s = str(x).strip().lower()
        if re.match(r"^\s*running\b", s):
            return "Running"
        if re.search(r"\b(ended|complete|completed|finished|closed)\b", s):
            return "Ended"
        if re.search(r"\b(to\s*be\s*determined|tbd|tba|pending|unknown)\b", s):
            return "To Be Determined"
        if s in {"ended", "running", "other"}:
            return s.title()
        if s == "to be determined":
            return "To Be Determined"
        return "Other"

    df["StatusGroupClean"] = df["StatusGroup"].map(_normalize_status)
    labels = [g for g in ("Ended", "Running", "To Be Determined", "Other") if g in df["StatusGroupClean"].unique()]

    # 绘图
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.8, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": False,
    })

    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=150)
    sns.boxplot(
        data=df, 
        x="StatusGroupClean", 
        y="Rating", 
        hue="StatusGroupClean",     
        order=labels,
        palette="deep", 
        showmeans=True, 
        notch=True, 
        linewidth=1.5, 
        ax=ax,
        legend=False            
    )
    ax.set_title("Ratings by StatusGroup", fontsize=10, loc="left", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Rating", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_scatter_yearsactive_vs_rating(d: pd.DataFrame, out_path: str) -> None:
    dd = d.dropna(subset=["YearsActive", "Rating"]).copy()
    if dd.empty:
        return
    x, y = dd["YearsActive"].values, dd["Rating"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=10, alpha=0.5, edgecolor="none", c="blue", label="Data points")

    if x.size >= 2:
        k, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(xx, k * xx + b, linewidth=2, linestyle="--", color="red", label=f"Fit: y={k:.2f}x + {b:.2f}")
        y_pred = k * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) if x.size > 1 else np.nan
        r2 = 1 - ss_res / ss_tot if ss_tot and np.isfinite(ss_tot) else np.nan
        ax.legend(loc="lower left", frameon=False, title=f"R² = {r2:.3f}")

    ax.set_xlabel("Years Active (years)")
    ax.set_ylabel("Rating")
    ax.set_title("Years Active vs Rating (with linear fit)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_avg_rating_by_network_top10(d: pd.DataFrame, out_path: str, top_n_by_count: int = 10, min_n: int = 5) -> None:
    dd = d.copy()
    dd["Network"] = dd["Network"].fillna("").replace("", "Unknown")
    counts = dd["Network"].value_counts()
    cand = counts.head(top_n_by_count).index.tolist()

    rows = []
    for net in cand:
        x = dd.loc[dd["Network"] == net, "Rating"].dropna().values
        if x.size >= min_n:
            m, lo, hi = _bootstrap_ci_mean(x)
            rows.append((net, m, lo, hi))
    if not rows:
        return

    tbl = pd.DataFrame(rows, columns=["Network", "Mean", "Low", "High"]).sort_values("Mean", ascending=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(tbl))
    colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(tbl)))

    ax.barh(y, tbl["Mean"], color=colors, edgecolor="none", height=0.6)
    xerr = np.vstack([tbl["Mean"] - tbl["Low"], tbl["High"] - tbl["Mean"]])
    ax.errorbar(tbl["Mean"], y, xerr=xerr, fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3)

    ax.set_yticks(y, labels=tbl["Network"])
    ax.set_xlabel("Average Rating")
    ax.set_title("Average Rating by Network (Top 10 by representation)", fontsize=12, weight="bold", pad=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, color="gray", alpha=0.4)
    ax.yaxis.grid(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_platform_type_share(d: pd.DataFrame, out_path: str) -> None:
    cnt = d["PlatformType"].value_counts()
    if cnt.empty:
        return
    share = (cnt / cnt.sum() * 100).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = np.arange(len(share))
    ax.barh(y, share.values, height=0.7, edgecolor="#222222", linewidth=0.8, alpha=0.9)
    ax.set_yticks(y, labels=share.index.tolist())
    ax.set_xlim(0, max(share.values.max() * 1.15, 10))
    for i, v in enumerate(share.values):
        ax.text(v + max(share.values) * 0.01, i, f"{v:.1f}%", va="center", ha="left", fontsize=9)
    ax.set_xlabel("Share of Shows (%)")
    ax.set_title("PlatformType Share (Streamer/Broadcast/Cable-Premium/Other)")
    fig.savefig(out_path)
    plt.close(fig)


def fig_new_shows_by_year(d: pd.DataFrame, out_path: str) -> None:
    s = d["YearFirst"].dropna().astype(int).value_counts().sort_index()
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(s.index.values, s.values, width=0.85, edgecolor="#222222", linewidth=0.6, alpha=0.9)
    ax.set_xlabel("YearFirst")
    ax.set_ylabel("Number of New Shows")
    ax.set_title("New Shows by Year (YearFirst)")
    fig.savefig(out_path)
    plt.close(fig)


def Visualization(csv_path: str, out_dir: str = "figures") -> None:
    configure_matplotlib()
    ensure_dir(out_dir)

    df = read_csv(csv_path)
    df = derive_variables(df)

    fig_rating_hist(df, os.path.join(out_dir, "fig_rating_hist.png"))
    fig_avg_rating_by_genre(df, os.path.join(out_dir, "fig_avg_rating_by_genre.png"), top_n=12, min_n=5)
    fig_box_rating_by_status(df, os.path.join(out_dir, "fig_box_rating_by_status.png"))
    fig_scatter_yearsactive_vs_rating(df, os.path.join(out_dir, "fig_scatter_yearsactive_vs_rating.png"))
    fig_avg_rating_by_network_top10(df, os.path.join(out_dir, "fig_avg_rating_by_network_top10.png"), top_n_by_count=10, min_n=5)
    fig_platform_type_share(df, os.path.join(out_dir, "fig_platform_type_share.png"))
    fig_new_shows_by_year(df, os.path.join(out_dir, "fig_new_shows_by_year.png"))
