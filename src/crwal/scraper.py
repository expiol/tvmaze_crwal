# -*- coding: utf-8 -*-
"""
Scrape ≥200 shows from TVMaze API and export to CSV with required columns.

Usage:
    python -m src.crwal.scraper --count 200 --out data/your_name+id.csv --log logs/run.log
"""
from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

# 兼容按包运行与直接脚本运行
try:
    from .util import (
        Config,
        get_session,
        safe_get_json,
        strip_html,
        truncate,
        normalize_date,
        extract_network,
        format_genres,
        save_csv,
        setup_logging,
    )
except ImportError: 
    from util import (
        Config,
        get_session,
        safe_get_json,
        strip_html,
        truncate,
        normalize_date,
        extract_network,
        format_genres,
        save_csv,
        setup_logging,
    )


COLUMNS = [
    "Title",
    "First air date",
    "End date",
    "Rating",
    "Genres",
    "Status",
    "Network",
    "Summary",
]


def fetch_shows(count: int) -> List[Dict[str, Any]]:
    sess = get_session()
    results: List[Dict[str, Any]] = []
    page = 0

    with tqdm(total=count, desc="Fetching shows", ncols=90) as bar:
        while len(results) < count:
            url = f"{Config.BASE_URL}/shows?page={page}"
            data = safe_get_json(sess, url)
            if not data:
                logging.warning("Empty page at %s, stop.", url)
                break

            for item in data:
                if len(results) >= count:
                    break
                results.append(item)
                bar.update(1)
            page += 1

    return results[:count]


def transform(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for it in items:
        title = (it.get("name") or "").strip()
        premiered = normalize_date(it.get("premiered"))
        ended = normalize_date(it.get("ended"))
        rating = None
        rating_obj = it.get("rating") or {}
        rating = rating_obj.get("average", None)

        genres = it.get("genres") or []
        status = (it.get("status") or "").strip()
        network = extract_network(it)

        summary_raw = it.get("summary") or ""
        summary = truncate(strip_html(summary_raw))

        rows.append(
            {
                "Title": title,
                "First air date": premiered,
                "End date": ended,
                "Rating": rating,
                "Genres": format_genres(genres),
                "Status": status,
                "Network": network,
                "Summary": summary,
            }
        )

    df = pd.DataFrame(rows, columns=COLUMNS)
    # 类型友好：Rating -> float
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    return df


def run(count: int, out_path: str, log_file: str | None = None) -> None:
    setup_logging(log_file)
    logging.info("Start scraping: target count=%d", count)
    items = fetch_shows(count)
    df = transform(items)
    save_csv(df, out_path)
    logging.info("Saved %d rows to %s", len(df), out_path)



