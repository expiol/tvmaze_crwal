from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from .util import (
    Config, get_session, safe_get_json, strip_html, truncate,
    normalize_date, extract_network, format_genres, save_csv, setup_logging
)


COLUMNS = [
    "Title", "First air date", "End date", "Rating",
    "Genres", "Status", "Network", "Summary"
]


def fetch_shows(count: int) -> List[Dict[str, Any]]:
    sess = get_session()
    results: List[Dict[str, Any]] = []
    page = 0

    with tqdm(total=count, desc="Fetching shows", ncols=90, unit="show") as bar:
        while len(results) < count:
            url = f"{Config.BASE_URL}/shows?page={page}"
            data = safe_get_json(sess, url)
            
            if not data or not isinstance(data, list):
                logging.warning(f"Empty or invalid page at {page}, stopping")
                break

            for item in data:
                if len(results) >= count:
                    break
                if isinstance(item, dict) and item.get("name"):
                    results.append(item)
                    bar.update(1)
            page += 1

    logging.info(f"Fetched {len(results)} shows from {page} pages")
    return results[:count]


def transform(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for it in items:
        rating_obj = it.get("rating") or {}
        rating = rating_obj.get("average") if isinstance(rating_obj, dict) else None

        rows.append({
            "Title": (it.get("name") or "").strip(),
            "First air date": normalize_date(it.get("premiered")),
            "End date": normalize_date(it.get("ended")),
            "Rating": rating,
            "Genres": format_genres(it.get("genres") or []),
            "Status": (it.get("status") or "").strip(),
            "Network": extract_network(it),
            "Summary": truncate(strip_html(it.get("summary") or ""))
        })

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    
    total = len(df)
    missing_rating = df["Rating"].isna().sum()
    logging.info(
        f"Data quality: Total={total}, Missing Rating={missing_rating} "
        f"({100 * missing_rating / total:.1f}%)"
    )
    
    return df


def run(count: int, out_path: str, log_file: str = None) -> pd.DataFrame:
    setup_logging(log_file)
    logging.info("=" * 60)
    logging.info("TVMaze Scraper Started")
    logging.info(f"Parameters: count={count}, output={out_path}")
    
    items = fetch_shows(count)
    if not items:
        logging.error("No data fetched")
        raise RuntimeError("Failed to fetch data")
    
    df = transform(items)
    save_csv(df, out_path)
    
    logging.info(f"SUCCESS: Saved {len(df)} rows to {out_path}")
    logging.info("=" * 60)
    
    return df
