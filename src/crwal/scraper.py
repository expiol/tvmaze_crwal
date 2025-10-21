from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
import requests
from tqdm import tqdm

from .util import (
    Config, get_session, safe_get_json, strip_html, truncate,
    normalize_date, extract_network, extract_web_channel, 
    get_earliest_air_date, format_genres, save_csv, setup_logging
)


COLUMNS = [
    "Title", "First air date", "End date", "Rating",
    "Genres", "Status", "Network", "Summary",
    "Web Channel", "Language", "Runtime", "Premiered Year"
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


def transform(items: List[Dict[str, Any]], sess: requests.Session, use_episodes_api: bool = True) -> pd.DataFrame:
    """
    转换原始数据为DataFrame
    Args:
        items: 从API获取的show数据列表
        sess: requests session用于获取episodes
        use_episodes_api: 是否使用episodes API获取真实首播日期
    """
    rows: List[Dict[str, Any]] = []

    for it in tqdm(items, desc="Processing shows", ncols=90, unit="show"):
        rating_obj = it.get("rating") or {}
        rating = rating_obj.get("average") if isinstance(rating_obj, dict) else None
        
        # 获取首播日期：优先使用episodes API
        show_id = it.get("id")
        first_air_date = it.get("premiered")
        
        if use_episodes_api and show_id:
            earliest_date = get_earliest_air_date(sess, show_id, exclude_specials=True)
            if earliest_date:
                first_air_date = earliest_date
                logging.debug(f"Show {show_id}: Using earliest episode date {earliest_date}")
        
        # 提取首播年份
        premiered_year = ""
        if first_air_date:
            try:
                premiered_year = str(first_air_date[:4])
            except:
                pass

        rows.append({
            "Title": (it.get("name") or "").strip(),
            "First air date": normalize_date(first_air_date),
            "End date": normalize_date(it.get("ended")),
            "Rating": rating,
            "Genres": format_genres(it.get("genres") or []),
            "Status": (it.get("status") or "").strip(),
            "Network": extract_network(it),
            "Summary": truncate(strip_html(it.get("summary") or "")),
            "Web Channel": extract_web_channel(it),
            "Language": (it.get("language") or "").strip(),
            "Runtime": it.get("runtime") if it.get("runtime") else "",
            "Premiered Year": premiered_year
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


def run(count: int, out_path: str, log_file: str = None, use_episodes_api: bool = True) -> pd.DataFrame:
    """
    运行数据抓取流程
    Args:
        count: 要抓取的节目数量
        out_path: CSV输出路径
        log_file: 日志文件路径
        use_episodes_api: 是否使用episodes API获取真实首播日期（推荐：True）
    """
    setup_logging(log_file)
    logging.info("=" * 60)
    logging.info("TVMaze Scraper Started")
    logging.info(f"Parameters: count={count}, output={out_path}, use_episodes_api={use_episodes_api}")
    
    sess = get_session()
    items = fetch_shows(count)
    if not items:
        logging.error("No data fetched")
        raise RuntimeError("Failed to fetch data")
    
    df = transform(items, sess, use_episodes_api=use_episodes_api)
    save_csv(df, out_path)
    
    logging.info(f"SUCCESS: Saved {len(df)} rows to {out_path}")
    logging.info("=" * 60)
    
    return df
