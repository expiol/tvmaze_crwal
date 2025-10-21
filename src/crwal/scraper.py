from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

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


def is_valid_show(item: Dict[str, Any]) -> bool:
    """检查节目数据是否完整（必需字段 + 评分）"""
    # Title
    if not item.get("name"):
        return False
    
    # First air date
    if not item.get("premiered"):
        return False
    
    # Rating（作业需要分析评分，必须有）
    rating_obj = item.get("rating") or {}
    rating = rating_obj.get("average") if isinstance(rating_obj, dict) else None
    if not rating:
        return False
    
    # Genres (至少有一个)
    genres = item.get("genres")
    if not genres or (isinstance(genres, list) and len(genres) == 0):
        return False
    
    # Status
    if not item.get("status"):
        return False
    
    # Network或WebChannel至少有一个
    network = item.get("network")
    web_channel = item.get("webChannel")
    if not network and not web_channel:
        return False
    
    return True


def fetch_show_by_id(sess: requests.Session, show_id: int) -> Optional[Dict[str, Any]]:
    """获取单个节目详情（用于并发）"""
    url = f"{Config.BASE_URL}/shows/{show_id}"
    data = safe_get_json(sess, url)
    if data and isinstance(data, dict) and data.get("name"):
        return data
    return None


def fetch_shows(count: int, fetch_newest: bool = True, filter_quality: bool = True) -> List[Dict[str, Any]]:
    """获取节目列表，默认获取最新的、数据完整的节目"""
    sess = get_session()
    results: List[Dict[str, Any]] = []
    skipped = 0
    
    if fetch_newest:
        print("→ Finding shows from page 260 down to 0 (with ratings)...")
        logging.info("Fetching shows from page 260 down to 0...")
        
        # 从page 260开始，一直往前翻到page 0，直到收集到200条
        start_page = 340
        page = start_page
        
        # 从这些页面开始抓取
        print(f"→ Fetching from page {page} down to page 0 until {count} shows...")
        
        with tqdm(total=count, desc="Fetching shows", ncols=100, 
                 unit="show", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as bar:
            while len(results) < count and page >= 0:
                url = f"{Config.BASE_URL}/shows?page={page}"
                data = safe_get_json(sess, url)
                
                if not data or not isinstance(data, list):
                    page -= 1
                    continue
                
                # 反向添加（从页面的最后开始，即ID最大的）
                for item in reversed(data):
                    if len(results) >= count:
                        break
                    if isinstance(item, dict) and item.get("name"):
                        # 数据质量过滤
                        if filter_quality:
                            if is_valid_show(item):
                                results.append(item)
                                bar.update(1)
                            else:
                                skipped += 1
                        else:
                            results.append(item)
                            bar.update(1)
                
                page -= 1
        
        if filter_quality and skipped > 0:
            print(f"→ Skipped {skipped} shows with missing data (Total collected: {len(results)})")
    else:
        print("→ Fetching from page 0 (oldest shows)...")
        page = 0
        with tqdm(total=count, desc="Fetching shows", ncols=100, unit="show",
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as bar:
            while len(results) < count and page < 50:  # 限制搜索范围
                url = f"{Config.BASE_URL}/shows?page={page}"
                data = safe_get_json(sess, url)
                
                if not data or not isinstance(data, list):
                    logging.warning(f"Empty or invalid page at {page}, stopping")
                    break

                for item in data:
                    if len(results) >= count:
                        break
                    if isinstance(item, dict) and item.get("name"):
                        # 数据质量过滤
                        if filter_quality:
                            if is_valid_show(item):
                                results.append(item)
                                bar.update(1)
                            else:
                                skipped += 1
                        else:
                            results.append(item)
                            bar.update(1)
                page += 1
        
        if filter_quality and skipped > 0:
            print(f"→ Skipped {skipped} shows with missing data (Total collected: {len(results)})")

    logging.info(f"Fetched {len(results)} shows ({'newest' if fetch_newest else 'oldest'} first)")
    if filter_quality:
        logging.info(f"Data quality filter: Skipped {skipped} shows with missing fields")
    return results[:count]


def fetch_episode_date_wrapper(args):
    """包装函数用于并发"""
    sess, show_id = args
    return show_id, get_earliest_air_date(sess, show_id, exclude_specials=True)


def transform(items: List[Dict[str, Any]], sess: requests.Session, use_episodes_api: bool = True) -> pd.DataFrame:
    """转换原始数据为DataFrame"""
    episode_dates = {}
    
    if use_episodes_api:
        print(f"→ Fetching earliest air dates from episodes API with {Config.MAX_WORKERS} threads...")
        show_ids = [it.get("id") for it in items if it.get("id")]
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            tasks = [(sess, show_id) for show_id in show_ids]
            
            with tqdm(total=len(show_ids), desc="Fetching episodes", ncols=100,
                     unit="show", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as bar:
                for show_id, date in executor.map(fetch_episode_date_wrapper, tasks):
                    if date:
                        episode_dates[show_id] = date
                    bar.update(1)
    
    # 处理数据
    rows: List[Dict[str, Any]] = []
    print("→ Processing show data...")
    
    for it in tqdm(items, desc="Processing shows", ncols=100, unit="show",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        rating_obj = it.get("rating") or {}
        rating = rating_obj.get("average") if isinstance(rating_obj, dict) else None
        
        show_id = it.get("id")
        first_air_date = it.get("premiered")
        
        # 如果有从episodes API获取的日期，使用它
        if use_episodes_api and show_id in episode_dates:
            first_air_date = episode_dates[show_id]
        
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


def run(count: int, out_path: str, log_file: str = None) -> pd.DataFrame:
    """运行数据抓取流程"""
    setup_logging(log_file)
    logging.info("=" * 60)
    logging.info("TVMaze Scraper Started")
    logging.info(f"Parameters: count={count}, output={out_path}")
    
    sess = get_session()
    items = fetch_shows(count, fetch_newest=True, filter_quality=True)
    if not items:
        logging.error("No data fetched")
        raise RuntimeError("Failed to fetch data")
    
    df = transform(items, sess, use_episodes_api=True)
    save_csv(df, out_path)
    
    logging.info(f"SUCCESS: Saved {len(df)} rows to {out_path}")
    logging.info("=" * 60)
    
    return df
