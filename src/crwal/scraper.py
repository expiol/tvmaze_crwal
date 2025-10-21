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


def fetch_shows(count: int, fetch_newest: bool = True) -> List[Dict[str, Any]]:
    """
    获取节目列表
    Args:
        count: 要获取的节目数量
        fetch_newest: True=获取最新节目, False=获取最早节目
    """
    sess = get_session()
    results: List[Dict[str, Any]] = []
    
    if fetch_newest:
        # 方法1: 先获取最近更新的节目
        logging.info("Fetching recently updated shows...")
        url = f"{Config.BASE_URL}/updates/shows"
        updates = safe_get_json(sess, url)
        
        if updates and isinstance(updates, dict):
            # updates返回的是 {show_id: timestamp} 的字典
            # 按时间戳排序，取最新的
            sorted_ids = sorted(updates.items(), key=lambda x: x[1], reverse=True)
            recent_ids = [int(show_id) for show_id, _ in sorted_ids[:count * 2]]  # 多取一些以防有无效数据
            
            logging.info(f"Got {len(recent_ids)} recently updated show IDs")
            
            # 获取这些节目的详细信息
            with tqdm(total=min(count, len(recent_ids)), desc="Fetching show details", ncols=90, unit="show") as bar:
                for show_id in recent_ids:
                    if len(results) >= count:
                        break
                    
                    url = f"{Config.BASE_URL}/shows/{show_id}"
                    data = safe_get_json(sess, url)
                    
                    if data and isinstance(data, dict) and data.get("name"):
                        results.append(data)
                        bar.update(1)
        
        # 如果还不够，从最新的页面开始补充
        if len(results) < count:
            logging.info(f"Fetching additional shows from recent pages (need {count - len(results)} more)...")
            # 估算最大页码（TVMaze大约有7万+节目，每页250个）
            estimated_max_page = 280
            page = estimated_max_page
            
            with tqdm(total=count - len(results), desc="Fetching from pages", ncols=90, unit="show") as bar:
                while len(results) < count and page >= 0:
                    url = f"{Config.BASE_URL}/shows?page={page}"
                    data = safe_get_json(sess, url)
                    
                    if not data or not isinstance(data, list):
                        page -= 1
                        continue
                    
                    # 反向添加（从页面的最后开始）
                    for item in reversed(data):
                        if len(results) >= count:
                            break
                        if isinstance(item, dict) and item.get("name"):
                            # 检查是否已经添加过（避免重复）
                            if not any(r.get("id") == item.get("id") for r in results):
                                results.append(item)
                                bar.update(1)
                    page -= 1
    else:
        # 原来的逻辑：从第0页开始获取最早的节目
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

    logging.info(f"Fetched {len(results)} shows ({'newest' if fetch_newest else 'oldest'} first)")
    return results[:count]


def transform(items: List[Dict[str, Any]], sess: requests.Session, use_episodes_api: bool = False) -> pd.DataFrame:
    """
    转换原始数据为DataFrame
    Args:
        items: 从API获取的show数据列表
        sess: requests session用于获取episodes
        use_episodes_api: 是否使用episodes API获取真实首播日期（默认False）
    """
    rows: List[Dict[str, Any]] = []

    for it in tqdm(items, desc="Processing shows", ncols=90, unit="show"):
        rating_obj = it.get("rating") or {}
        rating = rating_obj.get("average") if isinstance(rating_obj, dict) else None
        
        show_id = it.get("id")
        first_air_date = it.get("premiered")
        
        if use_episodes_api and show_id:
            earliest_date = get_earliest_air_date(sess, show_id, exclude_specials=True)
            if earliest_date:
                first_air_date = earliest_date
        
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


def run(count: int, out_path: str, log_file: str = None, 
        use_episodes_api: bool = False, fetch_newest: bool = True) -> pd.DataFrame:
    """
    运行数据抓取流程
    Args:
        count: 要抓取的节目数量
        out_path: CSV输出路径
        log_file: 日志文件路径
        use_episodes_api: 是否使用episodes API获取真实首播日期（默认False，使用premiered字段更快）
        fetch_newest: 是否获取最新节目（默认True，获取最近更新的节目）
    """
    setup_logging(log_file)
    logging.info("=" * 60)
    logging.info("TVMaze Scraper Started")
    logging.info(f"Parameters: count={count}, output={out_path}")
    logging.info(f"  - use_episodes_api={use_episodes_api}")
    logging.info(f"  - fetch_newest={fetch_newest}")
    
    sess = get_session()
    items = fetch_shows(count, fetch_newest=fetch_newest)
    if not items:
        logging.error("No data fetched")
        raise RuntimeError("Failed to fetch data")
    
    df = transform(items, sess, use_episodes_api=use_episodes_api)
    save_csv(df, out_path)
    
    logging.info(f"SUCCESS: Saved {len(df)} rows to {out_path}")
    logging.info("=" * 60)
    
    return df
