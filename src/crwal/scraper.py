from __future__ import annotations

import logging
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from .util import (
    get_session,
    strip_html,
    truncate,
    normalize_date,
    format_genres,
    save_csv,
    setup_logging,
    get_html,
)

# -------------------------
# Constants
# -------------------------

BASE = "https://www.tvmaze.com"
LISTING_TEMPLATE = "https://www.tvmaze.com/shows?page={page}"

# 输出 CSV 列（符合项目要求的8个必需字段）
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

# 页面范围配置
DEFAULT_START_PAGE = 1
DEFAULT_END_PAGE = 3344  # TVMaze 总页数（截至2025年）

# 请求节流与重试配置（已在util.py中统一管理）
SLEEP_RANGE = (0.2, 0.8)  # 已弃用，使用Config.MIN_INTERVAL
MAX_RETRIES = 2  # 已弃用，使用Config.MAX_RETRY


# -------------------------
# Data Model
# -------------------------

@dataclass
class ShowRecord:
    title: Optional[str] = None
    first_air_date: Optional[str] = None
    end_date: Optional[str] = None
    rating: Optional[str] = None
    genres: Optional[str] = None
    status: Optional[str] = None
    network: Optional[str] = None
    summary: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "Title": self.title,
            "First air date": self.first_air_date,
            "End date": self.end_date,
            "Rating": self.rating,
            "Genres": self.genres,
            "Status": self.status,
            "Network": self.network,
            "Summary": self.summary,
        }


# -------------------------
# Helpers
# -------------------------

def _sleep_jitter():
    time.sleep(random.uniform(*SLEEP_RANGE))

def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")

def _text(node: Optional[Tag]) -> Optional[str]:
    if not node:
        return None
    return node.get_text(strip=True)
def _non_empty(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        return val.strip() != ""
    return True

def _join_url(path: str) -> str:
    return urljoin(BASE, path)

def _episodes_url_from_show_url(show_url: str) -> str:
    return show_url.rstrip("/") + "/episodes"

def _clean_list(items: List[str]) -> List[str]:
    return [x.strip() for x in items if x and x.strip()]

def _safe_normalize_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    try:
        return normalize_date(date_str)
    except Exception:
        return date_str

def _norm_label(s: str) -> str:
    return re.sub(r":\s*$", "", s or "", flags=re.I).strip().lower()

def _select_one_with_classes(parent: Tag, classes: List[str]) -> Optional[Tag]:
    """
    在 parent 下查找包含 classes 中所有类名的第一个元素。
    """
    for el in parent.find_all(True):
        cl = set(el.get("class") or [])
        if set(classes).issubset(cl):
            return el
    return None


# -------------------------
# Parsers for 3 page types
# -------------------------

def parse_listing_page(html: str) -> List[str]:
    """解析列表页，提取节目详情页URL"""
    soup = _soup(html)
    urls: List[str] = []

    # 主要选择器：卡片标题里的详情链接
    for a in soup.select('div.card.primary.grid-x .content .title h2 a[href^="/shows/"]'):
        href = a.get("href") or ""
        # 仅保留 /shows/{id}/{slug} 格式的链接
        if re.fullmatch(r"/shows/\d+/.+", href):
            urls.append(_join_url(href))

    # 备用选择器：海报图上的链接（若主选择器未找到）
    if not urls:
        for a in soup.select('div.card.primary.grid-x figure.image a[href^="/shows/"]'):
            href = a.get("href") or ""
            if re.fullmatch(r"/shows/\d+/.+", href):
                urls.append(_join_url(href))

    # 去重并保持顺序
    urls = list(dict.fromkeys(urls))
    return urls


def parse_show_detail_page(html: str) -> Dict[str, Optional[str]]:
    """解析详情页，提取节目信息"""
    soup = _soup(html)

    # 标题
    title = _text(soup.select_one("h1.show-for-medium")) or _text(soup.select_one("h1"))

    # 右侧信息面板
    info_panel = soup.select_one("#general-info-panel")

    # 评分（AggregateRating）
    rating = None
    if info_panel:
        rating_node = info_panel.select_one('[itemprop="aggregateRating"] [itemprop="ratingValue"]')
        rating = _text(rating_node)

    # Network
    network = None
    if info_panel:
        for div in info_panel.find_all("div"):
            strong = div.find("strong")
            if not strong or not strong.text:
                continue
            label = _norm_label(strong.text)  
            if label in ("network", "web channel", "webchannel"):  
                a = div.find("a", href=True)
                network = _text(a)
                if network:
                    break
        if not network:
            net_a = info_panel.select_one('a[href^="/networks/"], a[href^="/webchannels/"]')
            network = _text(net_a)


    # Status  通用解析：在包含 <strong>Label:</strong> 的同辈元素中取值
    status = None
    if info_panel:
        for div in info_panel.find_all("div"):
            strong = div.find("strong")
            if not strong or not strong.text:
                continue
            label = _norm_label(strong.text)
            # 取 strong 所在父块的纯文本，去掉 label 前缀
            full_txt = div.get_text(" ", strip=True)
            val = re.sub(re.escape(strong.text), "", full_txt, count=1).strip()
            # 有些行后面跟括号/额外字段，这里对 Network 已单独处理，不在此处理
            if label == "status":
                status = val or status

    # Genres
    genres = None
    if info_panel:
        genres_container = None
        for div in info_panel.find_all("div"):
            strong = div.find("strong")
            if strong and _norm_label(strong.text) == "genres":
                genres_container = div
                break
        if genres_container:
            items = [ _text(x) for x in genres_container.select(".divider span") ]
            items = _clean_list(items)
            if items:
                genres = format_genres(items)

    summary = None
    summary_p = soup.select_one("#general-information article p")
    if summary_p:
        summary_html = str(summary_p)
        txt = strip_html(summary_html)
        if txt:
            if title:
                txt = re.sub(rf"^\s*{re.escape(title)}\s*", "", txt).strip()
            summary = txt

    return {
        "Title": title,
        "Rating": rating,
        "Genres": genres,
        "Status": status,
        "Network": network,
        "Summary": summary,
    }


from datetime import datetime, date
import re

# 英文月份日期，严格匹配，如 "Oct 30, 2025"
_DATE_RE = re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}$")

def _parse_en_us_date(s: str) -> Optional[str]:
    """把 'Oct 30, 2025' 转成 'YYYY-MM-DD'；不符合格式返回 None。"""
    s = (s or "").strip()
    if not _DATE_RE.match(s):
        return None
    try:
        dt = datetime.strptime(s, "%b %d, %Y")  # 先试短月
    except ValueError:
        try:
            dt = datetime.strptime(s, "%B %d, %Y")  # 再试全月
        except ValueError:
            return None
    return dt.strftime("%Y-%m-%d")

def parse_episodes_page(html: str) -> Tuple[Optional[str], Optional[str]]:
    """
    解析剧集页，返回 (first_air_date, end_date)：
    - 跳过 Specials
    - 只接受严格的 'Mon DD, YYYY' 日期
    - 如需只统计已播出，可开启过滤未来日期
    """
    soup = _soup(html)
    dates: List[str] = []

    for row in soup.select("article.episode-row"):
        # 1) 跳过 Specials：首列里有一个 <span class="special ...">S</span>
        first_col = row.select_one("div.small-1.cell")
        if first_col and first_col.select_one(".special"):
            continue  # 是 Specials

        # 2) 取日期列
        date_div = row.select_one("div.small-3.medium-2.cell")
        ds = _text(date_div)
        if not ds:
            continue

        # 3) 过滤非具体日期（TBA、占位等）
        if re.fullmatch(r"(?i)tba|to be announced", ds.strip()):
            continue

        iso = _parse_en_us_date(ds)
        if not iso:
            continue

        if iso > date.today().strftime("%Y-%m-%d"):
            continue

        dates.append(iso)

    if not dates:
        return (None, None)

    return (min(dates), max(dates))



# -------------------------
# Fetchers
# -------------------------

def fetch_listing_urls(session: requests.Session, page: int) -> List[str]:
    url = LISTING_TEMPLATE.format(page=page)
    html = get_html(session, url)
    _sleep_jitter()
    return parse_listing_page(html or "")

def fetch_show_detail(session: requests.Session, show_url: str) -> Dict[str, Optional[str]]:
    html = get_html(session, show_url)
    _sleep_jitter()
    return parse_show_detail_page(html or "")

def fetch_episodes_dates(session: requests.Session, episodes_url: str) -> Tuple[Optional[str], Optional[str]]:
    html = get_html(session, episodes_url)
    _sleep_jitter()
    return parse_episodes_page(html or "")


# -------------------------
# Orchestrators
# -------------------------

def _gather_all_show_urls(
    start_page: int,
    end_page: int,
    max_workers: Optional[int] = None,
) -> List[str]:
    """并行抓取列表页，收集所有节目URL"""
    session = get_session()
    pages = list(range(start_page, end_page + 1))
    all_urls: List[str] = []
    failed_pages: List[int] = []

    logging.info(f"Starting to gather show URLs from {len(pages)} listing pages...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_listing_urls, session, p): p for p in pages}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Listing pages"):
            p = futures[fut]
            try:
                urls = fut.result() or []
                all_urls.extend(urls)
            except Exception as e:
                logging.warning(f"[Listing] page={p} failed: {e}")
                failed_pages.append(p)

    # 去重并保持顺序
    all_urls = list(dict.fromkeys(all_urls))
    
    # 汇总统计
    success_rate = ((len(pages) - len(failed_pages)) / len(pages) * 100) if pages else 0
    logging.info(
        f"Collected {len(all_urls)} unique show URLs from pages {start_page}..{end_page} "
        f"(success rate: {success_rate:.1f}%, failed pages: {len(failed_pages)})"
    )
    
    if failed_pages and len(failed_pages) <= 10:
        logging.info(f"Failed pages: {failed_pages}")
    
    return all_urls


def _process_single_show(session: requests.Session, show_url: str) -> Optional[ShowRecord]:
    """处理单个节目，返回ShowRecord"""
    try:
        # 抓取详情页
        detail = fetch_show_detail(session, show_url)
        if not detail:
            logging.debug(f"[Show] No detail data for {show_url}")
            return None
        
        # 抓取剧集页
        episodes_url = _episodes_url_from_show_url(show_url)
        first_date, end_date = fetch_episodes_dates(session, episodes_url)

        # 组装记录
        rec = ShowRecord(
            title=detail.get("Title"),
            first_air_date=first_date,
            end_date=end_date,
            rating=detail.get("Rating"),
            genres=detail.get("Genres"),
            status=detail.get("Status"),
            network=detail.get("Network"),
            summary=detail.get("Summary"),
        )
        return rec
    except KeyboardInterrupt:
        # 允许用户中断
        raise
    except Exception as e:
        logging.warning(f"[Show] Failed to process {show_url}: {type(e).__name__}: {e}")
        return None


def transform_from_web(
    start_page: int,
    end_page: int,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    端到端：列表页 -> 详情页 -> 剧集页 -> DataFrame
    """
    show_urls = _gather_all_show_urls(start_page, end_page, max_workers=max_workers)

    session = get_session()
    records: List[ShowRecord] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_single_show, session, url): url for url in show_urls}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Show details"):
            url = futures[fut]
            try:
                rec = fut.result()
                if rec:
                    records.append(rec)
            except Exception as e:
                logging.warning(f"[Detail/Episodes] {url} failed: {e}")

    df = pd.DataFrame([r.to_row() for r in records])

    if not df.empty:
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[COLUMNS]
        # 可选：摘要截断，避免超长
        df["Summary"] = df["Summary"].apply(lambda x: truncate(x, 2000) if isinstance(x, str) else x)
    if df.empty:
        df = pd.DataFrame(columns=COLUMNS)
    else:
        completeness_mask = df[COLUMNS].apply(lambda s: s.map(_non_empty)).all(axis=1)
        dropped = len(df) - int(completeness_mask.sum())
        if dropped:
            logging.info(f"[Filter] Dropped {dropped} rows that failed 8-field completeness.")
        df = df[completeness_mask].copy()
    return df

# -------------------------
# Entrypoint
# -------------------------

def run(
    out_path: str,
    start_page: int = DEFAULT_START_PAGE,
    end_page: int = DEFAULT_END_PAGE,
    log_file: str = None,
    max_workers: int = None,
) -> pd.DataFrame:
    setup_logging(log_file)
    logging.info("=" * 60)
    logging.info("TVMaze Scraper (HTML) Started")
    logging.info(f"Parameters: range={start_page}..{end_page}, output={out_path}")

    if start_page < 1 or end_page < start_page:
        raise ValueError("Invalid page range")

    df = transform_from_web(start_page=start_page, end_page=end_page, max_workers=max_workers)
    save_csv(df, out_path)
    logging.info(f"SUCCESS: Saved {len(df)} rows to {out_path}")
    logging.info("=" * 60)
    return df
