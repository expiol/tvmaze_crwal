# util.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class Config:
    """全局配置（仅用于 HTML 抓取）"""
    TIMEOUT: int = 15
    SLEEP_AFTER_REQ: float = 0.10   # 基础间隔（会再叠加 polite_sleep 的最小间隔与抖动）
    MAX_RETRY: int = 3              # urllib3 层面的幂等重试
    BACKOFF: float = 0.3            # urllib3 指数退避因子
    SUMMARY_MAX_LEN: int = 280
    MAX_WORKERS: int = 4            # 线程池并发（建议 3~4）
    MIN_INTERVAL: float = 0.7       # 全局最小请求间隔（秒）
    JITTER_MAX: float = 0.35        # 抖动上限（秒）
    PER_HOST_CONCURRENCY: int = 3   # 同一域名并发上限（信号量）


# 常见桌面浏览器 UA（Chrome/Firefox/Edge/Safari，Windows/macOS）
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

# 全局速率控制器
_PER_HOST_LOCK = threading.Semaphore(Config.PER_HOST_CONCURRENCY)
_LAST_REQUEST_TS = 0.0
_RATE_LOCK = threading.Lock()


def setup_logging(log_file: Optional[str] = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=handlers
    )


def get_session() -> requests.Session:
    """构建更像真实浏览器的 Session：随机 UA、常见头、Keep-Alive、连接池与幂等重试。"""
    sess = requests.Session()

    retry = Retry(
        total=Config.MAX_RETRY,
        read=Config.MAX_RETRY,
        connect=Config.MAX_RETRY,
        backoff_factor=Config.BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)

    ua = random.choice(USER_AGENTS)
    sess.headers.update({
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        # requests 自带 gzip/deflate/br 解压，无需手动 Accept-Encoding
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
    })
    return sess


def polite_sleep(min_interval: float | None = None, jitter: float | None = None) -> None:
    """全局最小间隔 + 随机抖动，避免尖峰；所有请求都会通过这里节流。"""
    global _LAST_REQUEST_TS
    if min_interval is None:
        min_interval = Config.MIN_INTERVAL
    if jitter is None:
        jitter = Config.JITTER_MAX

    with _RATE_LOCK:
        now = time.time()
        wait = _LAST_REQUEST_TS + min_interval - now
        if wait > 0:
            time.sleep(wait)
        # 基础间隔 + 额外抖动
        time.sleep(random.uniform(0, jitter))
        _LAST_REQUEST_TS = time.time()


def get_html(
    sess: requests.Session,
    url: str,
    referer: Optional[str] = None,
    *,
    max_attempts: int = 6,
    base_sleep: float = 0.9,
) -> Optional[str]:
    """
    仅用于 HTML 页面抓取；指数退避；尊重 Retry-After；带 Referer。
    返回 text 或 None。
    """
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        polite_sleep()  # 全局节流
        with _PER_HOST_LOCK:
            try:
                headers = {}
                if referer:
                    headers["Referer"] = referer
                resp = sess.get(url, timeout=Config.TIMEOUT, headers=headers)
            except requests.RequestException as e:
                logging.debug(f"[GET] exception on {url}: {e}")
                time.sleep(base_sleep * (2 ** (attempt - 1)) + random.uniform(0.2, 0.6))
                continue

        ct = resp.headers.get("Content-Type", "")
        if resp.status_code == 200 and ("text/html" in ct or ct.startswith("text/")):
            time.sleep(Config.SLEEP_AFTER_REQ + random.uniform(0, 0.05))
            return resp.text

        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    sleep_s = float(ra)
                except ValueError:
                    sleep_s = base_sleep * (2 ** (attempt - 1))
            else:
                sleep_s = base_sleep * (2 ** (attempt - 1))
            logging.warning(f"[429] {url} -> sleep {sleep_s:.2f}s")
            time.sleep(sleep_s + random.uniform(0.2, 0.8))
            continue

        if 500 <= resp.status_code < 600:
            sleep_s = base_sleep * (2 ** (attempt - 1))
            logging.warning(f"[{resp.status_code}] {url} -> backoff {sleep_s:.2f}s")
            time.sleep(sleep_s + random.uniform(0.2, 0.8))
            continue

        if resp.status_code in (403, 401):
            # 被强拦：再试一次，延长等待
            sleep_s = base_sleep * (2 ** (attempt - 1)) + 0.8
            logging.warning(f"[{resp.status_code}] {url} -> throttling {sleep_s:.2f}s")
            time.sleep(sleep_s + random.uniform(0.4, 1.2))
            continue

        logging.debug(f"[GET] non-200 {resp.status_code} on {url}")
        return None

    logging.error(f"[GET] failed after retries: {url}")
    return None


_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: Optional[str]) -> str:
    if not text:
        return ""
    clean = _TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", clean).strip()


def truncate(text: str, max_len: int = Config.SUMMARY_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 1].rstrip() + "…"


def normalize_date(date_str: Optional[str]) -> str:
    return date_str.replace("-", "/") if date_str else ""


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def format_genres(genres: list[str]) -> str:
    try:
        return json.dumps(genres, ensure_ascii=False)
    except Exception:
        return "[]"
