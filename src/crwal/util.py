from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class Config:
    BASE_URL: str = "https://api.tvmaze.com"
    TIMEOUT: int = 15
    SLEEP_AFTER_REQ: float = 0.6
    MAX_RETRY: int = 3
    BACKOFF: float = 0.5
    SUMMARY_MAX_LEN: int = 280


def setup_logging(log_file: Optional[str] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
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
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


def safe_get_json(sess: requests.Session, url: str) -> Any:
    try:
        resp = sess.get(url, timeout=Config.TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        logging.warning(f"Non-200 response {resp.status_code} for {resp.url}")
    except requests.RequestException as e:
        logging.error(f"Request exception for {url}: {e}")
    finally:
        time.sleep(Config.SLEEP_AFTER_REQ)
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


def extract_network(item: Dict[str, Any]) -> str:
    net = item.get("network") or {}
    web = item.get("webChannel") or {}
    return (net.get("name") or web.get("name") or "").strip()


def format_genres(genres: List[str]) -> str:
    try:
        return json.dumps(genres, ensure_ascii=False)
    except Exception:
        return "[]"
