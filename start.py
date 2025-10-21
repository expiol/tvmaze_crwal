# start.py
"""
Convenience entry point for scraping assignment dataset.
Parameters are defined here for clarity instead of command-line args.
"""
import os
from src.crwal.scraper import run

if __name__ == "__main__":
    COUNT = 200  # 抓取数量
    OUT_PATH = os.path.join("data", "YangHong+20251234.csv")  
    LOG_PATH = os.path.join("logs", "scrape.log")  

    # === 运行 ===
    run(COUNT, OUT_PATH, LOG_PATH)
