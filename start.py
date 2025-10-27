# start.py
import os
import sys
import datetime

from src.crwal.scraper import run as scrape


if __name__ == "__main__":
    # =======================
    # 基础配置
    # =======================
    LIMIT = 200
    START_PAGE = 1
    LOG_PATH = os.path.join("logs", "scrape.log")

    today_str = datetime.date.today().isoformat()
    DATA_DIR = os.path.join("data", today_str)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    OUT_PATH = os.path.join(DATA_DIR, f"YangHong2255396_{LIMIT}.csv")

    print("\nStarting TVMaze HTML scraping...")
    print(f"Mode: limit-mode  | start_page: {START_PAGE} | limit: {LIMIT}")
    print(f"Output CSV: {OUT_PATH}")
    print(f"Log file:   {LOG_PATH}\n")

    # =======================
    # 执行爬虫
    # =======================
    try:
        df = scrape(
            out_path=OUT_PATH,
            start_page=START_PAGE,
            log_file=LOG_PATH,
            max_workers=10,
            limit=LIMIT,  
        )

        print("\n✓ Complete! Results:")
        print(f"  - Data saved to: {OUT_PATH}")
        print(f"  - Logs at:       {LOG_PATH}")
        print(f"  - Total shows scraped (after completeness filter): {len(df)}")

        if len(df) >= LIMIT:
            print(f"  - Reached LIMIT={LIMIT}, stopped early.")
        else:
            print(f"  - Reached last page before hitting LIMIT={LIMIT}.")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Scraping failed: {e}")
        sys.exit(1)
