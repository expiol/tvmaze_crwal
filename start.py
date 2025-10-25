import os
import datetime
from src.crwal.scraper import run as scrape
from src.crwal.Visualization import run as visualize

if __name__ == "__main__":
    # =======================
    # 基础配置
    # =======================
    START_PAGE = 1
    END_PAGE = 10
    LOG_PATH = os.path.join("logs", "scrape.log")

    # 当天日期（例如 '2025-10-24'）
    today_str = datetime.date.today().isoformat()

    # 自动生成 data/日期 目录
    DATA_DIR = os.path.join("data", today_str)
    FIG_DIR = os.path.join(DATA_DIR, "figures")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    OUT_PATH = os.path.join(DATA_DIR, "YangHong2255396.csv")

    print("Starting TVMaze HTML scraping and visualization...")
    print(f"Page range: {START_PAGE} → {END_PAGE}")
    print(f"Output CSV: {OUT_PATH}")
    print(f"Figures dir: {FIG_DIR}\n")

    # =======================
    # 执行爬虫与可视化
    # =======================
    try:
        df = scrape(
            out_path=OUT_PATH,
            start_page=START_PAGE,
            end_page=END_PAGE,
            log_file=LOG_PATH,
            max_workers=10
        )

        visualize(OUT_PATH, FIG_DIR, topk=20)

        print("\n✓ Complete! Check results:")
        print(f"  - Data saved to: {OUT_PATH}")
        print(f"  - Figures saved in: {FIG_DIR}")
        print(f"  - Logs at: {LOG_PATH}")
        print(f"  - Total shows scraped: {len(df)}")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Scraping failed: {e}")
