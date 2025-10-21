import os
from src.crwal.scraper import run as scrape
from src.crwal.Visualization import run as visualize

if __name__ == "__main__":
    COUNT = 200
    OUT_PATH = os.path.join("data", "YangHong2255396.csv")
    LOG_PATH = os.path.join("logs", "scrape.log")
    FIG_DIR = os.path.join("data", "figures")
    
    # 配置选项
    USE_EPISODES_API = False  # 是否调用episodes API（True=慢但完整，False=快速）
    FETCH_NEWEST = True       # 是否获取最新节目（True=最新，False=最旧）
    
    print("Starting TVMaze data scraping and visualization...")
    print(f"Count: {COUNT}, Output: {OUT_PATH}, Figures: {FIG_DIR}")
    print(f"Mode: {'Newest' if FETCH_NEWEST else 'Oldest'} shows, Episodes API: {USE_EPISODES_API}\n")
    
    scrape(COUNT, OUT_PATH, LOG_PATH, use_episodes_api=USE_EPISODES_API, fetch_newest=FETCH_NEWEST)
    visualize(OUT_PATH, FIG_DIR, topk=20)
    
    print("\n✓ Complete! Check:")
    print(f"  - Data: {OUT_PATH}")
    print(f"  - Figures: {FIG_DIR}")
    print(f"  - Logs: {LOG_PATH}")
