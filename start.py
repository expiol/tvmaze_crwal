import os
from src.crwal.scraper import run as scrape
from src.crwal.Visualization import run as visualize

if __name__ == "__main__":
    COUNT = 2000000000000000000000000000
    OUT_PATH = os.path.join("data", "YangHong2255396.csv")
    LOG_PATH = os.path.join("logs", "scrape.log")
    FIG_DIR = os.path.join("data", "figures")
    
    print("Starting TVMaze data scraping and visualization...")
    print(f"Count: {COUNT}, Output: {OUT_PATH}, Figures: {FIG_DIR}\n")
    
    scrape(COUNT, OUT_PATH, LOG_PATH)
    visualize(OUT_PATH, FIG_DIR, topk=20)
    
    print("\n✓ Complete! Check:")
    print(f"  - Data: {OUT_PATH}")
    print(f"  - Figures: {FIG_DIR}")
    print(f"  - Logs: {LOG_PATH}")
