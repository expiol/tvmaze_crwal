# test_max_page.py
"""
Test script: find the maximum valid page number for TVMaze /shows?page=N
It tries increasing page numbers until an empty response is returned.
"""

import requests
import time

BASE_URL = "https://api.tvmaze.com/shows?page="
TIMEOUT = 10
SLEEP = 0.4  # polite delay to avoid rate limit

def find_max_page():
    page = 0
    while True:
        url = f"{BASE_URL}{page}"
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code != 200:
                print(f"[!] Non-200 ({resp.status_code}) at page {page}")
                break

            data = resp.json()
            if not data:
                print(f"\n✅ Empty page at {page} — stop.")
                break

            print(f"Page {page}: {len(data)} shows")
            page += 1
            time.sleep(SLEEP)

        except Exception as e:
            print(f"[ERROR] {e}")
            break

    print(f"\n📊 Maximum valid page index ≈ {page - 1}")
    print(f"Total pages fetched: {page}")
    return page - 1


if __name__ == "__main__":
    print("🔍 Testing TVMaze /shows?page=N for maximum page number...\n")
    max_page = find_max_page()
    print(f"\n🎯 Result: The last non-empty page is {max_page}")
  