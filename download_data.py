import csv
import os
import urllib.request
from urllib.parse import urljoin

BASE_URL = "http://hyperscale-data.s3-website.ap-northeast-2.amazonaws.com/data/"
CSV_FILE = "sampled_files_20251128_012344.csv"
DATA_DIR = "data"

def download_file(url, local_path):
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            print(f"Downloading {url} to {local_path}")
            urllib.request.urlretrieve(url, local_path)
        else:
            print(f"Skipping {local_path} (already exists)")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file '{CSV_FILE}' not found.")
        return

    print(f"Reading from {CSV_FILE}...")
    
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Download JSON
            json_path = row.get('s3_json_path')
            if json_path:
                full_url = urljoin(BASE_URL, json_path)
                local_path = os.path.join(DATA_DIR, json_path)
                download_file(full_url, local_path)

            # Download Image
            image_path = row.get('s3_image_path')
            if image_path:
                full_url = urljoin(BASE_URL, image_path)
                local_path = os.path.join(DATA_DIR, image_path)
                download_file(full_url, local_path)

    print("Download process completed.")

if __name__ == "__main__":
    main()
