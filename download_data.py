import requests
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Saved {filename}")

if __name__ == "__main__":
    base_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"
    files = ["books.csv", "ratings.csv"]
    
    for file in files:
        download_file(base_url + file, file)
    
    print("Download complete.")
