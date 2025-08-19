import os
import requests
from bs4 import BeautifulSoup

# Processing config
DOCS_URL = "https://fastapi.tiangolo.com/"
RAW_DATA_DIR = "../data/raw/"
PROCESSED_DATA_DIR = "../data/processed/"

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Scraping data from fastapi docs
def scrape(url=DOCS_URL, path=RAW_DATA_DIR + "fastapi_main.html"):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "w", encoding="utf-8") as w:
            w.write(response.text)
        print("raw data saved")
    else:
        raise Exception(f"failed to fetch docs {response.status_code}")


# Cleaning the raw html to text
def clean(path_in = RAW_DATA_DIR + "fastapi_main.html"):
    with open(path_in, "r", encoding="utf-8") as r:
        soup = BeautifulSoup(r,"html.parser")

    # remove excess tags
    for tag in soup(["nav", "footer", "script", "style"]):
        tag.decompose()

    # extract content
    content = soup.find("main")
    if content == None:
        content = soup

    text = content.get_text(separator="\n")

    # remove blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    path_out = PROCESSED_DATA_DIR + "fastapi.clean.txt"
    with open(path_out, "w", encoding="utf-8") as w:
        w.write(clean_text)
    print("clean data saved")
