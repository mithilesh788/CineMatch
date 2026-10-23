from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEST = RAW / "ml-100k"
ARCHIVE = RAW / "ml-100k.zip"
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if (DEST / "u.data").exists() and (DEST / "u.item").exists():
        print(f"Dataset already exists: {DEST}")
        return
    print("Downloading MovieLens 100K...")
    urlretrieve(URL, ARCHIVE)
    with ZipFile(ARCHIVE) as zf:
        zf.extractall(RAW)
    print(f"Dataset ready: {DEST}")

if __name__ == "__main__":
    main()
