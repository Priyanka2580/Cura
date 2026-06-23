"""
Run PaddleOCR on every training image ONCE and save results to datasets/ocr_cache.json.

Run this script before starting LayoutLMv3 training so the training loop reads
from cache instead of re-running OCR on each image every epoch.

Usage (from project root):
    python scripts/precompute_ocr.py

Resumes automatically if interrupted — already-processed images are skipped.
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ocr_engine import OCREngine
from config import TRAINING_IMAGES_PATH, OCR_CACHE_PATH

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def collect_images(root: Path):
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in VALID_EXTENSIONS
    )


def main():
    root = Path(TRAINING_IMAGES_PATH)
    if not root.exists():
        print(f"ERROR: Dataset path not found: {root.resolve()}")
        sys.exit(1)

    cache_path = Path(OCR_CACHE_PATH)
    existing_cache = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            existing_cache = json.load(f)
        print(f"Resuming — loaded existing cache: {len(existing_cache)} entries.")

    images = collect_images(root)
    print(f"Found {len(images)} images under {root.resolve()}")

    engine = OCREngine()
    cache = dict(existing_cache)
    new_count = 0
    error_count = 0

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def save_cache():
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    for img_path in tqdm(images, desc="Running OCR"):
        key = str(img_path.resolve())
        if key in cache:
            continue
        try:
            words, boxes = engine.extract_ocr_data(str(img_path))
            cache[key] = {"words": words, "boxes": boxes}
            new_count += 1
            save_cache()  # save after every image so Ctrl+C never loses work
        except Exception as exc:
            print(f"\nERROR on {img_path.name}: {exc}")
            error_count += 1

    print(f"\nDone.")
    print(f"  New entries   : {new_count}")
    print(f"  Errors        : {error_count}")
    print(f"  Total in cache: {len(cache)}")
    print(f"  Saved to      : {cache_path.resolve()}")


if __name__ == "__main__":
    main()
