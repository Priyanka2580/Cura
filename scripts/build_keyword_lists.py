"""
One-time script: mine distinctive keywords per class from the OCR cache.
Run from project root:  python scripts/build_keyword_lists.py
Output: datasets/keyword_lists.json

Selection rule:
  A word qualifies as a class keyword if:
    - it appears in >= PRESENCE_THRESHOLD of that class's images
    - its class_frequency / max(other_class_frequencies) >= DISTINCTIVENESS_RATIO
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OCR_CACHE_PATH, KEYWORD_LISTS_PATH

CLASS_NAMES            = ["non_medical", "prescription", "report"]
PRESENCE_THRESHOLD     = 0.15   # word must appear in >=15% of its class images
DISTINCTIVENESS_RATIO  = 2.5    # class_freq must be 2.5x any other class_freq
MIN_WORD_LENGTH        = 2      # allow short medical abbrevs: mg, bd, od

STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "this", "that",
    "from", "have", "has", "had", "not", "but", "they", "their", "will",
    "been", "would", "could", "should", "may", "can", "all", "any",
    "per", "each", "also", "than", "then", "into", "out", "our",
    "you", "your", "his", "her", "its", "who", "which",
    "pvt", "ltd", "reg", "nos", "ref",
    "hospital", "clinic", "patient", "doctor", "centre", "center",
    "india", "phone", "email", "address", "road", "street",
    "private", "limited", "registered",
    "name", "age", "sex", "date", "time", "page", "no",
    "mr", "mrs", "ms", "dr", "sri", "smt",
    "east", "west", "north", "south",
}

# Hardcoded anchor keywords added regardless of frequency stats
# (well-known medical terms that may not meet frequency threshold)
ANCHOR_KEYWORDS = {
    "prescription": [
        "tablet", "capsule", "syrup", "injection", "ointment", "drops",
        "dose", "dosage", "medication", "medicine", "drug",
        "once", "twice", "thrice", "daily", "morning", "evening", "night",
        "empty", "stomach", "food", "meal", "bedtime",
        "mg", "ml", "mcg", "gm",
        "bd", "od", "tds", "qid", "sos", "prn", "stat",
        "refill", "dispense", "sig",
    ],
    "report": [
        "hemoglobin", "haemoglobin", "platelet", "leucocyte", "erythrocyte",
        "wbc", "rbc", "hct", "mcv", "mch",
        "creatinine", "urea", "sodium", "potassium", "chloride",
        "glucose", "cholesterol", "triglyceride", "bilirubin",
        "albumin", "protein", "calcium", "phosphorus",
        "specimen", "sample", "serum", "plasma",
        "normal", "abnormal", "reference", "range",
        "positive", "negative", "reactive",
        "impression", "findings", "conclusion",
        "test", "analysis", "result",
        "pathology", "laboratory", "radiology",
    ],
    "non_medical": [],
}


def tokenize(words):
    """Return a set of clean lowercase tokens from an OCR word list."""
    tokens = set()
    for word in words:
        word = str(word).lower().strip()
        # pure alpha (2+ chars) or alphanumeric starting with letter (3+ chars)
        if (re.match(r'^[a-z]{2,}$', word) or
                re.match(r'^[a-z][a-z0-9]{2,}$', word)):
            if word not in STOPWORDS:
                tokens.add(word)
    return tokens


def main():
    cache_path = Path(OCR_CACHE_PATH)
    if not cache_path.exists():
        print(f"ERROR: OCR cache not found at {cache_path.resolve()}")
        sys.exit(1)

    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    print(f"OCR cache loaded : {len(cache)} entries\n")

    # ── Group images by class ───────────────────────────────────────────────
    class_images = {cls: [] for cls in CLASS_NAMES}
    for path_str, data in cache.items():
        class_name = Path(path_str).parent.name
        if class_name not in CLASS_NAMES:
            continue
        tokens = tokenize(data.get("words", []))
        class_images[class_name].append(tokens)

    for cls in CLASS_NAMES:
        print(f"  {cls:15s}: {len(class_images[cls])} images in cache")

    # ── Compute per-class word frequency ────────────────────────────────────
    class_word_freq = {}
    for cls in CLASS_NAMES:
        images = class_images[cls]
        n = len(images)
        counter = defaultdict(int)
        for img_tokens in images:
            for word in img_tokens:
                counter[word] += 1
        class_word_freq[cls] = {w: c / n for w, c in counter.items()}

    # ── Select distinctive keywords ─────────────────────────────────────────
    keyword_lists = {}
    for cls in CLASS_NAMES:
        others = [c for c in CLASS_NAMES if c != cls]
        mined  = []

        for word, freq in class_word_freq[cls].items():
            if freq < PRESENCE_THRESHOLD:
                continue
            max_other = max(
                class_word_freq[other].get(word, 0.0) for other in others
            )
            # avoid division by zero
            ratio = freq / max_other if max_other > 0 else float("inf")
            if ratio >= DISTINCTIVENESS_RATIO:
                mined.append((word, round(freq, 3), round(ratio, 1)))

        mined.sort(key=lambda x: -x[2])   # sort by distinctiveness ratio

        # combine mined + anchor keywords (anchors go first as they are reliable)
        anchors  = ANCHOR_KEYWORDS.get(cls, [])
        mined_words = [w for w, _, _ in mined]
        combined = anchors + [w for w in mined_words if w not in set(anchors)]

        keyword_lists[cls] = combined

        print(f"\n{cls} — {len(anchors)} anchors + {len(mined)} mined = {len(combined)} total")
        print(f"  Top mined keywords:")
        for word, freq, ratio in mined[:15]:
            print(f"    {word:25s}  freq={freq:.2f}  ratio={ratio:.1f}x")

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = Path(KEYWORD_LISTS_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(keyword_lists, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path.resolve()}")
    total = sum(len(v) for v in keyword_lists.values())
    print(f"Total keywords : {total}  "
          f"({len(keyword_lists['prescription'])} prescription, "
          f"{len(keyword_lists['report'])} report, "
          f"{len(keyword_lists['non_medical'])} non_medical)")


if __name__ == "__main__":
    main()
