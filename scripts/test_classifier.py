"""
Test the trained LayoutLMv3 classifier + KeywordValidator on test_images/.

Usage (from project root):
    python scripts/test_classifier.py
"""
import sys
import csv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.image_classifier import DocumentClassifier

TEST_IMAGES_DIR = Path("test_images")
LABELS_CSV      = Path("datasets/test_labels.csv")

LABEL_MAP = {
    "prescription": "prescription",
    "Prescription": "prescription",
    "report":       "report",
    "Report":       "report",
    "non_medical":  "non_medical",
    "Non-medical":  "non_medical",
    "Non_medical":  "non_medical",
}


def load_labels(csv_path):
    labels = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name  = row["image_name"].strip()
            label = LABEL_MAP.get(row["true_label"].strip(),
                                  row["true_label"].strip().lower())
            labels[name] = label
    return labels


def main():
    true_labels = load_labels(LABELS_CSV)

    images = sorted(
        p for p in TEST_IMAGES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    )

    if not images:
        print(f"No images found in {TEST_IMAGES_DIR.resolve()}")
        return

    print("Loading DocumentClassifier ...")
    clf = DocumentClassifier()
    validator_loaded = clf.validator is not None
    print(f"Model loaded.  KeywordValidator: {'ON' if validator_loaded else 'OFF (run build_keyword_lists.py first)'}")
    print(f"Testing on {len(images)} images.\n")

    header = f"{'Image':<22} {'True':<15} {'LMv3':<15} {'Final':<15} {'Ovr':>4}  {'Conf':>7}  Result"
    print(header)
    print("-" * 90)

    correct_lmv3  = 0
    correct_final = 0
    wrong_final   = []

    for img_path in images:
        true = true_labels.get(img_path.name, "UNKNOWN")
        try:
            result = clf.predict(str(img_path), validate=True)

            lmv3_pred  = result["validation"]["lmv3_prediction"] if result["validation"] else result["doc_type"]
            final_pred = result["doc_type"]
            conf       = result["confidence"]
            overridden = result["validation"]["overridden"] if result["validation"] else False
            ovr_flag   = "YES" if overridden else "-"

            match_lmv3  = lmv3_pred  == true
            match_final = final_pred == true
            result_flag = "OK" if match_final else "FAIL"

            if match_lmv3:
                correct_lmv3 += 1
            if match_final:
                correct_final += 1
            else:
                wrong_final.append((img_path.name, true, lmv3_pred, final_pred, conf))

            print(f"{img_path.name:<22} {true:<15} {lmv3_pred:<15} {final_pred:<15} {ovr_flag:>4}  {conf:>6.1f}%  {result_flag}")

        except Exception as exc:
            print(f"{img_path.name:<22} ERROR: {exc}")

    total = len(images)
    print("-" * 90)
    print(f"\nLMv3 only accuracy  : {correct_lmv3}/{total}  ({correct_lmv3/total*100:.1f}%)")
    print(f"Final accuracy      : {correct_final}/{total}  ({correct_final/total*100:.1f}%)")

    if wrong_final:
        print(f"\nMisclassified ({len(wrong_final)}):")
        for name, true, lmv3, final, conf in wrong_final:
            arrow = f"{lmv3} -> {final}" if lmv3 != final else lmv3
            print(f"  {name} — true: {true}, predicted: {arrow} ({conf:.1f}%)")
    else:
        print("\nAll images classified correctly!")

    if validator_loaded:
        overrides = [r for r in wrong_final if r[2] != r[3]]
        # count correct overrides by checking images NOT in wrong_final that were overridden
        print(f"\nKeyword validator overrides in wrong list: {len(overrides)}")


if __name__ == "__main__":
    main()
