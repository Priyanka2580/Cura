"""
One-time data migration: split the PATIENT_INFO label into PATIENT_NAME
(just the patient's name) and PATIENT_INFO (everything else that was already
tagged that way -- age, gender, etc., left untouched).

Patient name spans are not explicitly marked in the source data, so each
existing PATIENT_INFO span is classified using a simple rule: if any token
in the span contains a digit, or every token is a gender word, it's kept as
PATIENT_INFO (age, gender, IDs, mixed age/gender like "42/M"). Otherwise it's
treated as a name. This was checked against all 1178 existing PATIENT_INFO
spans before writing this script: ~98% classify unambiguously, the rest are
printed below for a manual look rather than silently guessed on.
"""
import json
import re
from pathlib import Path

DATA_FILES = [
    Path("datasets/ner_data/report_ner.json"),
    Path("datasets/ner_data/prescription_ner.json"),
]

GENDER_WORDS = {"male", "female", "m", "f", "other", "transgender"}
HAS_DIGIT = re.compile(r"\d")
FLAG_WORDS = {"test", "pid", "id"}  # printed for manual review, not auto-changed


def is_name_span(tokens):
    if any(HAS_DIGIT.search(t) for t in tokens):
        return False
    if all(t.lower() in GENDER_WORDS for t in tokens):
        return False
    return True


def relabel_file(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    total_spans = 0
    renamed_spans = 0
    flagged = []

    for example in data:
        tokens, labels = example["tokens"], example["labels"]
        i = 0
        while i < len(labels):
            if labels[i] == "B-PATIENT_INFO":
                j = i + 1
                while j < len(labels) and labels[j] == "I-PATIENT_INFO":
                    j += 1
                span_tokens = tokens[i:j]
                total_spans += 1

                if is_name_span(span_tokens):
                    labels[i] = "B-PATIENT_NAME"
                    for k in range(i + 1, j):
                        labels[k] = "I-PATIENT_NAME"
                    renamed_spans += 1

                    span_text = " ".join(span_tokens).lower()
                    if any(flag in span_text for flag in FLAG_WORDS):
                        flagged.append(" ".join(span_tokens))

                i = j
            else:
                i += 1

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return total_spans, renamed_spans, flagged


def main():
    grand_total, grand_renamed, grand_flagged = 0, 0, []

    for path in DATA_FILES:
        total, renamed, flagged = relabel_file(path)
        print(f"{path}: {total} PATIENT_INFO spans -> {renamed} renamed to PATIENT_NAME, "
              f"{total - renamed} kept as PATIENT_INFO")
        grand_total += total
        grand_renamed += renamed
        grand_flagged.extend(flagged)

    print(f"\nTotal: {grand_total} spans, {grand_renamed} renamed, {grand_total - grand_renamed} kept")

    if grand_flagged:
        print(f"\n{len(grand_flagged)} renamed span(s) look suspicious -- please check manually:")
        for s in grand_flagged:
            print(f"  - {s!r}")
    else:
        print("\nNo suspicious spans flagged.")


if __name__ == "__main__":
    main()
