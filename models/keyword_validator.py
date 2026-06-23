"""
Keyword-based second-pass validator for LayoutLMv3 classifier output.

For each new image:
  1. Gets OCR words (from cache if available, else live OCR)
  2. Scores how many prescription / report keywords appear in the text
  3. Confirms or overrides the LMv3 prediction based on scores
"""
import json
import re
from pathlib import Path

# Fraction of class keywords that must match to be eligible for override
OVERRIDE_SCORE_THRESHOLD = 0.15
# Candidate score must be this many times higher than the predicted class score
OVERRIDE_RATIO = 1.5
# Absolute keyword-hit floor that can override on its own, regardless of the
# fraction above -- rescues short documents (e.g. a 2-medicine specialist
# note) that have strong specific evidence but too few hits to clear
# OVERRIDE_SCORE_THRESHOLD against the full keyword list.
MIN_OVERRIDE_HITS = 2


class KeywordValidator:
    """Validates or overrides a LayoutLMv3 prediction using keyword scoring."""

    def __init__(self, keyword_lists_path, ocr_engine=None, ocr_cache_path=None):
        with open(keyword_lists_path, encoding="utf-8") as f:
            self.keyword_lists = json.load(f)

        self.ocr_engine     = ocr_engine
        self.ocr_cache_path = ocr_cache_path
        self._cache         = None

        self._classes = list(self.keyword_lists.keys())

    # ── Internal helpers ────────────────────────────────────────────────────

    def _load_cache(self):
        if self._cache is not None:
            return
        path = Path(self.ocr_cache_path) if self.ocr_cache_path else Path("datasets/ocr_cache.json")
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self._cache = json.load(f)
        else:
            self._cache = {}

    def _get_words(self, image_path):
        """Return OCR word list — from cache if available, else live OCR."""
        self._load_cache()
        key = str(Path(image_path).resolve())
        if key in self._cache:
            return self._cache[key].get("words", [])
        if self.ocr_engine is not None:
            words, _ = self.ocr_engine.extract_ocr_data(image_path)
            return words
        return []

    def _hits(self, text_lower, class_name):
        """Return (hit_count, total_keywords) for a class's list against text.

        Matches require a non-letter (or string edge) on both sides instead of
        plain substring containment -- this lets unit abbreviations glued to a
        number still match (e.g. "mg" in "800mg") while blocking abbreviations
        from firing inside unrelated longer words (e.g. "od" in "good"/"food",
        "dr" in "address", "bp" in "kbps").
        """
        keywords = self.keyword_lists.get(class_name, [])
        if not keywords:
            return 0, 0
        hits = sum(
            1 for kw in keywords
            if re.search(rf"(?<![A-Za-z]){re.escape(kw)}(?![A-Za-z])", text_lower)
        )
        return hits, len(keywords)

    def _score(self, text_lower, class_name):
        """Return fraction of class keywords found in text (0.0 – 1.0)."""
        hits, total = self._hits(text_lower, class_name)
        return round(hits / total, 4) if total else 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    def validate(self, image_path, lmv3_prediction, lmv3_confidence, words=None):
        """Confirm or override an LMv3 prediction using keyword evidence.

        Args:
            image_path       : path to the image file
            lmv3_prediction  : class predicted by LayoutLMv3
            lmv3_confidence  : confidence % from LayoutLMv3
            words            : pre-extracted OCR words, if the caller already
                                ran OCR (e.g. DocumentClassifier). Skips the
                                cache/live-OCR lookup when provided.

        Returns a dict:
            final_prediction – confirmed or overridden class name
            lmv3_prediction  – original LMv3 prediction
            overridden       – True if the prediction was changed
            scores           – keyword match score per class (0-1)
            hit_counts       – raw keyword hit count per class
            confidence       – original LMv3 confidence
        """
        if words is None:
            words = self._get_words(image_path)
        text_lower = " ".join(str(w).lower() for w in words)

        hit_counts = {}
        scores = {}
        for cls in self._classes:
            hits, total = self._hits(text_lower, cls)
            hit_counts[cls] = hits
            scores[cls] = round(hits / total, 4) if total else 0.0

        final_prediction = lmv3_prediction
        overridden       = False

        current_score = scores.get(lmv3_prediction, 0.0)
        current_hits  = hit_counts.get(lmv3_prediction, 0)

        # Check every other non-non_medical class for a stronger keyword signal,
        # either as a fraction of its list or as a larger absolute hit count
        # (rescues short documents where the fraction never clears the threshold)
        for candidate, candidate_score in scores.items():
            if candidate == lmv3_prediction or candidate == "non_medical":
                continue
            strong_fraction = (candidate_score >= OVERRIDE_SCORE_THRESHOLD and
                               candidate_score >= current_score * OVERRIDE_RATIO)
            strong_absolute = (hit_counts[candidate] >= MIN_OVERRIDE_HITS and
                               hit_counts[candidate] > current_hits)
            if strong_fraction or strong_absolute:
                final_prediction = candidate
                overridden       = True
                break

        # If LMv3 said non_medical, check if keywords strongly suggest a medical class
        if lmv3_prediction == "non_medical":
            best = max(
                (c for c in scores if c != "non_medical"),
                key=lambda c: scores[c],
                default=None,
            )
            if best and (scores[best] >= OVERRIDE_SCORE_THRESHOLD or
                         hit_counts[best] >= MIN_OVERRIDE_HITS):
                final_prediction = best
                overridden       = True

        # Zero-evidence safety net: a real prescription/report always matches at least
        # one weak keyword (e.g. "mg", "tablet", "test"). If LMv3 landed on a medical
        # class but the OCR text matches NEITHER medical keyword list at all, there is
        # no textual evidence this is a medical document (e.g. a selfie or screenshot
        # LMv3 confidently but wrongly classified as a report). The override loop above
        # can't reach this case since it deliberately excludes non_medical as a candidate.
        if final_prediction in ("prescription", "report"):
            medical_scores = {c: s for c, s in scores.items() if c != "non_medical"}
            if all(s == 0.0 for s in medical_scores.values()):
                final_prediction = "non_medical"
                overridden       = True

        return {
            "final_prediction": final_prediction,
            "lmv3_prediction":  lmv3_prediction,
            "overridden":       overridden,
            "scores":           scores,
            "hit_counts":       hit_counts,
            "confidence":       lmv3_confidence,
        }
