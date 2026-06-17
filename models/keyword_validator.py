"""
Keyword-based second-pass validator for LayoutLMv3 classifier output.

For each new image:
  1. Gets OCR words (from cache if available, else live OCR)
  2. Scores how many prescription / report keywords appear in the text
  3. Confirms or overrides the LMv3 prediction based on scores
"""
import json
from pathlib import Path

# Fraction of class keywords that must match to be eligible for override
OVERRIDE_SCORE_THRESHOLD = 0.15
# Candidate score must be this many times higher than the predicted class score
OVERRIDE_RATIO = 1.5


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

    def _score(self, text_lower, class_name):
        """Return fraction of class keywords found in text (0.0 – 1.0)."""
        keywords = self.keyword_lists.get(class_name, [])
        if not keywords:
            return 0.0
        hits = sum(1 for kw in keywords if kw in text_lower)
        return hits / len(keywords)

    # ── Public API ──────────────────────────────────────────────────────────

    def validate(self, image_path, lmv3_prediction, lmv3_confidence):
        """Confirm or override an LMv3 prediction using keyword evidence.

        Args:
            image_path       : path to the image file
            lmv3_prediction  : class predicted by LayoutLMv3
            lmv3_confidence  : confidence % from LayoutLMv3

        Returns a dict:
            final_prediction – confirmed or overridden class name
            lmv3_prediction  – original LMv3 prediction
            overridden       – True if the prediction was changed
            scores           – keyword match score per class (0-1)
            confidence       – original LMv3 confidence
        """
        words      = self._get_words(image_path)
        text_lower = " ".join(str(w).lower() for w in words)
        scores     = {cls: round(self._score(text_lower, cls), 4)
                      for cls in self._classes}

        final_prediction = lmv3_prediction
        overridden       = False

        current_score = scores.get(lmv3_prediction, 0.0)

        # Check every other non-non_medical class for a stronger keyword signal
        for candidate, candidate_score in scores.items():
            if candidate == lmv3_prediction or candidate == "non_medical":
                continue
            if (candidate_score >= OVERRIDE_SCORE_THRESHOLD and
                    candidate_score >= current_score * OVERRIDE_RATIO):
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
            if best and scores[best] >= OVERRIDE_SCORE_THRESHOLD:
                final_prediction = best
                overridden       = True

        return {
            "final_prediction": final_prediction,
            "lmv3_prediction":  lmv3_prediction,
            "overridden":       overridden,
            "scores":           scores,
            "confidence":       lmv3_confidence,
        }
