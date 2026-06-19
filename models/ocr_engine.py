import os
import logging

# Must be set before paddleocr/paddlex is imported — disables oneDNN which
# crashes PaddlePaddle 3.x with ConvertPirAttribute2RuntimeAttribute error.
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR
from .image_preprocessor import load_image_for_ocr

for _name in ("ppocr", "paddleocr", "paddlex"):
    logging.getLogger(_name).setLevel(logging.ERROR)


class OCREngine:
    """PaddleOCR-based text extractor for medical document images.

    Single PaddleOCR instance. Use extract_all() to get every downstream
    output (words/boxes for LayoutLMv3, raw_text for KeywordValidator and
    BioBERT NER) from one OCR pass. extract_text() and extract_ocr_data()
    remain as convenience wrappers for single-output callers.
    """

    def __init__(self):
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='en', device='gpu:0')

    def _run_predict(self, image_path):
        """Run PaddleOCR once and return (texts, scores, polys, width, height).

        load_image_for_ocr() decodes the image itself (instead of handing
        PaddleOCR a path) so unsupported formats, oversized files, sideways
        EXIF orientation, and blurry photos are caught before OCR runs.
        """
        image_array = load_image_for_ocr(image_path)
        height, width = image_array.shape[:2]

        result = list(self.ocr.predict(image_array))
        if not result:
            return [], [], [], width, height

        page = result[0]
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("dt_polys") or []
        return texts, scores, polys, width, height

    def extract_all(self, image_path):
        """Run PaddleOCR once and return everything needed by every downstream
        consumer — LayoutLMv3 classifier, KeywordValidator, and BioBERT NER —
        so a single image never triggers more than one OCR pass.

        Returns a dict with:
          words          – word list in original detection order (LayoutLMv3 input)
          boxes          – matching boxes normalized 0-1000 (LayoutLMv3 input)
          raw_text       – all text joined by spaces, y-sorted (NER / validator input)
          lines          – list of {text, confidence} dicts, y-sorted
          avg_confidence – mean line confidence as a percentage
          word_count     – total word count of raw_text
        """
        texts, scores, polys, width, height = self._run_predict(image_path)

        words, boxes = [], []
        for text, poly in zip(texts, polys):
            if not text or poly is None:
                continue
            # poly coordinates come back as numpy.int16 — cast to plain int
            # before scaling by 1000, otherwise x0 * 1000 silently overflows
            # the 16-bit range and corrupts the box.
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

            words.append(str(text))
            boxes.append([
                max(0, min(1000, int(x0 * 1000 / width))),
                max(0, min(1000, int(y0 * 1000 / height))),
                max(0, min(1000, int(x1 * 1000 / width))),
                max(0, min(1000, int(y1 * 1000 / height))),
            ])

        if not words:
            words = [""]
            boxes = [[0, 0, 0, 0]]

        lines_data = []
        for text, score, poly in zip(texts, scores, polys):
            if not text:
                continue
            y_coord = float(poly[0][1]) if poly is not None and len(poly) > 0 else 0.0
            lines_data.append({
                "text": str(text),
                "confidence": round(float(score), 4),
                "y_coord": y_coord,
            })

        lines_data.sort(key=lambda x: x["y_coord"])
        lines_output = [{"text": l["text"], "confidence": l["confidence"]} for l in lines_data]
        raw_text = " ".join(l["text"] for l in lines_data)
        avg_confidence = (
            sum(l["confidence"] for l in lines_data) / len(lines_data) * 100
            if lines_data else 0.0
        )

        return {
            "words": words,
            "boxes": boxes,
            "raw_text": raw_text,
            "lines": lines_output,
            "avg_confidence": round(avg_confidence, 2),
            "word_count": len(raw_text.split()),
        }

    def extract_text(self, image_path):
        """Extract text for the BioBERT pipeline.

        Returns a dict with:
          raw_text       – all text joined by spaces
          lines          – list of {text, confidence} dicts
          avg_confidence – mean line confidence as a percentage
          word_count     – total word count of raw_text
        """
        data = self.extract_all(image_path)
        return {
            "raw_text": data["raw_text"],
            "lines": data["lines"],
            "avg_confidence": data["avg_confidence"],
            "word_count": data["word_count"],
        }

    def extract_ocr_data(self, image_path):
        """Extract words and bounding boxes for LayoutLMv3.

        Returns (words, boxes) where boxes are normalized to 0-1000.
        Falls back to [("", [0,0,0,0])] when OCR finds nothing.
        """
        data = self.extract_all(image_path)
        return data["words"], data["boxes"]

    def is_low_quality(self, avg_confidence, word_count):
        """True when an OCR result looks unreliable enough that the caller
        should ask the user for a clearer photo instead of trusting it."""
        return avg_confidence < 60 and word_count < 20
