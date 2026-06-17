import os
import logging
from PIL import Image

# Must be set before paddleocr/paddlex is imported — disables oneDNN which
# crashes PaddlePaddle 3.x with ConvertPirAttribute2RuntimeAttribute error.
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR
from .image_preprocessor import load_image_for_ocr

for _name in ("ppocr", "paddleocr", "paddlex"):
    logging.getLogger(_name).setLevel(logging.ERROR)


class OCREngine:
    """PaddleOCR-based text extractor for medical document images.

    Single PaddleOCR instance shared across two public methods:
      - extract_text()     → raw text + metadata (used by BioBERT pipeline)
      - extract_ocr_data() → (words, boxes) normalized 0-1000 (used by LayoutLMv3)
    """

    def __init__(self):
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='en', device='gpu:0')

    def _run_predict(self, image_path):
        """Run PaddleOCR once and return (texts, scores, polys, width, height)."""
        validated_path = load_image_for_ocr(image_path)
        img = Image.open(validated_path)
        width, height = img.size

        result = list(self.ocr.predict(validated_path))
        if not result:
            return [], [], [], width, height

        page = result[0]
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("dt_polys") or []
        return texts, scores, polys, width, height

    def extract_text(self, image_path):
        """Extract text for the BioBERT pipeline.

        Returns a dict with:
          raw_text       – all text joined by spaces
          lines          – list of {text, confidence} dicts
          avg_confidence – mean line confidence as a percentage
          word_count     – total word count of raw_text
        """
        texts, scores, polys, _, _ = self._run_predict(image_path)

        if not texts:
            return {"raw_text": "", "lines": [], "avg_confidence": 0.0, "word_count": 0}

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
            "raw_text": raw_text,
            "lines": lines_output,
            "avg_confidence": round(avg_confidence, 2),
            "word_count": len(raw_text.split()),
        }

    def extract_ocr_data(self, image_path):
        """Extract words and bounding boxes for LayoutLMv3.

        Returns (words, boxes) where boxes are normalized to 0-1000.
        Falls back to [("", [0,0,0,0])] when OCR finds nothing.
        """
        texts, scores, polys, width, height = self._run_predict(image_path)
        words, boxes = [], []

        for text, poly in zip(texts, polys):
            if not text or poly is None:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

            x0_n = max(0, min(1000, int(x0 * 1000 / width)))
            y0_n = max(0, min(1000, int(y0 * 1000 / height)))
            x1_n = max(0, min(1000, int(x1 * 1000 / width)))
            y1_n = max(0, min(1000, int(y1 * 1000 / height)))

            words.append(str(text))
            boxes.append([x0_n, y0_n, x1_n, y1_n])

        if not words:
            words = [""]
            boxes = [[0, 0, 0, 0]]

        return words, boxes
