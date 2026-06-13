import os
import logging

# PaddleX reads PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT at import time to decide
# whether to default run_mode to "mkldnn" for CPU inference. When "mkldnn" is
# the run_mode, PaddleX calls config.enable_mkldnn() on the paddle inference
# Config, which activates oneDNN ops. Those ops are compiled into PIR, and
# PaddlePaddle 3.x PIR crashes with:
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#   [pir::ArrayAttribute<pir::DoubleAttribute>]
# Setting this to "0" forces run_mode="paddle" (plain CPU, no oneDNN).
# Must be set before paddleocr / paddlex is imported.
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR
from .image_preprocessor import load_image_for_ocr

for _name in ("ppocr", "paddleocr", "paddlex"):
    logging.getLogger(_name).setLevel(logging.ERROR)


class OCREngine:
    """PaddleOCR-based text extractor for medical document images.

    Handles rotated/tilted text via textline orientation classification.
    The OCR engine is initialized once and reused for every extract_text() call.
    """

    def __init__(self):
        """Initialize PaddleOCR with textline orientation classification.

        PaddleOCR 3.x renamed use_angle_cls → use_textline_orientation and
        removed show_log / use_gpu from the constructor.
        """
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='en'
        )

    def extract_text(self, image_path):
        """Extract text from an image file, sorted top-to-bottom by position.

        Returns a dict with:
          raw_text       – all text joined by spaces
          lines          – list of {text, confidence} dicts (confidence 0–1)
          avg_confidence – mean line confidence as a percentage
          word_count     – total word count of raw_text

        PaddleOCR 3.x returns OCRResult objects with parallel lists
        (rec_texts, rec_scores, dt_polys) rather than the 2.x
        [[[coords], ('text', conf)], ...] format.
        """
        validated_path = load_image_for_ocr(image_path)
        result = list(self.ocr.predict(validated_path))

        if not result:
            return {
                "raw_text": "",
                "lines": [],
                "avg_confidence": 0.0,
                "word_count": 0
            }

        page = result[0]
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("dt_polys") or []

        if not texts:
            return {
                "raw_text": "",
                "lines": [],
                "avg_confidence": 0.0,
                "word_count": 0
            }

        lines_data = []
        for text, score, poly in zip(texts, scores, polys):
            if not text:
                continue
            y_coord = float(poly[0][1]) if poly is not None and len(poly) > 0 else 0.0
            lines_data.append({
                "text": str(text),
                "confidence": round(float(score), 4),
                "y_coord": y_coord
            })

        lines_data.sort(key=lambda x: x["y_coord"])

        lines_output = [
            {"text": l["text"], "confidence": l["confidence"]}
            for l in lines_data
        ]
        raw_text = " ".join(l["text"] for l in lines_data)
        avg_confidence = (
            sum(l["confidence"] for l in lines_data) / len(lines_data) * 100
            if lines_data else 0.0
        )

        return {
            "raw_text": raw_text,
            "lines": lines_output,
            "avg_confidence": round(avg_confidence, 2),
            "word_count": len(raw_text.split())
        }
