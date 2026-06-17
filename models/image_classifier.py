import json
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification

from config import MODEL_CLASSIFIER_PATH, MODEL_INFO_PATH, KEYWORD_LISTS_PATH
from models.ocr_engine import OCREngine
from models.keyword_validator import KeywordValidator


class DocumentClassifier:
    """LayoutLMv3 classifier for medical document type prediction.

    Classifies document images as one of: prescription, report, non_medical.
    Accepts a shared OCREngine instance so PaddleOCR is not loaded twice when
    the production pipeline already has one running.
    """

    def __init__(self, ocr_engine=None, use_validator=True):
        """Load model and processor.

        Args:
            ocr_engine:    An existing OCREngine instance to reuse. If None, a
                           new one is created internally.
            use_validator: If True, load the KeywordValidator for second-pass
                           validation. Set False to use LMv3 output only.
        """
        info_path = Path(MODEL_INFO_PATH)
        if not info_path.exists():
            raise FileNotFoundError(f"Model info not found: {MODEL_INFO_PATH}")

        with open(info_path) as f:
            model_info = json.load(f)

        self.class_names = model_info["class_names"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = Path(MODEL_CLASSIFIER_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_CLASSIFIER_PATH}")

        self.processor = LayoutLMv3Processor.from_pretrained(str(model_path), apply_ocr=False)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()

        self.ocr_engine = ocr_engine if ocr_engine is not None else OCREngine()

        if use_validator and Path(KEYWORD_LISTS_PATH).exists():
            self.validator = KeywordValidator(
                keyword_lists_path=KEYWORD_LISTS_PATH,
                ocr_engine=self.ocr_engine,
            )
        else:
            self.validator = None

    def predict(self, image_path, validate=True):
        """Predict document type for a given image file.

        Args:
            image_path: path to the image file
            validate:   if True and a KeywordValidator is loaded, run second-pass
                        keyword validation to confirm or override the LMv3 result

        Returns a dict with:
          doc_type          – final predicted class name
          confidence        – LMv3 confidence %
          all_probabilities – per-class probabilities as percentages
          validation        – keyword validation details (if validate=True)
        """
        image = Image.open(image_path).convert("RGB")
        words, boxes = self.ocr_engine.extract_ocr_data(image_path)

        encoding = self.processor(
            images=image,
            text=words,
            boxes=boxes,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
                bbox=encoding["bbox"].to(self.device),
                pixel_values=encoding["pixel_values"].to(self.device),
            )

        probs    = F.softmax(outputs.logits, dim=-1).squeeze(0)
        pred_idx = probs.argmax().item()

        lmv3_class = self.class_names[pred_idx]
        lmv3_conf  = round(probs[pred_idx].item() * 100, 2)

        result = {
            "doc_type":          lmv3_class,
            "confidence":        lmv3_conf,
            "all_probabilities": {
                name: round(probs[i].item() * 100, 2)
                for i, name in enumerate(self.class_names)
            },
            "validation": None,
        }

        if validate and self.validator is not None:
            val = self.validator.validate(image_path, lmv3_class, lmv3_conf)
            result["doc_type"]   = val["final_prediction"]
            result["validation"] = val

        return result
