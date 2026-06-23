"""
Phase 3 Pipeline (importable module)

Final pipeline: Classifier -> OCR -> BioBERT NER -> AbnormalityDetector
(reports only) -> Gemini (NER + abnormality-grounded prompts).

This is the importable .py counterpart of pipelines/phase3_pipeline.ipynb --
same logic, extracted so callers like app.py can do
`from pipelines.phase3_pipeline import run_phase3_pipeline` without needing
to execute a notebook. The notebook remains the place to run ad-hoc batches
via run_batch(); this module intentionally omits that batch-demo cell since
importing it should only load models, not kick off a run.

Models are loaded once at import time (module-level), so repeated calls to
run_phase3_pipeline within the same process reuse the already-loaded models.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import DocumentClassifier, OCREngine, NERExtractor
from models.image_preprocessor import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, BLUR_VARIANCE_THRESHOLD
from services.abnormality_detector import AbnormalityDetector
from services.gemini_summarizer_final import GeminiSummarizer

print("Loading OCREngine (PaddleOCR) ...")
ocr_engine = OCREngine()

print("Loading DocumentClassifier (LayoutLMv3) ...")
classifier = DocumentClassifier(ocr_engine=ocr_engine)

print("Loading NERExtractor (BioBERT) ...")
ner_extractor = NERExtractor()

print("Loading AbnormalityDetector ...")
abnormality_detector = AbnormalityDetector()

print("Loading GeminiSummarizer ...")
summarizer = GeminiSummarizer()

print("All models loaded.")


def check_image_quality(image_path):
    """Validate that an image can be opened and score its sharpness.

    Runs the same checks as models/image_preprocessor.load_image_for_ocr
    (file exists, extension allowed, size limit, not corrupted) but never
    raises on blur -- a blurry image is flagged via low_quality=True so the
    pipeline still attempts classification/OCR on it instead of rejecting
    it outright. If classification/OCR later fails anyway (e.g. the image
    is too blurry for PaddleOCR's own stricter check), that is caught as a
    normal pipeline failure further down.

    Returns a dict: valid, low_quality, blur_variance, error
    """
    path = Path(image_path)

    if not path.exists():
        return {"valid": False, "low_quality": False, "blur_variance": None,
                "error": f"Image file not found: {image_path}"}

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return {"valid": False, "low_quality": False, "blur_variance": None,
                "error": f"Unsupported file type '{path.suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}

    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        return {"valid": False, "low_quality": False, "blur_variance": None,
                "error": f"Image file too large ({file_size / 1_000_000:.1f} MB). "
                         f"Max allowed is {MAX_FILE_SIZE_BYTES / 1_000_000:.0f} MB."}

    try:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as e:
        return {"valid": False, "low_quality": False, "blur_variance": None,
                "error": f"Could not read image file: {e}"}

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blur_variance = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
    low_quality = blur_variance < BLUR_VARIANCE_THRESHOLD

    return {"valid": True, "low_quality": low_quality, "blur_variance": blur_variance, "error": None}


def call_gemini_prescription(raw_text, entities_by_type):
    """Summarize a prescription from raw OCR text + NER entities (no abnormality detection).

    Returns a dict: summary, doc_type, error (error is None on success).
    """
    return summarizer.summarize("prescription", raw_text, entities_by_type)


def call_gemini_report(raw_text, entities_by_type, abnormality_results, abnormality_summary):
    """Summarize a report from raw OCR text + NER entities + computed abnormality results.

    Returns a dict: summary, doc_type, error (error is None on success).
    """
    return summarizer.summarize(
        "report", raw_text, entities_by_type,
        abnormality_results=abnormality_results,
        abnormality_summary=abnormality_summary,
    )


def run_phase3_pipeline(image_path):
    """Run Phase 3 (Classifier -> OCR -> NER -> AbnormalityDetector [reports only] -> Gemini) on a single image."""
    filename = Path(image_path).name

    # 1. Preprocessing check
    quality = check_image_quality(image_path)
    if not quality["valid"]:
        return {"image": filename, "status": "failed", "message": quality["error"]}

    # 2. Classify
    try:
        classify_result = classifier.predict(image_path)
    except Exception as e:
        return {
            "image": filename,
            "status": "failed",
            "message": f"Classification/OCR failed: {e}",
            "low_quality": quality["low_quality"],
        }

    doc_type = classify_result["doc_type"]
    classifier_confidence = classify_result["confidence"]

    if doc_type == "non_medical":
        return {
            "image": filename,
            "status": "rejected",
            "message": "Please upload a valid medical document",
            "doc_type": "non_medical",
            "classifier_confidence": classifier_confidence,
            "low_quality": quality["low_quality"],
        }

    # 3. OCR text -- classifier.predict() already ran OCR internally (DocumentClassifier
    # shares the OCREngine instance), so reuse its raw_text instead of a second pass
    raw_text = classify_result["raw_text"]
    ocr_confidence = classify_result["ocr_avg_confidence"]
    ocr_word_count = classify_result["ocr_word_count"]

    # Covers fully blank OCR as well as handwritten/blurry/half-cut photos where OCR
    # read too little, too unreliably to trust for NER/Gemini -- ask for a clearer or
    # digital copy instead of summarizing unreliable/garbage text.
    if ocr_engine.is_low_quality(ocr_confidence, ocr_word_count):
        return {
            "image": filename,
            "status": "unreadable",
            "message": (
                "We couldn't read enough reliable text from this image. This can happen "
                "with blurry, handwritten, poorly lit, or partially captured documents. "
                "Please upload a clear, well-lit photo of the full document, or a digital "
                "copy if you have one."
            ),
            "doc_type": doc_type,
            "classifier_confidence": classifier_confidence,
            "ocr_confidence": ocr_confidence,
            "low_quality": quality["low_quality"],
        }

    # 4. NER -- entity_count=0 / entities=[] flows through unchanged if nothing matches
    ner_result = ner_extractor.extract_entities(raw_text)
    entities = ner_result["entities"]
    entity_count = ner_result["entity_count"]
    ner_avg_confidence = ner_result["avg_confidence"]
    entities_by_type = ner_extractor.get_entities_by_type(ner_result)

    # 5. Abnormality detection -- reports only
    abnormalities = None
    abnormality_results = None
    abnormality_summary = None
    if doc_type == "report":
        abnormality_results = abnormality_detector.detect(entities_by_type)
        abnormality_summary = abnormality_detector.get_summary(abnormality_results)
        abnormalities = {"results": abnormality_results, "summary": abnormality_summary}

    # 6. Gemini -- dispatch by doc_type
    if doc_type == "prescription":
        gemini_result = call_gemini_prescription(raw_text, entities_by_type)
    elif doc_type == "report":
        gemini_result = call_gemini_report(raw_text, entities_by_type, abnormality_results, abnormality_summary)
    else:
        gemini_result = {"summary": None, "doc_type": doc_type,
                          "error": f"No Gemini prompt for doc_type '{doc_type}'"}

    if gemini_result["error"]:
        return {
            "image": filename,
            "status": "failed",
            "message": gemini_result["error"],
            "doc_type": doc_type,
            "classifier_confidence": classifier_confidence,
            "ocr_confidence": ocr_confidence,
            "ner_avg_confidence": ner_avg_confidence,
            "entity_count": entity_count,
            "abnormalities": abnormalities,
            "low_quality": quality["low_quality"],
        }

    # 7. Result
    return {
        "image": filename,
        "status": "success",
        "doc_type": doc_type,
        "classifier_confidence": classifier_confidence,
        "ocr_confidence": ocr_confidence,
        "ner_avg_confidence": ner_avg_confidence,
        "entity_count": entity_count,
        "raw_text": raw_text,
        "entities": entities,
        "abnormalities": abnormalities,
        "summary": gemini_result["summary"],
        "low_quality": quality["low_quality"],
    }
