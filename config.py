"""
Project Configuration
Stores shared paths and environment variable names for the Med-AI pipeline.
"""

from pathlib import Path

# Anchored to this file's location so every path resolves correctly
# regardless of the caller's working directory (e.g. a notebook running
# from evaluation/ instead of the project root).
_PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_CLASSIFIER_PATH = str(_PROJECT_ROOT / "saved_models/classifier/layoutlmv3-classifier")
MODEL_INFO_PATH = str(_PROJECT_ROOT / "saved_models/classifier/layoutlmv3-classifier/model_info.json")
MODEL_CLASSIFIER_INFO_PATH = str(_PROJECT_ROOT / "saved_models/classifier/layoutlmv3-classifier/model_info.json")
MODEL_NER_PATH = str(_PROJECT_ROOT / "saved_models/ner/biobert-medical-ner-final")
TRAINING_IMAGES_PATH = str(_PROJECT_ROOT / "datasets/training_images") + "/"
NER_DATA_PATH = str(_PROJECT_ROOT / "datasets/ner_data") + "/"
DRUG_DATABASE_PATH = str(_PROJECT_ROOT / "datasets/drug_database") + "/"
TEST_IMAGES_PATH = str(_PROJECT_ROOT / "datasets/test_images") + "/"
LOG_PATH = str(_PROJECT_ROOT / "logs") + "/"
REFERENCE_SUMMARIES_PATH = str(_PROJECT_ROOT / "evaluation/reference_summaries.json")
TEST_RESULTS_PATH = str(_PROJECT_ROOT / "evaluation/test_results") + "/"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OCR_CACHE_PATH = str(_PROJECT_ROOT / "datasets/ocr_cache.json")
KEYWORD_LISTS_PATH = str(_PROJECT_ROOT / "datasets/keyword_lists.json")
