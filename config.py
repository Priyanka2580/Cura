"""
Project Configuration
Stores shared paths and environment variable names for the Med-AI pipeline.
"""

from pathlib import Path

# Anchored to this file's location so every path resolves correctly
# regardless of the caller's working directory (e.g. a notebook running
# from evaluation/ instead of the project root).
_PROJECT_ROOT = Path(__file__).resolve().parent

# Trained weights are too large for git (model.safetensors files exceed
# GitHub's 100MB limit), so they're hosted on Hugging Face Hub instead of
# saved_models/ and pulled down via from_pretrained() at runtime. Replace
# these with your own HF username/repo once you've uploaded the checkpoints.
MODEL_CLASSIFIER_REPO = "Dpriyanka/cura-layoutlmv3-classifier"
MODEL_NER_REPO = "Dpriyanka/cura-biobert-ner"
HF_TOKEN_ENV = "HF_TOKEN"
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
