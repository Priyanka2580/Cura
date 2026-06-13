"""
Project Configuration
Stores shared paths and environment variable names for the Med-AI pipeline.
"""

MODEL_CLASSIFIER_PATH = "saved_models/classifier/best_classifier.pth"
MODEL_INFO_PATH = "saved_models/classifier/model_info.json"
MODEL_CLASSIFIER_INFO_PATH = "saved_models/classifier/model_info.json"
MODEL_NER_PATH = "saved_models/ner/biobert-medical-ner-final"
TRAINING_IMAGES_PATH = "datasets/training_images/"
NER_DATA_PATH = "datasets/ner_data/"
DRUG_DATABASE_PATH = "datasets/drug_database/"
TEST_IMAGES_PATH = "test_images/"
LOG_PATH = "logs/"
REFERENCE_SUMMARIES_PATH = "evaluation/reference_summaries.json"
TEST_RESULTS_PATH = "evaluation/test_results/"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
