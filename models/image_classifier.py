import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path

from config import MODEL_CLASSIFIER_PATH, MODEL_INFO_PATH
from .image_preprocessor import preprocess_for_classifier


class DocumentClassifier:
    """EfficientNet-B0 classifier for medical document type prediction.

    Classifies document images as one of: prescription, report, non_medical.
    Model weights and class names are loaded once during initialization and
    reused for every predict() call.
    """

    def __init__(self):
        """Load model architecture, weights, and class names from disk."""
        info_path = Path(MODEL_INFO_PATH)
        if not info_path.exists():
            raise FileNotFoundError(f"Model info file not found: {MODEL_INFO_PATH}")

        with open(info_path, "r") as f:
            model_info = json.load(f)

        self.class_names = model_info["class_names"]
        num_classes = len(self.class_names)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mirror the exact architecture used during training
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

        weights_path = Path(MODEL_CLASSIFIER_PATH)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {MODEL_CLASSIFIER_PATH}")

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model

    def predict(self, image_path):
        """Predict document type for a given image file.

        Returns a dict with:
          doc_type         – predicted class name
          confidence       – confidence of the top prediction as a percentage
          all_probabilities – per-class probabilities as percentages
        """
        tensor = preprocess_for_classifier(image_path).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        pred_idx = probs.argmax().item()

        return {
            "doc_type": self.class_names[pred_idx],
            "confidence": round(probs[pred_idx].item() * 100, 2),
            "all_probabilities": {
                name: round(probs[i].item() * 100, 2)
                for i, name in enumerate(self.class_names)
            }
        }
