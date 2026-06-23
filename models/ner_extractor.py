import json
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

from config import MODEL_NER_PATH


class NERExtractor:
    """BioBERT-based Named Entity Recognizer for medical text.

    Extracts structured medical entities (drug names, dosages, doctor names,
    test names, etc.) from raw OCR text. Tokenizer and model are loaded once
    during initialization and reused for every extract_entities() call.
    """

    def __init__(self):
        """Load tokenizer, model, and label mappings from MODEL_NER_PATH."""
        model_path = Path(MODEL_NER_PATH)

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))

        label_file = model_path / "label_mappings.json"
        if not label_file.exists():
            raise FileNotFoundError(f"Label mappings not found: {label_file}")

        with open(label_file, "r") as f:
            label_mappings = json.load(f)

        # JSON keys are always strings; convert to int for id2label lookup
        self.id2label = {int(k): v for k, v in label_mappings["id2label"].items()}
        self.label2id = label_mappings["label2id"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def extract_entities(self, text):
        """Extract medical named entities from input text.

        Handles BERT subword tokens (## prefix) by reconstructing full words
        before grouping consecutive B-/I- tagged words into entity spans.

        Returns a dict with:
          entities        – list of {entity_type, text, confidence} dicts
          entity_count    – number of entities found
          avg_confidence  – mean entity confidence as a percentage
          raw_predictions – token-level {token, label, confidence} list
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )

        # offset_mapping is not accepted by the model; pop before forwarding
        encoding.pop("offset_mapping")
        input_ids = encoding["input_ids"].squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())

        model_input = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**model_input)

        logits = outputs.logits.squeeze(0)
        probs = F.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1).tolist()
        confidences = probs.max(dim=-1).values.tolist()

        special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }
        special_ids.discard(None)

        raw_predictions = []
        # Each entry: (word_string, label, avg_confidence_0_to_1)
        words = []
        current_word = None
        current_label = "O"
        current_confs = []

        for token_id, token, pred_id, conf in zip(
            input_ids.tolist(), tokens, pred_ids, confidences
        ):
            if token_id in special_ids:
                if current_word is not None:
                    words.append((current_word, current_label,
                                  sum(current_confs) / len(current_confs)))
                    current_word = None
                    current_confs = []
                continue

            label = self.id2label.get(pred_id, "O")
            raw_predictions.append({
                "token": token,
                "label": label,
                "confidence": round(conf * 100, 2)
            })

            if token.startswith("##"):
                # Continuation subword: append to the current word
                if current_word is not None:
                    current_word += token[2:]
                    current_confs.append(conf)
                else:
                    # Edge case: ## with no preceding root token
                    current_word = token[2:]
                    current_label = label
                    current_confs = [conf]
            else:
                # New word starts; flush the previous word first
                if current_word is not None:
                    words.append((current_word, current_label,
                                  sum(current_confs) / len(current_confs)))
                current_word = token
                current_label = label  # label of the first (root) subword owns the word
                current_confs = [conf]

        # Flush any remaining word after the loop
        if current_word is not None:
            words.append((current_word, current_label,
                          sum(current_confs) / len(current_confs)))

        # Group consecutive B- and I- tagged words into entity spans
        entities = []
        i = 0
        while i < len(words):
            word, label, conf = words[i]
            if label.startswith("B-"):
                entity_type = label[2:]
                entity_words = [word]
                entity_confs = [conf]
                j = i + 1
                while j < len(words) and words[j][1] == f"I-{entity_type}":
                    entity_words.append(words[j][0])
                    entity_confs.append(words[j][2])
                    j += 1
                entities.append({
                    "entity_type": entity_type,
                    "text": " ".join(entity_words),
                    "confidence": round(
                        sum(entity_confs) / len(entity_confs) * 100, 2
                    )
                })
                i = j
            else:
                i += 1

        avg_conf = (
            sum(e["confidence"] for e in entities) / len(entities)
            if entities else 0.0
        )

        return {
            "entities": entities,
            "entity_count": len(entities),
            "avg_confidence": round(avg_conf, 2),
            "raw_predictions": raw_predictions
        }

    def get_entities_by_type(self, entities_dict):
        """Group entities from extract_entities() output by their type.

        Takes the dict returned by extract_entities() and returns a new dict
        mapping each entity_type to a list of matched text strings — convenient
        for downstream components like drug_lookup or abnormality_detector.
        """
        grouped = {}
        for entity in entities_dict.get("entities", []):
            etype = entity["entity_type"]
            if etype not in grouped:
                grouped[etype] = []
            grouped[etype].append(entity["text"])
        return grouped
