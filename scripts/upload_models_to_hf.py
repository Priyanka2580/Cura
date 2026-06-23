"""
One-time uploader: pushes the trained model checkpoints from saved_models/
to Hugging Face Hub, so the Streamlit Cloud deployment (which only has
whatever is in the GitHub repo) can pull them at runtime via from_pretrained()
instead of a local path that doesn't exist on the cloud filesystem.

Before running:
    1. Create a Hugging Face access token (Write role) at
       https://huggingface.co/settings/tokens
    2. Add it to .env:  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
    3. Edit MODEL_CLASSIFIER_REPO / MODEL_NER_REPO in config.py to your own
       "username/repo-name".

Run with:  .venv\\Scripts\\python.exe scripts\\upload_models_to_hf.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

from config import MODEL_CLASSIFIER_REPO, MODEL_NER_REPO, HF_TOKEN_ENV

load_dotenv()

token = os.environ.get(HF_TOKEN_ENV)
if not token:
    raise SystemExit(f"{HF_TOKEN_ENV} not set. Add it to your .env file first.")

if "your-hf-username" in MODEL_CLASSIFIER_REPO or "your-hf-username" in MODEL_NER_REPO:
    raise SystemExit(
        "Edit MODEL_CLASSIFIER_REPO / MODEL_NER_REPO in config.py to your "
        "own Hugging Face username/repo-name before running this."
    )

api = HfApi(token=token)

uploads = [
    (MODEL_CLASSIFIER_REPO, "saved_models/classifier/layoutlmv3-classifier", ["best_checkpoint/*"]),
    (MODEL_NER_REPO, "saved_models/ner/biobert-medical-ner-final", []),
]

for repo_id, local_folder, ignore_patterns in uploads:
    print(f"Creating repo (if needed): {repo_id}")
    create_repo(repo_id, token=token, exist_ok=True)

    print(f"Uploading {local_folder} -> {repo_id} ...")
    api.upload_folder(
        folder_path=local_folder,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=ignore_patterns,
    )
    print(f"Done: https://huggingface.co/{repo_id}\n")

print("All uploads complete.")
