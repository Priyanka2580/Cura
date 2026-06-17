from pathlib import Path


def load_image_for_ocr(image_path):
    """Validate image path and return it as a string for PaddleOCR.

    PaddleOCR accepts a file path directly, so we only need to confirm
    the file exists before handing the path over.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return str(path)
