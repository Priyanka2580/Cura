from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms


def preprocess_for_classifier(image_path):
    """Preprocess an image file for EfficientNet-B0 input.

    Opens any image format, converts to RGB, resizes to 224x224,
    and applies ImageNet normalization.

    Returns a (1, 3, 224, 224) float tensor.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = Image.open(path)
    except UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {image_path}")
    except Exception as e:
        raise ValueError(f"Failed to open image {image_path}: {e}")

    try:
        image = image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Image conversion failed for {image_path}: {e}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image).unsqueeze(0)


def load_image_for_ocr(image_path):
    """Validate image path and return it as a string for PaddleOCR.

    PaddleOCR accepts a file path directly, so we only need to confirm
    the file exists before handing the path over.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return str(path)
