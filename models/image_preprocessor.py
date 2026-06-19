import cv2
import numpy as np
import pillow_heif
from pathlib import Path
from PIL import Image, ImageOps

# Registers a PIL plugin for HEIC/HEIF (iPhone default photo format).
# PaddleOCR's own path-based reader is OpenCV-based and can't decode HEIC
# even with this installed, which is why load_image_for_ocr() below decodes
# the image itself and hands OCREngine a numpy array instead of a path.
pillow_heif.register_heif_opener()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_DIMENSION = 4000  # matches PaddleOCR's own text-detection max_side_limit
BLUR_VARIANCE_THRESHOLD = 100.0  # Laplacian variance below this looks too blurry to read


class UnsupportedImageError(Exception):
    """Raised when an uploaded file can't be used as an OCR input."""


def load_image_for_ocr(image_path):
    """Validate and decode an image into an OCR-ready RGB numpy array.

    Handles the cases a raw file path can't: wrong/unsupported extension,
    oversized files, corrupt files, sideways phone photos (EXIF orientation),
    oversized pixel dimensions, and images too blurry to read. Raises
    UnsupportedImageError with a clear reason on any failure.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UnsupportedImageError(
            f"Unsupported file type '{path.suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise UnsupportedImageError(
            f"Image file too large ({file_size / 1_000_000:.1f} MB). "
            f"Max allowed is {MAX_FILE_SIZE_BYTES / 1_000_000:.0f} MB."
        )

    try:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except Exception as e:
        raise UnsupportedImageError(f"Could not read image file: {image_path}") from e

    width, height = image.size
    longest_side = max(width, height)
    if longest_side > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest_side
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(new_size, Image.LANCZOS)

    image_array = np.array(image)

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_variance < BLUR_VARIANCE_THRESHOLD:
        raise UnsupportedImageError(
            f"Image is too blurry to read (sharpness score {blur_variance:.1f}, "
            f"need at least {BLUR_VARIANCE_THRESHOLD:.0f}). Please retake the photo."
        )

    return image_array
