"""
Add 10 extra prescription images (041–050) with the same 6-variant augmentation
used in augment_dataset.py: original, _b, _br, _d, _r, _c
"""
import random
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance

EXTRA_IMAGES_DIR = Path("C:/Users/PRIYANKA/Desktop/Cura/datasets/Extra Images")
OUTPUT_DIR       = Path("C:/Users/PRIYANKA/Desktop/Cura/datasets/training_images/prescription")
START_INDEX      = 41

def augment_image(image: Image.Image, base_name: str, output_folder: Path) -> int:
    image = image.convert("RGB")
    saved = 0

    image.save(output_folder / f"{base_name}.png", format="PNG")
    saved += 1

    image.filter(ImageFilter.GaussianBlur(radius=1)).save(
        output_folder / f"{base_name}_b.png", format="PNG")
    saved += 1

    ImageEnhance.Brightness(image).enhance(1.4).save(
        output_folder / f"{base_name}_br.png", format="PNG")
    saved += 1

    ImageEnhance.Brightness(image).enhance(0.6).save(
        output_folder / f"{base_name}_d.png", format="PNG")
    saved += 1

    angle = random.uniform(-10, 10)
    image.rotate(angle, resample=Image.BICUBIC, expand=False,
                 fillcolor=(255, 255, 255)).save(
        output_folder / f"{base_name}_r.png", format="PNG")
    saved += 1

    ImageEnhance.Contrast(image).enhance(1.5).save(
        output_folder / f"{base_name}_c.png", format="PNG")
    saved += 1

    return saved


def main():
    extra_images = sorted([
        p for p in EXTRA_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    ])

    if len(extra_images) != 10:
        print(f"Expected 10 images, found {len(extra_images)}. Aborting.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    random.seed(42)  # reproducible rotations

    for i, img_path in enumerate(extra_images):
        index     = START_INDEX + i
        base_name = f"prescription_{index:03d}"

        with Image.open(img_path) as img:
            saved = augment_image(img, base_name, OUTPUT_DIR)
            total_saved += saved
            print(f"  {img_path.name}  ->  {base_name}.*  ({saved} files)")

    print(f"\nDone. {total_saved} files written to {OUTPUT_DIR}")

    # Verify final count
    all_files = sorted(OUTPUT_DIR.glob("prescription_*.png"))
    print(f"Total prescription files in folder : {len(all_files)}")
    print(f"Last 6 files:")
    for f in all_files[-6:]:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
