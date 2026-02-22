"""Process data/original/*.bmp → data/fragment_v2/{NNN}/ with original, gray, lined variants."""

from pathlib import Path

import cv2
import numpy as np


SRC_DIR = Path(__file__).resolve().parent / "data" / "original"
DST_DIR = Path(__file__).resolve().parent / "data" / "fragment_v2"


def gray_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = hsv[:, :, 2]
    return gray


def lined_image(img):
    gray = cv2.bitwise_not(img)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_blur, 50, 130)

    kernel = np.ones((6, 6), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((4, 4), np.uint8)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    lined = cv2.bitwise_not(eroded)
    return lined


def process_one(src_path: Path, dst_folder: Path):
    """Read one BMP, resize to 640×480, write original/gray/lined PNGs."""
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"  [SKIP] cannot read {src_path.name}")
        return
    img = cv2.resize(img, (640, 480))

    dst_folder.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(dst_folder / "original.png"), img)

    gray = gray_image(img)
    cv2.imwrite(str(dst_folder / "gray.png"), gray)

    lined = lined_image(gray)
    cv2.imwrite(str(dst_folder / "lined.png"), lined)


def main():
    bmps = sorted(SRC_DIR.glob("*.bmp"))
    print(f"Found {len(bmps)} images in {SRC_DIR}")

    for bmp in bmps:
        # Extract the number from filenames like "001 .bmp"
        name = bmp.stem.strip()  # "001 " → "001"
        folder_name = name.zfill(3)
        dst_folder = DST_DIR / folder_name
        process_one(bmp, dst_folder)
        print(f"  {bmp.name} → {dst_folder.name}/")

    print(f"\nDone. {len(bmps)} images → {DST_DIR}")


if __name__ == "__main__":
    main()
