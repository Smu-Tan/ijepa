#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

from PIL import Image


TRAIN_IMAGES_FILE = "train_X.bin"
TRAIN_LABELS_FILE = "train_y.bin"
TEST_IMAGES_FILE = "test_X.bin"
TEST_LABELS_FILE = "test_y.bin"
CLASS_NAMES_FILE = "class_names.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the torchvision STL-10 binary release into ImageFolder format."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the stl10_binary directory downloaded by torchvision.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output directory for the ImageFolder dataset.",
    )
    parser.add_argument(
        "--eval-split-name",
        choices=("val", "test"),
        default="val",
        help="Name to use for the evaluation split in the exported ImageFolder layout.",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpg"),
        default="png",
        help="Output image format.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing image files if they already exist.",
    )
    return parser.parse_args()


def slugify_class_name(name: str) -> str:
    slug = name.lower().replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "unknown"


def load_labels(path: Path):
    return [value - 1 for value in path.read_bytes()]


def load_images(path: Path):
    data = path.read_bytes()
    image_size = 3 * 96 * 96
    if len(data) % image_size != 0:
        raise ValueError(f"Unexpected STL-10 image file size: {path}")

    images = []
    for index in range(0, len(data), image_size):
        sample = data[index : index + image_size]
        channels = []
        for channel_idx in range(3):
            start = channel_idx * 96 * 96
            end = start + 96 * 96
            channel = sample[start:end]
            rows = [channel[row_start : row_start + 96] for row_start in range(0, 96 * 96, 96)]
            channels.append(rows)

        pixels = []
        for y in range(96):
            row = []
            for x in range(96):
                row.append((channels[0][y][x], channels[1][y][x], channels[2][y][x]))
            pixels.append(row)

        image = Image.new("RGB", (96, 96))
        image.putdata([pixel for row in pixels for pixel in row])
        images.append(image)
    return images


def load_class_names(path: Path) -> list[str]:
    if path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    ]


def export_split(
    split_name: str,
    images,
    labels,
    class_names: list[str],
    dst_root: Path,
    image_format: str,
    force: bool,
) -> None:
    if len(images) != len(labels):
        raise ValueError(f"Mismatched STL-10 split sizes for {split_name}: {len(images)} images vs {len(labels)} labels")

    image_ext = ".png" if image_format == "png" else ".jpg"
    for index, (image, label) in enumerate(zip(images, labels)):
        class_name = slugify_class_name(class_names[label])
        class_dir = dst_root / split_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        out_path = class_dir / f"{index:05d}{image_ext}"
        if out_path.exists() and not force:
            continue

        if image_format == "jpg":
            image.save(out_path, quality=95)
        else:
            image.save(out_path)


def main() -> None:
    args = parse_args()

    src_root = args.src.expanduser().resolve()
    dst_root = args.dst.expanduser().resolve()

    required = [
        src_root / TRAIN_IMAGES_FILE,
        src_root / TRAIN_LABELS_FILE,
        src_root / TEST_IMAGES_FILE,
        src_root / TEST_LABELS_FILE,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing STL-10 files: {missing_paths}")

    class_names = load_class_names(src_root / CLASS_NAMES_FILE)
    train_images = load_images(src_root / TRAIN_IMAGES_FILE)
    train_labels = load_labels(src_root / TRAIN_LABELS_FILE)
    test_images = load_images(src_root / TEST_IMAGES_FILE)
    test_labels = load_labels(src_root / TEST_LABELS_FILE)

    export_split(
        split_name="train",
        images=train_images,
        labels=train_labels,
        class_names=class_names,
        dst_root=dst_root,
        image_format=args.image_format,
        force=args.force,
    )
    export_split(
        split_name=args.eval_split_name,
        images=test_images,
        labels=test_labels,
        class_names=class_names,
        dst_root=dst_root,
        image_format=args.image_format,
        force=args.force,
    )

    print(f"Exported STL-10 to ImageFolder format at: {dst_root}")


if __name__ == "__main__":
    main()
