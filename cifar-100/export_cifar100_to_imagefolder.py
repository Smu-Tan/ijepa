#!/usr/bin/env python3

import argparse
import pickle
import re
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the official CIFAR-100 Python release into ImageFolder format."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the extracted cifar-100-python directory.",
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


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def slugify_class_name(name: str) -> str:
    slug = name.lower().replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "unknown"


def reshape_cifar_image(flat_pixels) -> Image.Image:
    # CIFAR stores each sample as 3072 bytes: 1024 R, 1024 G, 1024 B.
    channels = []
    for channel_idx in range(3):
        start = channel_idx * 1024
        end = start + 1024
        channel = flat_pixels[start:end]
        rows = [channel[row_start : row_start + 32] for row_start in range(0, 1024, 32)]
        channels.append(rows)

    pixels = []
    for y in range(32):
        row = []
        for x in range(32):
            row.append((channels[0][y][x], channels[1][y][x], channels[2][y][x]))
        pixels.append(row)

    image = Image.new("RGB", (32, 32))
    image.putdata([pixel for row in pixels for pixel in row])
    return image


def export_split(
    split_name: str,
    split_data: dict,
    class_names: list[str],
    dst_root: Path,
    image_format: str,
    force: bool,
) -> None:
    image_ext = ".png" if image_format == "png" else ".jpg"
    labels = split_data["fine_labels"]
    filenames = split_data["filenames"]
    flat_images = split_data["data"]

    for index, (flat_pixels, label, original_name) in enumerate(
        zip(flat_images, labels, filenames)
    ):
        class_name = slugify_class_name(class_names[label])
        class_dir = dst_root / split_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(original_name).stem
        out_path = class_dir / f"{index:05d}_{stem}{image_ext}"
        if out_path.exists() and not force:
            continue

        image = reshape_cifar_image(flat_pixels)
        if image_format == "jpg":
            image.save(out_path, quality=95)
        else:
            image.save(out_path)


def main() -> None:
    args = parse_args()

    src_root = args.src.expanduser().resolve()
    dst_root = args.dst.expanduser().resolve()

    meta_path = src_root / "meta"
    train_path = src_root / "train"
    test_path = src_root / "test"

    missing = [path for path in (meta_path, train_path, test_path) if not path.exists()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing CIFAR-100 files: {missing_paths}")

    meta = load_pickle(meta_path)
    class_names = meta["fine_label_names"]

    train_data = load_pickle(train_path)
    test_data = load_pickle(test_path)

    export_split(
        split_name="train",
        split_data=train_data,
        class_names=class_names,
        dst_root=dst_root,
        image_format=args.image_format,
        force=args.force,
    )
    export_split(
        split_name=args.eval_split_name,
        split_data=test_data,
        class_names=class_names,
        dst_root=dst_root,
        image_format=args.image_format,
        force=args.force,
    )

    print(f"Exported CIFAR-100 to ImageFolder format at: {dst_root}")


if __name__ == "__main__":
    main()
