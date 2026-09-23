"""Predict a MNIST test digit or a tightly cropped image of one digit."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.datasets import MNIST

from data_utils import mnist_transform
from model import DigitCNN


def prepare_image(path, invert=False):
    with Image.open(path) as source:
        image = source.convert("L")
    if invert:
        image = ImageOps.invert(image)
    # Preserve aspect ratio and center the digit in the 28 x 28 canvas.
    bounds = image.getbbox()
    if bounds is None:
        raise ValueError("The image is blank after preprocessing.")
    image = image.crop(bounds)
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    return mnist_transform()(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--index", type=int, help="MNIST test index (default: 0)")
    source.add_argument("--image", type=Path, help="Image of a single light digit on a black background")
    parser.add_argument("--invert", action="store_true", help="For a dark digit on a white background")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint missing. Run python train.py first.")
    if args.invert and args.image is None:
        parser.error("--invert requires --image")
    if args.index is not None and not 0 <= args.index < 10000:
        parser.error("--index must be between 0 and 9999")
    torch.set_num_threads(4)
    model = DigitCNN()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if args.image:
        if not args.image.is_file():
            parser.error(f"Image not found: {args.image}")
        try:
            image = prepare_image(args.image, args.invert)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    else:
        dataset = MNIST(args.data_dir, train=False, download=True, transform=mnist_transform())
        index = args.index if args.index is not None else 0
        image, label = dataset[index]
        print(f"MNIST test index: {index} | True label: {label}")
    with torch.inference_mode():
        probabilities = model(image.unsqueeze(0)).softmax(dim=1)[0]
    scores, digits = probabilities.topk(3)
    print(f"Predicted digit: {digits[0].item()}")
    print("Top 3 softmax scores (not calibrated confidence):")
    for digit, score in zip(digits.tolist(), scores.tolist()):
        print(f"  {digit}: {score:.2%}")


if __name__ == "__main__":
    main()
