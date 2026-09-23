"""Shared preprocessing keeps training and inference consistent."""

from torchvision import transforms

MEAN = 0.1307
STD = 0.3081


def mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
