"""Offline checks for the model, metric weighting and image preprocessing."""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import DigitCNN
from predict import prepare_image
from train import run_epoch


class PipelineTests(unittest.TestCase):
    def test_training_and_checkpoint_round_trip(self):
        torch.manual_seed(7)
        model = DigitCNN()
        images, labels = torch.randn(4, 1, 28, 28), torch.tensor([0, 1, 2, 3])
        before = model.features[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        run_epoch(model, DataLoader(TensorDataset(images, labels), batch_size=4),
                  nn.CrossEntropyLoss(), "cpu", optimizer)
        self.assertFalse(torch.equal(before, model.features[0].weight))
        model.eval()
        with torch.inference_mode():
            expected = model(images)
        self.assertEqual(expected.shape, (4, 10))
        self.assertTrue(torch.isfinite(expected).all())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = DigitCNN().eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            with torch.inference_mode():
                torch.testing.assert_close(restored(images), expected)

    def test_metrics_weight_the_last_partial_batch(self):
        logits = torch.tensor([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]])
        labels = torch.tensor([0, 0, 1])
        criterion = nn.CrossEntropyLoss()
        loss, accuracy = run_epoch(nn.Identity(),
                                  DataLoader(TensorDataset(logits, labels), batch_size=2),
                                  criterion, "cpu")
        self.assertAlmostEqual(loss, criterion(logits, labels).item(), places=6)
        self.assertAlmostEqual(accuracy, 2 / 3)

    def test_image_polarity_and_blank_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "digit.png"
            image = Image.new("L", (40, 60), 0)
            ImageDraw.Draw(image).line((20, 5, 20, 55), fill=255, width=5)
            image.save(path)
            expected = prepare_image(path)
            ImageOps.invert(image).save(path)
            torch.testing.assert_close(prepare_image(path, invert=True), expected)
            self.assertEqual(expected.shape, (1, 28, 28))
            Image.new("L", (28, 28), 0).save(path)
            with self.assertRaises(ValueError):
                prepare_image(path)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
