"""Classification metrics on small hand-checked examples and against scikit-learn."""

from __future__ import annotations

import importlib.util
import unittest

import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import accuracy, confusion_matrix, macro_metrics, per_class_metrics
from src.training import validate

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def logits_for(predictions: list[int], num_classes: int) -> torch.Tensor:
    return one_hot(torch.tensor(predictions), num_classes).float()


class TestClassificationMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = torch.tensor([0, 0, 0, 1, 1, 2])
        self.logits = logits_for([0, 0, 1, 1, 1, 1], 3)
        self.confusion = confusion_matrix(self.logits, self.targets, 3)

    def test_rows_are_actual_and_columns_predicted(self) -> None:
        self.assertEqual(self.confusion.tolist(), [[2, 1, 0], [0, 2, 0], [0, 1, 0]])

    def test_per_class_values(self) -> None:
        per_class = per_class_metrics(self.confusion)
        for values, expected in ((per_class["precision"], [1, 0.5, 0]), (per_class["recall"], [2 / 3, 1, 0]),
                                 (per_class["f1"], [0.8, 2 / 3, 0])):
            self.assertTrue(torch.allclose(values, torch.tensor(expected, dtype=torch.double)))
        self.assertEqual(per_class["support"].tolist(), [3, 2, 1])

    def test_macro_values(self) -> None:
        metrics = macro_metrics(self.confusion)
        self.assertAlmostEqual(metrics["macro_precision"], 0.5)
        self.assertAlmostEqual(metrics["macro_recall"], 5 / 9)
        self.assertAlmostEqual(metrics["macro_f1"], (0.8 + 2 / 3) / 3)

    def test_class_missing_from_the_split_does_not_lower_the_average(self) -> None:
        wider = confusion_matrix(logits_for([0, 0, 1, 1, 1, 1], 4), self.targets, 4)
        self.assertEqual(macro_metrics(wider), macro_metrics(self.confusion))

    def test_empty_confusion_matrix(self) -> None:
        with self.assertRaises(ValueError):
            macro_metrics(torch.zeros(3, 3, dtype=torch.long))

    def test_validate_fills_the_confusion_matrix_over_all_batches(self) -> None:
        class Passthrough(nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return inputs

        loader = DataLoader(TensorDataset(self.logits, self.targets), batch_size=4)
        confusion = torch.zeros(3, 3, dtype=torch.long)
        metrics = validate(Passthrough(), loader, nn.CrossEntropyLoss(), torch.device("cpu"), confusion=confusion)
        self.assertEqual(confusion.tolist(), self.confusion.tolist())
        self.assertAlmostEqual(metrics["accuracy"], confusion.diag().sum().item() / confusion.sum().item())

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
    def test_matches_scikit_learn(self) -> None:
        from sklearn import metrics as sklearn_metrics

        generator = torch.Generator().manual_seed(0)
        targets = torch.randint(0, 20, (500,), generator=generator)
        logits = torch.randn(500, 20, generator=generator)
        predictions = logits.argmax(dim=1).numpy()
        confusion = confusion_matrix(logits, targets, 20)

        self.assertEqual(confusion.tolist(), sklearn_metrics.confusion_matrix(targets, predictions, labels=range(20)).tolist())
        self.assertAlmostEqual(confusion.diag().sum().item() / 500, accuracy(logits, targets))
        macro = macro_metrics(confusion)
        for name, function in (("macro_precision", sklearn_metrics.precision_score),
                               ("macro_recall", sklearn_metrics.recall_score), ("macro_f1", sklearn_metrics.f1_score)):
            self.assertAlmostEqual(macro[name], function(targets, predictions, average="macro", zero_division=0))


if __name__ == "__main__":
    unittest.main()
