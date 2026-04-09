import unittest
from types import SimpleNamespace

import torch

from models.PatchTST import Model


def make_configs(task_name="long_term_forecast"):
    return SimpleNamespace(
        task_name=task_name,
        seq_len=96,
        pred_len=24,
        enc_in=7,
        d_model=16,
        factor=1,
        n_heads=1,
        d_ff=32,
        dropout=0.1,
        activation="gelu",
        e_layers=1,
        num_class=3,
    )


class TestPatchTSTForecastOnly(unittest.TestCase):
    def test_builds_forecast_head(self):
        model = Model(make_configs())

        self.assertTrue(hasattr(model, "head"))
        self.assertFalse(hasattr(model, "projection"))

    def test_forecast_output_shape_is_unchanged(self):
        model = Model(make_configs())
        x_enc = torch.randn(2, 96, 7)
        x_mark_enc = torch.randn(2, 96, 4)
        x_dec = torch.randn(2, 72, 7)
        x_mark_dec = torch.randn(2, 72, 4)

        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        self.assertEqual(output.shape, (2, 24, 7))

    def test_rejects_non_forecast_tasks(self):
        with self.assertRaises(ValueError):
            Model(make_configs(task_name="classification"))


if __name__ == "__main__":
    unittest.main()
