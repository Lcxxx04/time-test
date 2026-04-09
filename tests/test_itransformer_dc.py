import unittest
from types import SimpleNamespace

import torch

from models.DC_iTransformer import Model


def make_configs(task_name="long_term_forecast"):
    return SimpleNamespace(
        task_name=task_name,
        seq_len=96,
        pred_len=24,
        enc_in=7,
        d_model=16,
        embed="timeF",
        freq="h",
        dropout=0.1,
        factor=1,
        n_heads=1,
        d_ff=32,
        activation="gelu",
        e_layers=1,
    )


class TestDCiTransformer(unittest.TestCase):
    def test_builds_depthwise_local_cnn(self):
        cfg = make_configs()
        model = Model(cfg)
        enc_in = cfg.enc_in

        self.assertTrue(hasattr(model, "local_cnn"))
        self.assertIsInstance(model.local_cnn[0], torch.nn.Conv1d)
        self.assertEqual(model.local_cnn[0].in_channels, enc_in)
        self.assertEqual(model.local_cnn[0].out_channels, enc_in)
        self.assertEqual(model.local_cnn[0].groups, enc_in)
        self.assertEqual(model.local_cnn[0].kernel_size, (3,))
        self.assertEqual(model.local_cnn[0].padding, (1,))
        self.assertIsInstance(model.local_cnn[1], torch.nn.GELU)

    def test_forecast_output_shape_is_unchanged(self):
        cfg = make_configs()
        model = Model(cfg)
        batch, seq_len, enc_in, pred_len = 2, cfg.seq_len, cfg.enc_in, cfg.pred_len
        x_enc = torch.randn(batch, seq_len, enc_in)
        x_mark_enc = torch.randn(batch, seq_len, 4)
        x_dec = torch.randn(batch, 72, enc_in)
        x_mark_dec = torch.randn(batch, 72, 4)

        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        self.assertEqual(output.shape, (batch, pred_len, enc_in))

    def test_rejects_non_forecast_tasks(self):
        with self.assertRaises(ValueError):
            Model(make_configs(task_name="imputation"))


if __name__ == "__main__":
    unittest.main()
