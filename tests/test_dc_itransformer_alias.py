import unittest
from types import SimpleNamespace

import torch

from exp.exp_basic import Exp_Basic
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


class TestDCiTransformerAlias(unittest.TestCase):
    def test_has_local_cnn_groups_match_enc_in(self):
        cfg = make_configs()
        model = Model(cfg)
        self.assertTrue(hasattr(model, "local_cnn"))
        self.assertIsInstance(model.local_cnn[0], torch.nn.Conv1d)
        self.assertEqual(model.local_cnn[0].groups, cfg.enc_in)

    def test_forward_output_shape(self):
        cfg = make_configs()
        model = Model(cfg)
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        x_mark_enc = torch.randn(2, cfg.seq_len, 4)
        x_dec = torch.randn(2, 72, cfg.enc_in)
        x_mark_dec = torch.randn(2, 72, 4)

        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        self.assertEqual(output.shape, (2, cfg.pred_len, cfg.enc_in))

    def test_exp_basic_resolves_dc_itransformer_alias(self):
        args = make_configs()
        args.use_gpu = False
        args.gpu_type = "cuda"
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = "0"
        args.model = "DC-iTransformer"

        class DummyExp(Exp_Basic):
            def _build_model(self):
                return self.model_dict[self.args.model](self.args).float()

        exp = DummyExp(args)
        self.assertIn("DC-iTransformer", exp.model_dict.model_map)
        self.assertIsInstance(exp.model, Model)


if __name__ == "__main__":
    unittest.main()
