import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from exp.exp_basic import Exp_Basic
from models.DC_iTransformer_GLU import Model


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


class _PassThroughLocalCnn(torch.nn.Module):
    """Preserves (B, C, L) so patched forecast path matches DC-iTransformer layout."""

    def forward(self, x):
        return x


class TestDCiTransformerGLU(unittest.TestCase):
    def test_has_local_cnn_proj_linear_and_proj_gate(self):
        model = Model(make_configs())
        self.assertTrue(hasattr(model, "local_cnn"))
        self.assertTrue(hasattr(model, "proj_linear"))
        self.assertTrue(hasattr(model, "proj_gate"))
        self.assertIsInstance(model.proj_linear, torch.nn.Linear)
        self.assertIsInstance(model.proj_gate, torch.nn.Linear)

    def test_no_projection(self):
        model = Model(make_configs())
        self.assertFalse(hasattr(model, "projection"))

    def test_forward_output_shape(self):
        model = Model(make_configs())
        x_enc = torch.randn(2, 96, 7)
        x_mark_enc = torch.randn(2, 96, 4)
        x_dec = torch.randn(2, 72, 7)
        x_mark_dec = torch.randn(2, 72, 4)

        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        self.assertEqual(tuple(output.shape), (2, 24, 7))

    def test_non_forecast_raises_value_error(self):
        with self.assertRaises(ValueError):
            Model(make_configs(task_name="classification"))

    def test_gate_path_uses_sigmoid(self):
        class FakeEmbedding(torch.nn.Module):
            def forward(self, x, x_mark):
                return torch.zeros(2, 7, 16)

        class FakeEncoder(torch.nn.Module):
            def forward(self, x, attn_mask=None):
                return x, None

        model = Model(make_configs())
        model.local_cnn = _PassThroughLocalCnn()
        model.enc_embedding = FakeEmbedding()
        model.encoder = FakeEncoder()
        torch.nn.init.zeros_(model.proj_linear.weight)
        torch.nn.init.ones_(model.proj_linear.bias)
        torch.nn.init.zeros_(model.proj_gate.weight)
        torch.nn.init.constant_(model.proj_gate.bias, -20.0)

        pattern = torch.tensor([-1.0, 1.0]).repeat(48).view(1, 96, 1)
        x_enc = pattern.repeat(2, 1, 7)

        output = model(x_enc, None, None, None)

        self.assertTrue(torch.allclose(output, torch.zeros_like(output), atol=1e-4))

    def test_strict_load_rejects_single_module_projection_keys(self):
        """Single-module (e.g. DC-iTransformer) checkpoints use projection.*; GLU model has none."""
        model = Model(make_configs())
        sd = dict(model.state_dict())
        sd["projection.weight"] = model.proj_linear.weight.detach().clone()
        sd["projection.bias"] = model.proj_linear.bias.detach().clone()
        with self.assertRaises(RuntimeError):
            model.load_state_dict(sd, strict=True)

    def test_strict_load_rejects_state_dict_missing_local_cnn(self):
        model = Model(make_configs())
        sd = {k: v for k, v in model.state_dict().items() if not k.startswith("local_cnn.")}
        with self.assertRaises(RuntimeError):
            model.load_state_dict(sd, strict=True)

    def test_strict_load_rejects_dc_only_head_checkpoint(self):
        """DC-style head (projection only) cannot satisfy GLU parameters under strict=True."""
        model = Model(make_configs())
        sd = dict(model.state_dict())
        for key in list(sd.keys()):
            if key.startswith("proj_linear.") or key.startswith("proj_gate."):
                del sd[key]
        sd["projection.weight"] = model.proj_linear.weight.detach().clone()
        sd["projection.bias"] = model.proj_linear.bias.detach().clone()
        with self.assertRaises(RuntimeError):
            model.load_state_dict(sd, strict=True)

    def test_exp_basic_resolves_dc_itransformer_glu_alias(self):
        args = make_configs()
        args.use_gpu = False
        args.gpu_type = "cuda"
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = "0"
        args.model = "DC-iTransformer-GLU"

        class DummyExp(Exp_Basic):
            def _build_model(self):
                return self.model_dict[self.args.model](self.args).float()

        exp = DummyExp(args)
        self.assertIn("DC-iTransformer-GLU", exp.model_dict.model_map)
        self.assertIsInstance(exp.model, Model)


if __name__ == "__main__":
    unittest.main()
