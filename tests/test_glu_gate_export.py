import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from exp.exp_long_term_forecasting import summarize_gate_scores
from models.DC_iTransformer_GLU import Model as DCGLUModel
from models.iTransformer_GLU import Model as GLUModel
from vis_gate import summarize_peak_region_analysis


def make_configs():
    return SimpleNamespace(
        task_name="long_term_forecast",
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


class TestGLUGateExport(unittest.TestCase):
    def test_summarize_gate_scores_fields(self):
        gate = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float32).numpy()
        stats = summarize_gate_scores(gate)
        self.assertTrue({"gate_mean", "gate_std", "gate_min", "gate_max"} <= set(stats))

    def test_summarize_peak_region_analysis_fields(self):
        gate = torch.tensor([[0.2, 0.8, 0.9]], dtype=torch.float32).numpy()
        linear_out = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).numpy()
        pred = torch.tensor([[0.9, 1.7, 2.4]], dtype=torch.float32).numpy()
        true = torch.tensor([[1.0, 1.5, 3.0]], dtype=torch.float32).numpy()
        stats = summarize_peak_region_analysis(gate, linear_out, pred, true)
        self.assertTrue(
            {
                "true_peak_threshold",
                "gate_mean_peak_region",
                "gate_mean_non_peak_region",
                "linear_mae_peak_region",
                "linear_mae_non_peak_region",
                "final_mae_peak_region",
                "final_mae_non_peak_region",
            }
            <= set(stats)
        )

    def test_glu_model_stores_last_gate_score(self):
        cfg = make_configs()
        model = GLUModel(cfg)
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        x_mark_enc = torch.randn(2, cfg.seq_len, 4)
        x_dec = torch.randn(2, 72, cfg.enc_in)
        x_mark_dec = torch.randn(2, 72, 4)

        model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        self.assertTrue(hasattr(model, "last_gate_score"))
        self.assertIsNotNone(model.last_gate_score)
        self.assertEqual(tuple(model.last_gate_score.shape), (2, cfg.enc_in, cfg.pred_len))
        self.assertTrue(hasattr(model, "last_linear_out"))
        self.assertIsNotNone(model.last_linear_out)
        self.assertEqual(tuple(model.last_linear_out.shape), (2, cfg.pred_len, cfg.enc_in))

    def test_dc_glu_model_stores_last_gate_score(self):
        cfg = make_configs()
        model = DCGLUModel(cfg)
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        x_mark_enc = torch.randn(2, cfg.seq_len, 4)
        x_dec = torch.randn(2, 72, cfg.enc_in)
        x_mark_dec = torch.randn(2, 72, 4)

        model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        self.assertTrue(hasattr(model, "last_gate_score"))
        self.assertIsNotNone(model.last_gate_score)
        self.assertEqual(tuple(model.last_gate_score.shape), (2, cfg.enc_in, cfg.pred_len))
        self.assertTrue(hasattr(model, "last_linear_out"))
        self.assertIsNotNone(model.last_linear_out)
        self.assertEqual(tuple(model.last_linear_out.shape), (2, cfg.pred_len, cfg.enc_in))


if __name__ == "__main__":
    unittest.main()
