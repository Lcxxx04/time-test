import os
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from models.iTransformer import Model

_CHECKPOINTS_DIR = _REPO_ROOT / "checkpoints"


def parse_itransformer_checkpoint_dirname(dirname: str):
    """Parse hyperparameters from Time-Series-Library checkpoint folder names."""

    def grab(pattern: str):
        m = re.search(pattern, dirname)
        return int(m.group(1)) if m else None

    return {
        "seq_len": grab(r"_sl(\d+)_"),
        "pred_len": grab(r"_pl(\d+)_"),
        "d_model": grab(r"_dm(\d+)_"),
        "n_heads": grab(r"_nh(\d+)_"),
        "e_layers": grab(r"_el(\d+)_"),
        "d_ff": grab(r"_df(\d+)_"),
        "factor": grab(r"_fc(\d+)_"),
        "label_len": grab(r"_ll(\d+)_"),
    }


def verify_itransformer_checkpoints():
    """
    Load each checkpoints/*iTransformer*/checkpoint.pth with strict=True and run one forward.
    Returns (failures, successes) where failures is a list of (dirname, error_string).
    """
    if not _CHECKPOINTS_DIR.is_dir():
        return [], []

    failures = []
    successes = []
    subdirs = sorted(
        d
        for d in os.listdir(_CHECKPOINTS_DIR)
        if "iTransformer" in d
        and "DC-iTransformer" not in d
        and "DC_iTransformer" not in d
        and (_CHECKPOINTS_DIR / d).is_dir()
        and (_CHECKPOINTS_DIR / d / "checkpoint.pth").is_file()
    )

    for name in subdirs:
        ckpt_path = _CHECKPOINTS_DIR / name / "checkpoint.pth"
        try:
            parsed = parse_itransformer_checkpoint_dirname(name)
            required = (
                "seq_len",
                "pred_len",
                "d_model",
                "n_heads",
                "e_layers",
                "d_ff",
                "factor",
            )
            missing = [k for k in required if parsed[k] is None]
            if missing:
                raise ValueError(f"could not parse from dirname: {missing}")

            label_len = parsed["label_len"] or 48
            configs = SimpleNamespace(
                task_name="long_term_forecast",
                seq_len=parsed["seq_len"],
                pred_len=parsed["pred_len"],
                enc_in=7,
                d_model=parsed["d_model"],
                embed="timeF",
                freq="h",
                dropout=0.1,
                factor=parsed["factor"],
                n_heads=parsed["n_heads"],
                d_ff=parsed["d_ff"],
                activation="gelu",
                e_layers=parsed["e_layers"],
            )

            state = torch.load(ckpt_path, map_location="cpu")
            model = Model(configs)
            model.load_state_dict(state, strict=True)
            model.eval()

            b, seq_len, pred_len, n_vars = 2, parsed["seq_len"], parsed["pred_len"], 7
            x_enc = torch.randn(b, seq_len, n_vars)
            x_mark_enc = torch.randn(b, seq_len, 4)
            dec_len = label_len + pred_len
            x_dec = torch.randn(b, dec_len, n_vars)
            x_mark_dec = torch.randn(b, dec_len, 4)

            with torch.no_grad():
                out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            expected = (b, pred_len, n_vars)
            if tuple(out.shape) != expected:
                raise AssertionError(f"bad output shape {tuple(out.shape)}, expected {expected}")
            successes.append(name)
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))

    return failures, successes


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


class TestITransformerBaseline(unittest.TestCase):
    def test_no_local_cnn(self):
        model = Model(make_configs())
        self.assertFalse(hasattr(model, "local_cnn"))

    def test_forward_output_shape(self):
        model = Model(make_configs())
        x_enc = torch.randn(2, 96, 7)
        x_mark_enc = torch.randn(2, 96, 4)
        x_dec = torch.randn(2, 72, 7)
        x_mark_dec = torch.randn(2, 72, 4)

        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        self.assertEqual(output.shape, (2, 24, 7))

    def test_checkpoint_dirs_strict_load_and_forward(self):
        failures, successes = verify_itransformer_checkpoints()
        if not failures and not successes:
            self.skipTest("no checkpoints/*iTransformer*/checkpoint.pth found")
        if failures:
            lines = [f"  {d}: {err}" for d, err in failures]
            self.fail("checkpoint verification failed:\n" + "\n".join(lines))
        self.assertGreater(len(successes), 0)


if __name__ == "__main__":
    if "--verify-checkpoints" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--verify-checkpoints"]
        failures, successes = verify_itransformer_checkpoints()
        print(f"checkpoints dir: {_CHECKPOINTS_DIR}")
        print(f"found {len(successes) + len(failures)} iTransformer checkpoint(s) with checkpoint.pth")
        for name in successes:
            print(f"  OK  {name}")
        for name, err in failures:
            print(f"  FAIL {name}")
            print(f"       {err}")
        if failures:
            sys.exit(1)
        if not successes:
            print("  (none)")
        sys.exit(0)
    unittest.main()
