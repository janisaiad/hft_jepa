"""Tests for eb_jepa module (from facebookresearch/eb_jepa)."""
import pytest
import torch


def test_eb_jepa_imports():
    """We verify core eb_jepa modules can be imported."""
    from eb_jepa.jepa import JEPA, JEPAbase, JEPAProbe
    from eb_jepa.losses import VCLoss, SquareLossSeq
    from eb_jepa.architectures import ResNet5, ImpalaEncoder, RNNPredictor
    from eb_jepa.logging import get_logger
    assert JEPA is not None
    assert JEPAbase is not None


def test_jepa_base_forward():
    """We verify JEPAbase can encode observations."""
    from eb_jepa.jepa import JEPAbase
    enc = torch.nn.Linear(3, 8)
    aenc = torch.nn.Linear(1, 8)
    pred = torch.nn.Linear(16, 8)
    jepa = JEPAbase(enc, aenc, pred)
    x = torch.randn(2, 5, 3)
    z = jepa.encode(x)
    assert z.shape == (2, 5, 8)


def test_hft_jepa_import():
    """We verify hft_jepa package can be imported."""
    import hft_jepa as hft
    assert hft is not None


def test_ac_timeseries_jepa_training_step():
    """We verify ac_timeseries_jepa runs one epoch."""
    import subprocess
    result = subprocess.run(
        [
            "uv", "run", "python", "-m", "examples.ac_timeseries_jepa.main",
            "--folder", "/tmp/hft_jepa_test_run",
            "--optim.epochs", "1",
        ],
        cwd="/Data/janis.aiad/hftjepa",
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
