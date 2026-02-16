# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
# ---

# %% [markdown]
# # AC Time Series JEPA - Interactive Explore
#
# Explore data, run training steps, and verify training works. Run cells in Jupyter or VS Code.

# %% imports and setup
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# %% load data
from eb_jepa.datasets.utils import init_data

loader, val_loader, data_config = init_data("hft_timeseries")
x, a, loc, _, _ = next(iter(loader))

print("Batch shapes: x", x.shape, "| a", a.shape, "| loc", loc.shape)
print("Batches:", len(loader), "| batch_size:", data_config.batch_size)

# %% plot sample sequence
feat_names = ["dPrice", "volume", "spread", "bid_sz", "ask_sz", "bid_ct", "ask_ct", "num_trades"]
n_feat = loc.shape[2]
n_cols = min(4, n_feat)
n_rows = (n_feat + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2 * n_rows), sharex=True)
axes = np.array(axes).flatten()
sample_idx = 0
t = torch.arange(loc.shape[1])
for i in range(n_feat):
    axes[i].plot(t, loc[sample_idx, :, i].numpy())
    axes[i].set_ylabel(feat_names[i] if i < len(feat_names) else f"feat_{i}")
for j in range(n_feat, len(axes)):
    axes[j].set_visible(False)
axes[-1].set_xlabel("Time step")
plt.suptitle(f"Sample sequence: state ({n_feat} features)")
plt.tight_layout()
out_dir = ROOT / "notebooks" / "outputs"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "sample_sequence.png", dpi=100)
plt.show()

# %% build model
from eb_jepa.architectures import (
    InverseDynamicsModel,
    RNNPredictor,
    TimeSeriesEncoder,
)
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.state_decoder import MLPStateHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset
state_dim = getattr(data_config, "state_dim", ds.state_dim)
action_dim = 1
seq_len = getattr(data_config, "seq_len", ds.seq_len)
nsteps = 8

encoder = TimeSeriesEncoder(input_dim=state_dim, hidden_dim=64, output_dim=2)
test_out = encoder(torch.rand(1, state_dim, seq_len, 1, 1))
_, f, _, h, w = test_out.shape

predictor = RNNPredictor(
    hidden_size=encoder.mlp_output_dim,
    action_dim=action_dim,
    final_ln=nn.LayerNorm(encoder.mlp_output_dim),
)
idm = InverseDynamicsModel(
    state_dim=h * w * f,
    hidden_dim=256,
    action_dim=action_dim,
).to(device)
regularizer = VC_IDM_Sim_Regularizer(
    cov_coeff=8, std_coeff=16, sim_coeff_t=12, idm_coeff=1,
    idm=idm, first_t_only=False, projector=None,
    spatial_as_samples=False, idm_after_proj=False, sim_t_after_proj=False,
)
jepa = JEPA(encoder, nn.Identity(), predictor, regularizer, SquareLossSeq()).to(device)

state_head = MLPStateHead(
    input_dim=encoder.mlp_output_dim,
    output_dim=state_dim,
    normalizer=getattr(ds, "normalizer", None),
).to(device)
state_prober = JEPAProbe(jepa=jepa, head=state_head, hcost=nn.MSELoss())

jepa_opt = AdamW(jepa.parameters(), lr=1e-3, weight_decay=1e-6)
probe_opt = AdamW(state_head.parameters(), lr=1e-3, weight_decay=1e-5)
scaler = GradScaler(device.type, enabled=True)
dtype = torch.float16

print("Model built. Encoder out dim:", encoder.mlp_output_dim)

# %% training loop - run N steps and record losses
NUM_STEPS = 50
losses = []

for step, (x_b, a_b, loc_b, _, _) in enumerate(loader):
    if step >= NUM_STEPS:
        break
    x_b = x_b.permute(0, 2, 1, 3, 4).to(device)
    a_b = a_b.permute(0, 2, 1).to(device)
    loc_b = loc_b.permute(0, 2, 1).to(device)

    jepa_opt.zero_grad()
    with autocast(device.type, enabled=True, dtype=dtype):
        _, (jepa_loss, regl, _, _, pl) = jepa.unroll(
            x_b, a_b, nsteps=nsteps,
            unroll_mode="autoregressive", ctxt_window_time=1,
            compute_loss=True, return_all_steps=False,
        )
    scaler.scale(jepa_loss).backward()
    scaler.step(jepa_opt)
    scaler.update()

    probe_opt.zero_grad()
    with autocast(device.type, enabled=True, dtype=dtype):
        probe_loss = state_prober(
            observations=x_b[:, :, :1],
            targets=loc_b[:, :, :1],
        )
    scaler.scale(probe_loss).backward()
    scaler.step(probe_opt)
    scaler.update()

    total = jepa_loss.item() + probe_loss.item()
    losses.append({"total": total, "jepa": jepa_loss.item(), "pred": pl.item(), "probe": probe_loss.item()})
    if step % 10 == 0:
        print(f"Step {step:3d} | loss={total:.4f} | pred={pl.item():.4f} | probe={probe_loss.item():.4f}")

# %% plot training loss
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot([l["total"] for l in losses], label="total")
ax.plot([l["pred"] for l in losses], label="pred", alpha=0.8)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("Training loss over steps")
plt.tight_layout()
plt.savefig(out_dir / "training_loss.png", dpi=100)
plt.show()

# %% test: verify training goes well (loss should decrease)
first_avg = sum(l["total"] for l in losses[:10]) / min(10, len(losses))
last_avg = sum(l["total"] for l in losses[-10:]) / min(10, len(losses))
decreased = last_avg < first_avg
print(f"First 10 steps avg loss: {first_avg:.4f}")
print(f"Last 10 steps avg loss:  {last_avg:.4f}")
print(f"Loss decreased: {decreased} {'✓' if decreased else '✗'}")
assert decreased, "Training failed: loss did not decrease"

# %% inference
x_infer, _, _, _, _ = next(iter(loader))
x_infer = x_infer.permute(0, 2, 1, 3, 4).to(device)
with torch.no_grad():
    z = jepa.encode(x_infer)
print("Encoded shape:", z.shape)

# %% [markdown]
# ## Summary
#
# - **Data**: HFT state (dPrice, volume, spread) + action (imbalance)
# - **Model**: TimeSeriesEncoder + RNNPredictor + VC_IDM_Sim_Regularizer
# - **Full train**: `uv run python -m examples.ac_timeseries_jepa.main`
