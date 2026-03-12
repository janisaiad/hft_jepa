# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---
# Sync (preserve outputs): from repo root: python -m jupytext --to ipynb --update notebooks/multiscale_plotly_identify.py

# %% [markdown]
# # Multi-Scale JEPA: Identify Points and Their Dataset (Plotly)
#
# Same training as multiscale_training_compare, but with Plotly plots so you can hover/identify
# points and see the corresponding raw sequence (dprice, volume, imbalance) in the dataset.

# %% imports and setup
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from torch.amp import GradScaler, autocast
    _autocast = lambda dt, en, dtype: autocast(dt, enabled=en, dtype=dtype)
    _scaler = lambda dt, en: GradScaler(dt, enabled=en)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _autocast = lambda dt, en, dtype: autocast(enabled=en)
    _scaler = lambda dt, en: GradScaler(enabled=en)
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else (Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent)
MODELS = ROOT / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from eb_jepa.datasets.utils import init_data
from eb_jepa.architectures import InverseDynamicsModel, RNNPredictor, TimeSeriesEncoder
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.losses import CosineLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.state_decoder import MLPStateHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = ROOT / "notebooks" / "outputs"
out_dir.mkdir(exist_ok=True)

def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except NameError:
        return False
IN_NOTEBOOK = _in_notebook()

# %% scales and training config
SCALES = [
    {"bar_sec": 0.02, "label": "20ms"},
    {"bar_sec": 1, "label": "1s"},
    {"bar_sec": 60, "label": "60s"},
]
NUM_STEPS_PER_SCALE = int(os.environ.get("PLOTLY_NUM_STEPS", "2000"))

# %% build_jepa and train_scale (same as multiscale_training_compare)
def build_jepa(state_dim, seq_len, device):
    encoder = TimeSeriesEncoder(input_dim=state_dim, hidden_dim=64, output_dim=2, use_final_ln=False)
    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim,
        action_dim=1,
        final_ln=nn.Identity(),
    )
    _, f, _, h, w = encoder(torch.rand(1, state_dim, seq_len, 1, 1)).shape
    idm = InverseDynamicsModel(state_dim=h * w * f, hidden_dim=256, action_dim=1).to(device)
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=8, std_coeff=8, sim_coeff_t=1, idm_coeff=0.5,
        idm=idm, first_t_only=False, projector=None,
        spatial_as_samples=False, idm_after_proj=False, sim_t_after_proj=False,
    )
    predcost = CosineLossSeq()
    jepa = JEPA(encoder, nn.Identity(), predictor, regularizer, predcost, pred_coeff=8.0).to(device)
    return jepa

def train_scale(loader, data_config, scale_label, num_steps):
    ds = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset
    state_dim = getattr(data_config, "state_dim", ds.state_dim)
    seq_len = getattr(data_config, "seq_len", ds.seq_len)
    jepa = build_jepa(state_dim, seq_len, device)
    state_head = MLPStateHead(
        input_dim=jepa.encoder.mlp_output_dim,
        output_dim=state_dim,
        normalizer=getattr(ds, "normalizer", None),
    ).to(device)
    state_prober = JEPAProbe(jepa=jepa, head=state_head, hcost=nn.MSELoss())
    jepa_opt = AdamW(jepa.parameters(), lr=1e-3, weight_decay=1e-6)
    probe_opt = AdamW(state_head.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = _scaler(device.type, False)
    dtype = torch.float32
    losses = []
    step = 0
    loader_iter = iter(loader)
    while step < num_steps:
        try:
            x_b, a_b, loc_b, _, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x_b, a_b, loc_b, _, _ = next(loader_iter)
        x_b = x_b.permute(0, 2, 1, 3, 4).to(device)
        a_b = a_b.permute(0, 2, 1).to(device)
        loc_b = loc_b.permute(0, 2, 1).to(device)
        x_b = torch.nan_to_num(x_b, nan=0.0, posinf=0.0, neginf=0.0)
        a_b = torch.nan_to_num(a_b, nan=0.0, posinf=0.0, neginf=0.0)
        loc_b = torch.nan_to_num(loc_b, nan=0.0, posinf=0.0, neginf=0.0)
        jepa_opt.zero_grad()
        with _autocast(device.type, False, dtype):
            _, (jepa_loss, rloss, _, _, pl) = jepa.unroll(
                x_b, a_b, nsteps=8, unroll_mode="autoregressive", ctxt_window_time=1,
                compute_loss=True, return_all_steps=False,
            )
        scaler.scale(jepa_loss).backward()
        torch.nn.utils.clip_grad_norm_(jepa.parameters(), max_norm=1.0)
        scaler.step(jepa_opt)
        scaler.update()
        probe_opt.zero_grad()
        with _autocast(device.type, False, dtype):
            probe_loss = state_prober(observations=x_b[:, :, :1], targets=loc_b[:, :, :1])
        scaler.scale(probe_loss).backward()
        torch.nn.utils.clip_grad_norm_(state_head.parameters(), max_norm=1.0)
        scaler.step(probe_opt)
        scaler.update()
        losses.append(jepa_loss.item() + probe_loss.item())
        step += 1
        if step % 100 == 0:
            print(f"  {scale_label} step {step}/{num_steps} total={jepa_loss.item() + probe_loss.item():.4f}")
    jepa.eval()
    all_z = []
    raw_records = []  # list of dicts: dprice, volume, spread, imbalance (each seq_len)
    with torch.no_grad():
        for i, (x_b, a_b, loc_b, _, _) in enumerate(loader):
            if i >= 16:
                break
            x_b = x_b.permute(0, 2, 1, 3, 4).to(device)
            x_b = torch.nan_to_num(x_b, nan=0.0, posinf=0.0, neginf=0.0)
            z_b = jepa.encode(x_b)
            all_z.append(z_b.cpu())
            # loc_b [B, T, state_dim], a_b [B, T, action_dim] from loader
            for b in range(loc_b.size(0)):
                raw_records.append({
                    "dprice": loc_b[b, :, 0].numpy().copy(),
                    "volume": loc_b[b, :, 1].numpy().copy(),
                    "spread": loc_b[b, :, 2].numpy().copy() if loc_b.size(2) > 2 else np.zeros(loc_b.size(1)),
                    "imbalance": a_b[b, :, 0].numpy().copy(),
                })
    z_all = torch.cat(all_z, dim=0)
    z_2d = z_all[:, :, :, 0, 0].permute(0, 2, 1)
    return jepa, np.array(losses), z_2d, raw_records

# %% run training and collect embeddings + raw data per scale
results = {}
for scale in SCALES:
    print(f"\n--- Scale: {scale['label']} (bar_sec={scale['bar_sec']}) ---")
    loader, _, data_config = init_data("hft_timeseries", cfg_data={"bar_sec": scale["bar_sec"]})
    jepa, losses, z_2d, raw_records = train_scale(loader, data_config, scale["label"], NUM_STEPS_PER_SCALE)
    results[scale["label"]] = {"losses": losses, "z_2d": z_2d, "raw_records": raw_records, "jepa": jepa}

# %% Plotly: scatter of final embeddings per scale with hover = index + dataset summary
def _hover_row(idx, z0, z1, rec):
    dprice = rec["dprice"]
    vol = rec["volume"]
    imb = rec["imbalance"]
    last5_d = ", ".join([f"{x:.4f}" for x in dprice[-5:]])
    last5_v = ", ".join([f"{x:.2f}" for x in vol[-5:]])
    last5_i = ", ".join([f"{x:.3f}" for x in imb[-5:]])
    return (
        f"<b>Index</b> {idx}<br>"
        f"<b>z</b> ({z0:.3f}, {z1:.3f})<br>"
        f"<b>dprice</b> mean={dprice.mean():.4f} std={dprice.std():.4f}<br>last5: {last5_d}<br>"
        f"<b>volume</b> mean={vol.mean():.2f} sum={vol.sum():.2f}<br>last5: {last5_v}<br>"
        f"<b>imbalance</b> mean={imb.mean():.3f}<br>last5: {last5_i}"
    )

for label, r in results.items():
    z = r["z_2d"].numpy()
    z_end = z[:, -1, :]
    raw = r["raw_records"]
    hover_texts = [_hover_row(i, z_end[i, 0], z_end[i, 1], raw[i]) for i in range(len(z_end))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=z_end[:, 0], y=z_end[:, 1],
        mode="markers",
        marker=dict(size=6, opacity=0.7),
        text=hover_texts,
        hoverinfo="text",
        name=label,
    ))
    fig.update_layout(
        title=f"Embeddings ({label}): hover to see index and dataset (dprice, volume, imbalance)",
        xaxis_title="Embedding dim 0",
        yaxis_title="Embedding dim 1",
        template="plotly_white",
        height=500,
    )
    fig.write_html(out_dir / f"plotly_embed_{label}.html")
    # fig.show()  # skip inline display to avoid huge notebook; open .html in browser

# %% Plotly: trajectory lines (path from z_0 to z_end) per scale, hover on points
for label, r in results.items():
    z = r["z_2d"].numpy()
    raw = r["raw_records"]
    n_show = min(30, z.shape[0])
    fig = go.Figure()
    for i in range(n_show):
        rec = raw[i]
        hover_t = [f"t={t} z=({z[i,t,0]:.3f},{z[i,t,1]:.3f}) dprice={rec['dprice'][t]:.4f} vol={rec['volume'][t]:.2f}" for t in range(z.shape[1])]
        fig.add_trace(go.Scatter(
            x=z[i, :, 0], y=z[i, :, 1],
            mode="lines+markers",
            line=dict(width=1),
            marker=dict(size=4),
            text=hover_t,
            hoverinfo="text",
            name=f"idx={i}",
        ))
    fig.update_layout(
        title=f"Trajectories ({label}): first {n_show} samples, hover on path",
        xaxis_title="Embedding dim 0",
        yaxis_title="Embedding dim 1",
        template="plotly_white",
        height=500,
        showlegend=False,
    )
    fig.write_html(out_dir / f"plotly_trajectories_{label}.html")
    # fig.show()  # skip inline display to avoid huge notebook; open .html in browser

# %% Plotly: dropdown to pick scale and sample index, show that sample's dprice / volume / imbalance
def _make_sample_figure(scale_label, idx):
    r = results[scale_label]
    raw = r["raw_records"][idx]
    z = r["z_2d"].numpy()[idx]
    T = len(raw["dprice"])
    fig = make_subplots(rows=3, cols=1, subplot_titles=("dprice", "volume", "imbalance"), shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=list(range(T)), y=raw["dprice"], name="dprice", mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(T)), y=raw["volume"], name="volume", mode="lines+markers"), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(T)), y=raw["imbalance"], name="imbalance", mode="lines+markers"), row=3, col=1)
    fig.update_layout(title=f"Scale={scale_label} Index={idx} (z_end=({z[-1,0]:.3f}, {z[-1,1]:.3f}))", height=450, template="plotly_white")
    return fig

if IN_NOTEBOOK:
    import ipywidgets as widgets
    from IPython.display import display
    scale_dropdown = widgets.Dropdown(options=list(results.keys()), description="Scale:")
    max_idx = max(len(r["raw_records"]) for r in results.values())
    idx_slider = widgets.IntSlider(min=0, max=max(0, max_idx - 1), value=0, description="Index:")
    out = widgets.Output()

    def on_scale_change(change):
        scale_label = scale_dropdown.value
        n = len(results[scale_label]["raw_records"])
        idx_slider.max = max(0, n - 1)
        if idx_slider.value > idx_slider.max:
            idx_slider.value = idx_slider.max

    def on_change(change):
        with out:
            out.clear_output(wait=True)
            scale_label = scale_dropdown.value
            idx = idx_slider.value
            r = results[scale_label]
            if idx >= len(r["raw_records"]):
                idx = len(r["raw_records"]) - 1
                idx_slider.value = idx
            fig = _make_sample_figure(scale_label, idx)
            fig.show()

    scale_dropdown.observe(on_scale_change, names="value")
    scale_dropdown.observe(on_change, names="value")
    idx_slider.observe(on_change, names="value")
    on_scale_change(None)
    display(widgets.HBox([scale_dropdown, idx_slider]))
    display(out)
    on_change(None)
else:
    print(f"Widget skipped (run in Jupyter for interactive scale/index picker). HTML files in {out_dir}")

# %% optional: single Plotly figure with all three scales (different colors), same hover
fig = go.Figure()
colors = {"20ms": "red", "1s": "blue", "60s": "green"}
for label, r in results.items():
    z = r["z_2d"].numpy()
    z_end = z[:, -1, :]
    raw = r["raw_records"]
    hover_texts = [_hover_row(i, z_end[i, 0], z_end[i, 1], raw[i]) for i in range(len(z_end))]
    fig.add_trace(go.Scatter(
        x=z_end[:, 0], y=z_end[:, 1],
        mode="markers",
        marker=dict(size=5, opacity=0.6, color=colors.get(label, "gray")),
        text=hover_texts,
        hoverinfo="text",
        name=label,
    ))
fig.update_layout(
    title="All scales: hover to see index and dataset",
    xaxis_title="Embedding dim 0",
    yaxis_title="Embedding dim 1",
    template="plotly_white",
    height=550,
)
fig.write_html(out_dir / "plotly_embed_all_scales.html")
if IN_NOTEBOOK:
    fig.show()

# %% cone analysis helper: compute stats for one index
def _analyze_one(idx, z, raw):
    z_0 = z[idx, 0, :]
    z_end = z[idx, -1, :]
    disp = z_end - z_0
    angle_rad = np.arctan2(disp[1], disp[0])
    angle_deg = np.degrees(angle_rad)
    r_ = raw[idx]
    dprice, vol, imb = r_["dprice"], r_["volume"], r_["imbalance"]
    return {
        "idx": idx, "z_0": z_0, "z_end": z_end, "disp": disp, "angle_deg": angle_deg,
        "disp_norm": np.linalg.norm(disp),
        "dprice_sum": dprice.sum(), "dprice_mean": dprice.mean(), "dprice_std": dprice.std(),
        "dprice_cumsum_end": np.cumsum(dprice)[-1],
        "volume_sum": vol.sum(), "volume_mean": vol.mean(),
        "imbalance_mean": imb.mean(), "imbalance_std": imb.std(),
    }

# %% 60s cone: base 941, triangle 447 & 400
scale_label = "60s"
r = results[scale_label]
z = r["z_2d"].numpy()
raw = r["raw_records"]
n_total = z.shape[0]
base_idx, cone_idx_a, cone_idx_b = 941, 447, 400
for idx in [base_idx, cone_idx_a, cone_idx_b]:
    if idx >= n_total:
        raise ValueError(f"index {idx} out of range for {scale_label} (n={n_total})")
base = _analyze_one(base_idx, z, raw)
cone_a = _analyze_one(cone_idx_a, z, raw)
cone_b = _analyze_one(cone_idx_b, z, raw)
print("=== 60s cone: base 941, triangle 447 & 400 ===\n")
for name, d in [("Base 941", base), ("Cone 447", cone_a), ("Cone 400", cone_b)]:
    print(f"  {name}: z_0=({d['z_0'][0]:.4f},{d['z_0'][1]:.4f}) z_end=({d['z_end'][0]:.4f},{d['z_end'][1]:.4f}) disp_norm={d['disp_norm']:.4f} angle={d['angle_deg']:.1f}deg")
    print(f"    dprice sum={d['dprice_sum']:.6f} vol_sum={d['volume_sum']:.2f} imb_mean={d['imbalance_mean']:.4f}")
dist_ba = np.linalg.norm(base["z_0"] - cone_a["z_0"])
dist_bb = np.linalg.norm(base["z_0"] - cone_b["z_0"])
print(f"  z_0 distances: base->447={dist_ba:.4f} base->400={dist_bb:.4f}")
print("\n  Interpretation: Base 941 has tiny displacement (apex). 447 moves mainly along dim0 (angle≈0°, positive dprice). 400 moves along dim1 (angle≈-86°). Cone = same region of starts, two divergent directions (return vs liquidity).")

# %% 20ms cone: base 780, triangle 705 & 901
scale_label = "20ms"
r = results[scale_label]
z = r["z_2d"].numpy()
raw = r["raw_records"]
n_total = z.shape[0]
base_idx, cone_idx_a, cone_idx_b = 780, 705, 901
for idx in [base_idx, cone_idx_a, cone_idx_b]:
    if idx >= n_total:
        raise ValueError(f"index {idx} out of range for {scale_label} (n={n_total})")
base = _analyze_one(base_idx, z, raw)
cone_a = _analyze_one(cone_idx_a, z, raw)
cone_b = _analyze_one(cone_idx_b, z, raw)
print("=== 20ms cone: base 780, triangle 705 & 901 ===\n")
for name, d in [("Base 780", base), ("Cone 705", cone_a), ("Cone 901", cone_b)]:
    print(f"  {name}: z_0=({d['z_0'][0]:.4f},{d['z_0'][1]:.4f}) z_end=({d['z_end'][0]:.4f},{d['z_end'][1]:.4f}) disp_norm={d['disp_norm']:.4f} angle={d['angle_deg']:.1f}deg")
    print(f"    dprice sum={d['dprice_sum']:.6f} vol_sum={d['volume_sum']:.2f} imb_mean={d['imbalance_mean']:.4f}")
dist_ba = np.linalg.norm(base["z_0"] - cone_a["z_0"])
dist_bb = np.linalg.norm(base["z_0"] - cone_b["z_0"])
print(f"  z_0 distances: base->705={dist_ba:.4f} base->901={dist_bb:.4f}")
print("\n  Interpretation: Base 780 has smallest displacement; moves mainly in dim1 (angle≈90°). 705 and 901 have large displacements at angles -141° and -110° (down-left): both negative dprice, activity drop. Cone = shared start, base stays near apex, two sides diverge (similar down-move direction).")

# %% plot 60s trajectories (941, 447, 400)
r = results["60s"]
z, raw = r["z_2d"].numpy(), r["raw_records"]
T = z.shape[1]
fig = make_subplots(rows=2, cols=2, subplot_titles=("60s: embedding paths (941 base, 447 & 400 cone)", "dprice", "volume", "imbalance"))
for name, idx, color in [("base 941", 941, "black"), ("cone 447", 447, "red"), ("cone 400", 400, "blue")]:
    fig.add_trace(go.Scatter(x=z[idx, :, 0], y=z[idx, :, 1], mode="lines+markers", name=name, line=dict(color=color), marker=dict(size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(T)), y=raw[idx]["dprice"], mode="lines+markers", line=dict(color=color), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(range(T)), y=raw[idx]["volume"], mode="lines+markers", line=dict(color=color), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(T)), y=raw[idx]["imbalance"], mode="lines+markers", line=dict(color=color), showlegend=False), row=2, col=2)
fig.update_layout(title="60s cone: base 941 vs cone 447 & 400", height=500, template="plotly_white")
fig.write_html(out_dir / "plotly_cone_analysis_60s.html")
if IN_NOTEBOOK:
    fig.show()

# %% plot 20ms trajectories (780, 705, 901)
r = results["20ms"]
z, raw = r["z_2d"].numpy(), r["raw_records"]
T = z.shape[1]
fig = make_subplots(rows=2, cols=2, subplot_titles=("20ms: embedding paths (780 base, 705 & 901 cone)", "dprice", "volume", "imbalance"))
for name, idx, color in [("base 780", 780, "black"), ("cone 705", 705, "red"), ("cone 901", 901, "blue")]:
    fig.add_trace(go.Scatter(x=z[idx, :, 0], y=z[idx, :, 1], mode="lines+markers", name=name, line=dict(color=color), marker=dict(size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(T)), y=raw[idx]["dprice"], mode="lines+markers", line=dict(color=color), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(range(T)), y=raw[idx]["volume"], mode="lines+markers", line=dict(color=color), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(T)), y=raw[idx]["imbalance"], mode="lines+markers", line=dict(color=color), showlegend=False), row=2, col=2)
fig.update_layout(title="20ms cone: base 780 vs cone 705 & 901", height=500, template="plotly_white")
fig.write_html(out_dir / "plotly_cone_analysis_20ms.html")
if IN_NOTEBOOK:
    fig.show()
print("Done. HTML files in", out_dir)
