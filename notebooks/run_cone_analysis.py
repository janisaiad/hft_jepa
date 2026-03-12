"""Run training (reduced steps), then cone analysis for 60s and 20ms. Print analysis for inspection."""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
sys.path.insert(0, str(MODELS))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

try:
    from torch.amp import GradScaler, autocast
    _autocast = lambda dt, en, dtype: autocast(dt, enabled=en, dtype=dtype)
    _scaler = lambda dt, en: GradScaler(dt, enabled=en)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _autocast = lambda dt, en, dtype: autocast(enabled=en)
    _scaler = lambda dt, en: GradScaler(enabled=en)
from torch.optim import AdamW

from eb_jepa.datasets.utils import init_data
from eb_jepa.architectures import InverseDynamicsModel, RNNPredictor, TimeSeriesEncoder
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.losses import CosineLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.state_decoder import MLPStateHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALES = [{"bar_sec": 0.02, "label": "20ms"}, {"bar_sec": 1, "label": "1s"}, {"bar_sec": 60, "label": "60s"}]
NUM_STEPS = 300  # reduced to get results quickly

def build_jepa(state_dim, seq_len, device):
    encoder = TimeSeriesEncoder(input_dim=state_dim, hidden_dim=64, output_dim=2, use_final_ln=False)
    predictor = RNNPredictor(hidden_size=encoder.mlp_output_dim, action_dim=1, final_ln=nn.Identity())
    _, f, _, h, w = encoder(torch.rand(1, state_dim, seq_len, 1, 1)).shape
    idm = InverseDynamicsModel(state_dim=h * w * f, hidden_dim=256, action_dim=1).to(device)
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=8, std_coeff=8, sim_coeff_t=1, idm_coeff=0.5,
        idm=idm, first_t_only=False, projector=None,
        spatial_as_samples=False, idm_after_proj=False, sim_t_after_proj=False,
    )
    jepa = JEPA(encoder, nn.Identity(), predictor, regularizer, CosineLossSeq(), pred_coeff=8.0).to(device)
    return jepa

def train_and_collect(loader, data_config, scale_label, num_steps):
    ds = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset
    state_dim = getattr(data_config, "state_dim", ds.state_dim)
    seq_len = getattr(data_config, "seq_len", ds.seq_len)
    jepa = build_jepa(state_dim, seq_len, device)
    state_head = MLPStateHead(
        input_dim=jepa.encoder.mlp_output_dim, output_dim=state_dim,
        normalizer=getattr(ds, "normalizer", None),
    ).to(device)
    state_prober = JEPAProbe(jepa=jepa, head=state_head, hcost=nn.MSELoss())
    jepa_opt = AdamW(jepa.parameters(), lr=1e-3, weight_decay=1e-6)
    probe_opt = AdamW(state_head.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = _scaler(device.type, False)
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
        _, (jepa_loss, _, _, _, _) = jepa.unroll(
            x_b, a_b, nsteps=8, unroll_mode="autoregressive", ctxt_window_time=1,
            compute_loss=True, return_all_steps=False,
        )
        scaler.scale(jepa_loss).backward()
        torch.nn.utils.clip_grad_norm_(jepa.parameters(), max_norm=1.0)
        scaler.step(jepa_opt)
        scaler.update()
        probe_opt.zero_grad()
        probe_loss = state_prober(observations=x_b[:, :, :1], targets=loc_b[:, :, :1])
        scaler.scale(probe_loss).backward()
        scaler.step(probe_opt)
        scaler.update()
        step += 1
    jepa.eval()
    all_z = []
    raw_records = []
    with torch.no_grad():
        for i, (x_b, a_b, loc_b, _, _) in enumerate(loader):
            if i >= 16:
                break
            x_b = x_b.permute(0, 2, 1, 3, 4).to(device)
            x_b = torch.nan_to_num(x_b, nan=0.0, posinf=0.0, neginf=0.0)
            z_b = jepa.encode(x_b)
            all_z.append(z_b.cpu())
            for b in range(loc_b.size(0)):
                raw_records.append({
                    "dprice": loc_b[b, :, 0].numpy().copy(),
                    "volume": loc_b[b, :, 1].numpy().copy(),
                    "spread": loc_b[b, :, 2].numpy().copy() if loc_b.size(2) > 2 else np.zeros(loc_b.size(1)),
                    "imbalance": a_b[b, :, 0].numpy().copy(),
                })
    z_all = torch.cat(all_z, dim=0)
    z_2d = z_all[:, :, :, 0, 0].permute(0, 2, 1).numpy()
    return z_2d, raw_records

def analyze_cone(z, raw, scale_label, base_idx, cone_a_idx, cone_b_idx):
    n_total = z.shape[0]
    for idx in [base_idx, cone_a_idx, cone_b_idx]:
        if idx >= n_total:
            return None
    def one(idx):
        z_0 = z[idx, 0, :]
        z_end = z[idx, -1, :]
        disp = z_end - z_0
        angle_deg = np.degrees(np.arctan2(disp[1], disp[0]))
        r_ = raw[idx]
        dprice, vol, imb = r_["dprice"], r_["volume"], r_["imbalance"]
        return {
            "z_0": z_0, "z_end": z_end, "disp_norm": np.linalg.norm(disp), "angle_deg": angle_deg,
            "dprice_sum": dprice.sum(), "dprice_mean": dprice.mean(), "dprice_std": dprice.std(),
            "volume_sum": vol.sum(), "volume_mean": vol.mean(),
            "imbalance_mean": imb.mean(), "imbalance_std": imb.std(),
        }
    base = one(base_idx)
    a = one(cone_a_idx)
    b = one(cone_b_idx)
    dist_0_ba = np.linalg.norm(base["z_0"] - a["z_0"])
    dist_0_bb = np.linalg.norm(base["z_0"] - b["z_0"])
    dist_0_ab = np.linalg.norm(a["z_0"] - b["z_0"])
    return {
        "scale": scale_label,
        "base_idx": base_idx, "cone_a": cone_a_idx, "cone_b": cone_b_idx,
        "base": base, "a": a, "b": b,
        "dist_z0_base_to_a": dist_0_ba, "dist_z0_base_to_b": dist_0_bb, "dist_z0_a_to_b": dist_0_ab,
    }

def main():
    results = {}
    for scale in SCALES:
        print(f"Training {scale['label']} ({NUM_STEPS} steps)...", file=sys.stderr)
        loader, _, data_config = init_data("hft_timeseries", cfg_data={"bar_sec": scale["bar_sec"]})
        z_2d, raw_records = train_and_collect(loader, data_config, scale["label"], NUM_STEPS)
        results[scale["label"]] = {"z_2d": z_2d, "raw_records": raw_records}
        print(f"  n_samples={len(raw_records)}", file=sys.stderr)

    # 60s: base 941, cone 447 & 400
    out_60 = analyze_cone(
        results["60s"]["z_2d"], results["60s"]["raw_records"],
        "60s", 941, 447, 400,
    )
    # 20ms: base 780, cone 705 & 901
    out_20 = analyze_cone(
        results["20ms"]["z_2d"], results["20ms"]["raw_records"],
        "20ms", 780, 705, 901,
    )

    # Print analysis to stdout for capture
    def print_cone(out):
        if out is None:
            print("Analysis skipped (index out of range)\n")
            return
        s, base_i, ai, bi = out["scale"], out["base_idx"], out["cone_a"], out["cone_b"]
        base, a, b = out["base"], out["a"], out["b"]
        print(f"=== {s} cone: base {base_i}, triangle {ai} & {bi} ===\n")
        for name, idx, d in [("Base", base_i, base), ("Cone A", ai, a), ("Cone B", bi, b)]:
            print(f"  {name} (idx {idx}): z_0=({d['z_0'][0]:.4f},{d['z_0'][1]:.4f}) z_end=({d['z_end'][0]:.4f},{d['z_end'][1]:.4f})")
            print(f"    disp_norm={d['disp_norm']:.4f} angle={d['angle_deg']:.1f}deg")
            print(f"    dprice sum={d['dprice_sum']:.6f} mean={d['dprice_mean']:.6f} vol_sum={d['volume_sum']:.2f} imb_mean={d['imbalance_mean']:.4f}")
        print(f"  Distances at z_0: base->a={out['dist_z0_base_to_a']:.4f} base->b={out['dist_z0_base_to_b']:.4f} a->b={out['dist_z0_a_to_b']:.4f}")
        print()

    print_cone(out_60)
    print_cone(out_20)

if __name__ == "__main__":
    main()
