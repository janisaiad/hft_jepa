"""AC time series JEPA: action-conditioned JEPA for HFT state/action sequences.

Adapted from ac_video_jepa. State=(dPrice, volume, spread), action=imbalance.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from time import time

import fire
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.architectures import (
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
    TimeSeriesEncoder,
)
from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.logging import add_file_handler, get_logger
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.state_decoder import MLPStateHead
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_checkpoint,
    load_config,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)

logger = get_logger(__name__)


def run(
    fname: str = "examples/ac_timeseries_jepa/cfgs/train.yaml",
    cfg=None,
    folder=None,
    log_dir: str | None = None,
    **overrides,
):
    """Train action-conditioned time series JEPA on HFT state/action sequences."""
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    if folder is None:
        if cfg.meta.get("model_folder"):
            folder = Path(cfg.meta.model_folder)
            exp_name = folder.name
        else:
            sweep_name = get_default_dev_name()
            exp_name = get_exp_name("ac_timeseries_jepa", cfg)
            folder = get_unified_experiment_dir(
                example_name="ac_timeseries_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        folder = Path(folder)
        exp_name = folder.name
    os.makedirs(folder, exist_ok=True)

    losses_csv_path = None
    if log_dir:
        log_path = add_file_handler(log_dir)
        logger.info("Logging to %s", log_path)
        losses_csv_path = Path(log_dir) / "losses.csv"

    loader, val_loader, data_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
    )

    setup_device("auto")
    setup_seed(cfg.meta.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_run = setup_wandb(
        project="hft_jepa",
        config={"example": "ac_timeseries_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=folder,
        run_name=exp_name,
        tags=["ac_timeseries_jepa"],
        enabled=cfg.logging.get("log_wandb", False),
    )

    log_data_info(
        cfg.data.env_name,
        len(loader),
        data_config.batch_size,
        train_samples=getattr(data_config, "size", len(loader.dataset)),
        val_samples=getattr(data_config, "val_size", 0),
    )

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16)
    use_amp = cfg.training.get("use_amp", True)
    scaler = GradScaler(device.type, enabled=use_amp)

    steps_per_epoch = getattr(data_config, "size", len(loader) * data_config.batch_size) // data_config.batch_size
    total_steps = cfg.optim.epochs * steps_per_epoch
    latest_ckpt_path = folder / "latest.pth.tar"
    config_path = folder / "config.yaml"
    OmegaConf.save(cfg, config_path)

    state_dim = getattr(data_config, "state_dim", cfg.model.get("state_dim", 8))
    action_dim = cfg.model.get("action_dim", 1)
    seq_len = getattr(data_config, "seq_len", 17)

    test_input = torch.rand(1, state_dim, seq_len, 1, 1)
    encoder = TimeSeriesEncoder(
        input_dim=state_dim,
        hidden_dim=cfg.model.henc,
        output_dim=cfg.model.get("embed_dim", 2),
    )
    test_output = encoder(test_input)
    _, f, _, h, w = test_output.shape

    # final_ln=Identity: LayerNorm on predictor forces zero-mean -> y=-x for 2D embeddings
    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim,
        action_dim=action_dim,
        final_ln=nn.Identity(),
    )
    aencoder = nn.Identity()

    idm = InverseDynamicsModel(
        state_dim=h * w * f,
        hidden_dim=256,
        action_dim=action_dim,
    ).to(device)
    projector = None
    if cfg.model.regularizer.get("use_proj"):
        projector = Projector(f"{encoder.mlp_output_dim}-{encoder.mlp_output_dim*4}-{encoder.mlp_output_dim*4}")

    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg.model.regularizer.cov_coeff,
        std_coeff=cfg.model.regularizer.std_coeff,
        sim_coeff_t=cfg.model.regularizer.sim_coeff_t,
        idm_coeff=cfg.model.regularizer.get("idm_coeff", 0.1),
        idm=idm,
        first_t_only=cfg.model.regularizer.get("first_t_only"),
        projector=projector,
        spatial_as_samples=cfg.model.regularizer.get("spatial_as_samples", False),
        idm_after_proj=cfg.model.regularizer.get("idm_after_proj", False),
        sim_t_after_proj=cfg.model.regularizer.get("sim_t_after_proj", False),
    )
    ploss = SquareLossSeq()
    pred_coeff = cfg.model.get("pred_coeff", 15.0)
    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss, pred_coeff=pred_coeff).to(device)

    log_model_info(jepa, {})
    log_config(cfg)

    state_head = MLPStateHead(
        input_dim=encoder.mlp_output_dim,
        output_dim=state_dim,
        normalizer=getattr(loader.dataset, "normalizer", None),
    ).to(device)
    state_prober = JEPAProbe(jepa=jepa, head=state_head, hcost=nn.MSELoss())

    jepa_optimizer = AdamW(
        jepa.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.get("weight_decay", 1e-6),
    )
    jepa_scheduler = CosineWithWarmup(jepa_optimizer, total_steps, warmup_ratio=0.1)
    probe_optimizer = AdamW(state_head.parameters(), lr=1e-3, weight_decay=1e-5)
    probe_scheduler = CosineWithWarmup(probe_optimizer, total_steps, warmup_ratio=0.1)

    start_epoch = 0
    ckpt_info = {}
    if cfg.meta.load_model:
        ckpt_path = folder / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        if ckpt_path.exists():
            ckpt_info = load_checkpoint(ckpt_path, jepa, jepa_optimizer, jepa_scheduler, device=device)
            start_epoch = ckpt_info.get("epoch", 0)
            if "state_head_state_dict" in ckpt_info:
                state_head.load_state_dict(ckpt_info["state_head_state_dict"])

    for epoch in range(start_epoch, cfg.optim.epochs):
        epoch_start = time()
        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch}/{cfg.optim.epochs - 1}",
        )
        for idx, (x, a, loc, _, _) in pbar:
            itr_start = time()
            global_step = epoch * len(loader) + idx

            x = x.to(device)
            a = a.to(device)
            loc = loc.to(device)
            x = x.permute(0, 2, 1, 3, 4)
            a = a.permute(0, 2, 1)
            loc = loc.permute(0, 2, 1)

            total_loss = torch.tensor(0.0, device=device)

            jepa_optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                _, (jepa_loss, regl, regl_unweight, regldict, pl) = jepa.unroll(
                    x,
                    a,
                    nsteps=cfg.model.nsteps,
                    unroll_mode="autoregressive",
                    ctxt_window_time=1,
                    compute_loss=True,
                    return_all_steps=False,
                )
                total_loss += jepa_loss

            scaler.scale(jepa_loss).backward()
            if cfg.optim.get("grad_clip_enc") and cfg.optim.get("grad_clip_pred"):
                scaler.unscale_(jepa_optimizer)
                torch.nn.utils.clip_grad_norm_(jepa.encoder.parameters(), cfg.optim.grad_clip_enc)
                torch.nn.utils.clip_grad_norm_(jepa.predictor.parameters(), cfg.optim.grad_clip_pred)
            scaler.step(jepa_optimizer)
            scaler.update()
            jepa_scheduler.step()

            probe_optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                probe_loss = state_prober(
                    observations=x[:, :, :1],
                    targets=loc[:, :, :1],
                )
                if hasattr(loader.dataset, "normalizer") and loader.dataset.normalizer is not None:
                    probe_loss = loader.dataset.normalizer.unnormalize_mse(probe_loss)
                total_loss += probe_loss

            scaler.scale(probe_loss).backward()
            scaler.step(probe_optimizer)
            scaler.update()
            probe_scheduler.step()

            pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "reg": f"{regl.item():.4f}", "pred": f"{pl.item():.4f}"})

            if global_step % cfg.logging.log_every == 0:
                log_data = {
                    "train/total_loss": total_loss.item(),
                    "train/reg_loss": regl.item(),
                    "train/pred_loss": pl.item(),
                    "train/probe_loss": probe_loss.item(),
                    "global_step": global_step,
                    "epoch": epoch,
                }
                for k, v in regldict.items():
                    log_data[f"train/regl/{k}"] = v
                if cfg.logging.get("log_wandb"):
                    wandb_run.log(log_data, step=global_step)
                if losses_csv_path:
                    write_header = not losses_csv_path.exists()
                    row = {"step": global_step, "epoch": epoch, "loss": total_loss.item(), "reg": regl.item(), "pred": pl.item(), "probe": probe_loss.item(), **regldict}
                    with open(losses_csv_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=row.keys())
                        if write_header:
                            w.writeheader()
                        w.writerow(row)
                logger.info("step=%d epoch=%d loss=%.4f reg=%.4f pred=%.4f probe=%.4f", global_step, epoch, total_loss.item(), regl.item(), pl.item(), probe_loss.item())

        epoch_time = time() - epoch_start
        log_epoch(
            epoch,
            {"loss": total_loss.item(), "reg": regl.item(), "pred": pl.item(), "probe": probe_loss.item()},
            total_epochs=cfg.optim.epochs,
            elapsed_time=epoch_time,
        )

        save_checkpoint(
            latest_ckpt_path,
            model=jepa,
            optimizer=jepa_optimizer,
            scheduler=jepa_scheduler,
            epoch=epoch,
            step=global_step,
            state_head_state_dict=state_head.state_dict(),
            probe_optimizer_state_dict=probe_optimizer.state_dict(),
            probe_scheduler_state_dict=probe_scheduler.state_dict(),
        )
        if epoch % cfg.logging.save_every_n_epochs == 0:
            save_checkpoint(
                folder / f"e-{epoch}.pth.tar",
                model=jepa,
                optimizer=jepa_optimizer,
                scheduler=jepa_scheduler,
                epoch=epoch,
                step=global_step,
                state_head_state_dict=state_head.state_dict(),
            )

    logger.info("Training complete.")
    return {"folder": str(folder)}


if __name__ == "__main__":
    fire.Fire(run)
