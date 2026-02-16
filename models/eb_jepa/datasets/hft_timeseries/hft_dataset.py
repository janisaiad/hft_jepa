"""HFT time series dataset: state=(dPrice, volume, spread), action=imbalance.

Adapted from ac_video_jepa for time series. Outputs (x, a, loc) compatible with
JEPA unroll: x [B,T,3,1,1], a [B,T,1], loc [B,T,3] for probing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def _parse_time_mins(s) -> int:
    """Parse 'HH:MM' or time-like to minutes since midnight."""
    s = str(s).strip()
    if hasattr(s, "hour"):  # datetime.time
        return getattr(s, "hour", 0) * 60 + getattr(s, "minute", 0)
    parts = s.split(":")
    return int(parts[0]) * 60 + int(parts[1]) if len(parts) >= 2 else 0


@dataclass
class HFTDatasetConfig:
    """Config for HFT time series dataset."""

    batch_size: int = 64
    seq_len: int = 17
    bar_sec: Union[int, float] = 60
    state_dim: int = 8
    action_dim: int = 1
    num_samples: int = 10000
    val_ratio: float = 0.1
    data_path: Optional[str] = None
    symbols: tuple[str, ...] = ("GOOGL", "AAPL", "AAL")
    use_synthetic: bool = False
    state_features: str = "rich"
    max_files_per_symbol: int = 10
    # per-symbol trading hours (start, end) as "HH:MM"; filters ts_event by time-of-day
    symbol_hours: Optional[dict[str, Union[tuple[str, str], list[str]]]] = None


def _aggregate_to_bars(df: "pl.DataFrame", bar_sec: Union[int, float], state_features: str = "rich") -> "pl.DataFrame":
    """Aggregate tick data to 1-min bars. Returns state cols + imbalance.
    minimal: dprice, volume, spread (3). rich: + bid_sz_00, ask_sz_00, bid_ct_00, ask_ct_00, num_trades (8).
    """
    df = df.with_columns(
        (pl.col("ts_event").dt.epoch("ns") // int(bar_sec * 1e9)).alias("bar_id")
    )
    mid = (pl.col("bid_px_00") + pl.col("ask_px_00")) / 2
    action_str = pl.col("action").cast(pl.Utf8).str.to_uppercase()
    is_trade = action_str == "T"
    agg_exprs = [
        pl.col("bid_px_00").last().alias("bid_end"),
        pl.col("ask_px_00").last().alias("ask_end"),
        mid.first().alias("mid_start"),
        mid.last().alias("mid_end"),
        pl.col("size").filter(is_trade).sum().alias("volume"),
    ]
    if state_features == "rich":
        agg_exprs.extend([
            pl.col("bid_sz_00").last().alias("bid_sz"),
            pl.col("ask_sz_00").last().alias("ask_sz"),
            pl.col("bid_ct_00").last().alias("bid_ct"),
            pl.col("ask_ct_00").last().alias("ask_ct"),
            is_trade.sum().alias("num_trades"),
        ])
    bars = df.group_by("bar_id").agg(*agg_exprs).sort("bar_id")
    bars = bars.with_columns(
        ((pl.col("mid_end") - pl.col("mid_start")) / (pl.col("mid_start") + 1e-12)).alias("dprice"),
        (pl.col("ask_end") - pl.col("bid_end")).alias("spread"),
    )
    # we fill null dprice/spread when mid or bid/ask are missing (e.g. quote-only bars, schema gaps)
    bars = bars.with_columns([
        pl.col("dprice").fill_null(0).clip(-1e6, 1e6),
        pl.col("spread").fill_null(0),
        pl.col("volume").fill_null(0),
    ])
    trades_df = df.filter(is_trade).with_columns(
        pl.when(pl.col("side").cast(pl.Utf8).str.to_uppercase() == "B")
        .then(pl.col("size")).otherwise(0).alias("bid_sz"),
        pl.when(pl.col("side").cast(pl.Utf8).str.to_uppercase() == "A")
        .then(pl.col("size")).otherwise(0).alias("ask_sz"),
    )
    if len(trades_df) > 0:
        vol_by_bar = trades_df.group_by("bar_id").agg(
            pl.col("bid_sz").sum().alias("bvol"),
            pl.col("ask_sz").sum().alias("avol"),
        )
        vol_by_bar = vol_by_bar.with_columns(
            ((pl.col("bvol") - pl.col("avol")) / (pl.col("bvol") + pl.col("avol") + 1e-12)).alias("imbalance")
        )
        bars = bars.join(vol_by_bar.select(["bar_id", "imbalance"]), on="bar_id", how="left")
    else:
        bars = bars.with_columns(pl.lit(0.0).alias("imbalance"))
    bars = bars.with_columns(pl.col("imbalance").fill_null(0))
    if state_features == "rich":
        bars = bars.with_columns([
            pl.col("bid_sz").fill_null(0), pl.col("ask_sz").fill_null(0),
            pl.col("bid_ct").fill_null(0), pl.col("ask_ct").fill_null(0),
            pl.col("num_trades").fill_null(0),
        ])
    if state_features == "minimal":
        return bars.select(["dprice", "volume", "spread", "imbalance"])
    return bars.select(["dprice", "volume", "spread", "bid_sz", "ask_sz", "bid_ct", "ask_ct", "num_trades", "imbalance"])


def _filter_by_hours(df: "pl.DataFrame", start_mins: int, end_mins: int) -> "pl.DataFrame":
    """Filter rows where ts_event time-of-day is in [start_mins, end_mins) (minutes since midnight)."""
    mins = pl.col("ts_event").dt.hour() * 60 + pl.col("ts_event").dt.minute()
    return df.filter((mins >= start_mins) & (mins < end_mins))


def _load_parquet_sequences(
    data_path: str,
    symbols: tuple[str, ...],
    bar_sec: Union[int, float],
    seq_len: int,
    state_features: str = "rich",
    symbol_hours: Optional[dict[str, Union[tuple[str, str], list[str]]]] = None,
    max_files_per_symbol: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Load parquet data and build (states, actions) sequences.
    Volume and count-like features are normalized by bar_sec so they are comparable across scales.
    """
    if not HAS_POLARS:
        return np.zeros((0, 0, 8)), np.zeros((0, 0, 1))
    base = Path(data_path)
    bar_sec_f = float(bar_sec)
    state_cols = ["dprice", "volume", "spread"] if state_features == "minimal" else [
        "dprice", "volume", "spread", "bid_sz", "ask_sz", "bid_ct", "ask_ct", "num_trades"
    ]
    all_states, all_actions = [], []
    for sym in symbols:
        hours = (symbol_hours or {}).get(sym)
        start_mins = _parse_time_mins(hours[0]) if hours else 0
        end_mins = _parse_time_mins(hours[1]) if hours else 24 * 60
        for fp in list(base.glob(f"{sym}/{sym}_*.parquet"))[:max_files_per_symbol]:
            try:
                df = pl.read_parquet(fp)
                if "bid_px_00" not in df.columns:
                    continue
                if "ts_event" in df.columns and hours:
                    df = _filter_by_hours(df, start_mins, end_mins)
                if len(df) == 0:
                    continue
                bars = _aggregate_to_bars(df, bar_sec, state_features)
                if len(bars) < seq_len + 1:
                    continue
                s = bars.select(state_cols).to_numpy().astype(np.float32)
                a = bars.select(["imbalance"]).to_numpy().astype(np.float32)
                s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                # we normalize volume and count-like features by bar_sec so comparable across scales
                s[:, :, 1] = s[:, :, 1] / (bar_sec_f + 1e-12)
                if state_features == "rich":
                    s[:, :, 7] = s[:, :, 7] / (bar_sec_f + 1e-12)
                std = s.std(axis=0) + 1e-8
                s = (s - s.mean(axis=0)) / std
                s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                for i in range(len(s) - seq_len):
                    all_states.append(s[i : i + seq_len])
                    all_actions.append(a[i : i + seq_len])
            except Exception:
                continue
    if not all_states:
        return np.zeros((0, 0, len(state_cols))), np.zeros((0, 0, 1))
    return np.stack(all_states), np.stack(all_actions)


def _synthetic_sequences(
    num_samples: int,
    seq_len: int,
    state_dim: int,
    action_dim: int,
    seed: Optional[int] = None,
    bar_sec: Union[int, float] = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (state, action) sequences for testing.
    Volume (idx 1) and num_trades (idx 7) are scaled by 1/bar_sec for cross-scale comparability.
    """
    rng = np.random.default_rng(seed)
    states = np.cumsum(rng.standard_normal((num_samples, seq_len, state_dim)) * 0.01, axis=1).astype(np.float32)
    states[:, :, 1] = np.abs(states[:, :, 1]) + 0.1
    states[:, :, 2] = np.abs(states[:, :, 2]) + 0.01
    bar_sec_f = float(bar_sec)
    states[:, :, 1] = states[:, :, 1] / (bar_sec_f + 1e-12)
    if state_dim > 7:
        states[:, :, 7] = np.abs(states[:, :, 7]) + 0.01
        states[:, :, 7] = states[:, :, 7] / (bar_sec_f + 1e-12)
    actions = rng.standard_normal((num_samples, seq_len, action_dim)).astype(np.float32) * 0.5
    actions = np.clip(actions, -1, 1)
    return states, actions


class HFTTimeSeriesDataset(Dataset):
    """Dataset of (state, action) sequences for AC-JEPA on time series.

    Each sample: x [T, 3, 1, 1], a [T, 1], loc [T, 3].
    JEPA expects x as [B, C, T, H, W] so we use C=3, H=1, W=1.
    """

    def __init__(self, config: HFTDatasetConfig):
        self.config = config
        self.seq_len = config.seq_len
        self.action_dim = config.action_dim
        data_path = config.data_path or os.getenv("FOLDER_PATH") or os.getenv("DATA_PATH")
        if data_path and Path(data_path).exists() and not config.use_synthetic:
            self.states, self.actions = _load_parquet_sequences(
                data_path, config.symbols, config.bar_sec, config.seq_len, config.state_features,
                symbol_hours=config.symbol_hours,
                max_files_per_symbol=getattr(config, "max_files_per_symbol", 10),
            )
        else:
            self.states, self.actions = _synthetic_sequences(
                config.num_samples, config.seq_len, config.state_dim, config.action_dim,
                bar_sec=config.bar_sec,
            )
        if len(self.states) == 0:
            self.states, self.actions = _synthetic_sequences(
                config.num_samples, config.seq_len, config.state_dim, config.action_dim,
                bar_sec=config.bar_sec,
            )
        self.state_dim = self.states.shape[-1]
        self.normalizer = _Normalizer(self.states)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.states[idx]
        a = self.actions[idx]
        s_norm = self.normalizer(s)
        x = torch.from_numpy(s_norm).float().unsqueeze(-1).unsqueeze(-1)
        a_t = torch.from_numpy(a).float()
        loc = torch.from_numpy(s).float()
        dummy = torch.zeros(1)
        return x, a_t, loc, dummy, dummy

    def get_seq_length(self, idx: int) -> int:
        return self.seq_len


class _Normalizer:
    """Simple normalizer for states."""

    def __init__(self, data: np.ndarray):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        self.mean = data.reshape(-1, data.shape[-1]).mean(axis=0).astype(np.float32)
        self.std = (data.reshape(-1, data.shape[-1]).std(axis=0).astype(np.float32) + 1e-8)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = (x - self.mean) / self.std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def unnormalize_mse(self, mse: torch.Tensor) -> torch.Tensor:
        return mse
