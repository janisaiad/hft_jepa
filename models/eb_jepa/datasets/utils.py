from pathlib import Path

import torch
import yaml

from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig

DATASETS_DIR = Path(__file__).parent


def load_env_data_config(env_name: str, overrides: dict = None) -> dict:
    """Load base data config for an environment and apply overrides."""
    config_path = DATASETS_DIR / env_name / "data_config.yaml"
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    if overrides:
        base_config.update(overrides)
    return base_config


def _init_hft_timeseries(cfg_data=None):
    """Initialize HFT time series data loaders."""
    from eb_jepa.datasets.hft_timeseries.hft_dataset import HFTDatasetConfig, HFTTimeSeriesDataset

    merged_cfg = load_env_data_config("hft_timeseries", cfg_data)
    config = HFTDatasetConfig(**{k: v for k, v in merged_cfg.items() if k in HFTDatasetConfig.__dataclass_fields__})
    num_workers = merged_cfg.get("num_workers", 0)
    pin_mem = merged_cfg.get("pin_mem", False)
    persistent_workers = merged_cfg.get("persistent_workers", False) and num_workers > 0

    dset = HFTTimeSeriesDataset(config=config)
    n = len(dset)
    n_val = max(1, int(n * config.val_ratio))
    n_train = n - n_val
    train_dset = torch.utils.data.Subset(dset, range(n_train))
    val_dset = torch.utils.data.Subset(dset, range(n_train, n))

    loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=min(16, len(val_dset)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    config.size = n_train
    config.val_size = n_val
    config.img_size = 1
    config.batch_size = config.batch_size
    config.state_dim = dset.state_dim
    config.seq_len = dset.seq_len
    train_dset.normalizer = dset.normalizer
    val_dset.normalizer = dset.normalizer
    return loader, val_loader, config


def init_data(env_name, cfg_data=None, **kwargs):
    """Initialize data loaders for the specified environment.

    Loads base config from eb_jepa/datasets/{env_name}/data_config.yaml
    and merges with any overrides from cfg_data.

    Args:
        env_name: Name of the environment ("two_rooms" or "hft_timeseries").
        cfg_data: Configuration overrides for the dataset.

    Returns:
        Tuple of (train_loader, val_loader, config).
    """
    if env_name == "hft_timeseries":
        return _init_hft_timeseries(cfg_data)
    if env_name != "two_rooms":
        raise ValueError(f"Unknown env: {env_name}. Use 'two_rooms' or 'hft_timeseries'.")

    merged_cfg = load_env_data_config(env_name, cfg_data)
    config = update_config_from_yaml(WallDatasetConfig, merged_cfg)

    num_workers = merged_cfg.get("num_workers", 0)
    pin_mem = merged_cfg.get("pin_mem", False)
    persistent_workers = merged_cfg.get("persistent_workers", False) and num_workers > 0

    dset = WallDataset(config=config)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
    )

    val_dset = WallDataset(config=config)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
    )

    return loader, val_loader, config
