import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=force,
    )
    return logging.getLogger(name=name)


def add_file_handler(log_dir: str | Path) -> Path:
    """Add file handler to root logger. Returns path to log file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{ts}.log"
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logging.getLogger().addHandler(handler)
    return log_path
