# HFT World Model

High-frequency world model for market microstructure. Predicts the next market state from the current state and order flow imbalance.

## What We Are Doing

1. **Simple world model (Phase 1)**  
   Learn to predict next state (dPrice, traded volume, bid-ask spread) from current state and imbalance. No JEPA.

2. **JEPA-based world model (Phase 2)**  
   Use [EB-JEPA](https://github.com/facebookresearch/eb_jepa) AC-Video-JEPA adapted to time series. Run:
   ```bash
   uv run python -m examples.ac_timeseries_jepa.main
   ```
   **Interactive notebook**: `notebooks/ac_timeseries_jepa_explore.py` — explore data, quick train, visualize.

## Setup

```bash
chmod +x launch.sh
./launch.sh
uv pip install -e .
```

Set `FOLDER_PATH` in `.env` to your Databento MBP data directory.

## Plan

See `refs/plan.md` for the full plan (state space, control, data format, roadmap).

## License

MIT. Supervisors: Mathieu Rosenbaum, Charles-Albert Lehalle. Data: Databento.
