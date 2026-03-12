# HFT World Model

High-frequency world model for market microstructure. Predicts the next market state from the current state and order flow imbalance. Uses [JEPA](https://github.com/facebookresearch/eb_jepa) (Joint Embedding Predictive Architecture): learn embeddings and predict in embedding space instead of raw state space.

---

## Architecture Overview

### Phase 1: Simple world model

```
┌─────────────────────────────────────────────────────────────────┐
│  state_t  │  imbalance_t  │  →  [ MLP / LSTM ]  →  state_{t+1}  │
└─────────────────────────────────────────────────────────────────┘
```

Direct prediction: concatenate current state (dPrice, volume, spread, etc.) and control (imbalance), predict the next state.

---

### Phase 2: JEPA-based world model

**JEPA** (Joint Embedding Predictive Architecture): learn embeddings and a predictor in latent space instead of predicting raw observations. This improves generalization and supports planning.

```
                    ┌─────────────────────┐
                    │   state sequence     │  (dPrice, volume, spread per bar)
                    │   [B, C=3, T, 1, 1]  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  TimeSeriesEncoder   │  MLP, per-timestep
                    │  (MLP + LayerNorm)   │
                    └──────────┬──────────┘
                               │
                    embeddings z_t [B, D, T, 1, 1]
                               │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
  ┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐
  │  VC reg      │   │  RNNPredictor   │   │  IDM (optional)   │
  │  std + cov   │   │  z_t + action   │   │  predict action   │
  │  sim_t       │   │  → ẑ_{t+1}     │   │  from z_t, z_{t+1}│
  └──────────────┘   └────────┬────────┘   └──────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Cosine loss         │  pred_coeff * (1 - cos(z_{t+1}, ẑ_{t+1}))
                    │  target: z_{t+1}     │
                    └─────────────────────┘
```

- **State**: dPrice, traded volume, bid-ask spread (aggregated per 1-minute bar).
- **Control**: order flow imbalance `(bid_vol - ask_vol) / (bid_vol + ask_vol)`.
- **Encoder**: maps raw states to embeddings.
- **Predictor**: GRU takes current embedding + action, outputs predicted next embedding.
- **Regularizer** (VC_IDM_Sim): variance-covariance (std + cov), temporal similarity (sim_t), inverse dynamics (IDM) to avoid collapse and improve representations.

---

## What We Are Doing

1. **Phase 1 — Simple world model**  
   Train an MLP or LSTM to predict next state from (state, imbalance). No embeddings.

2. **Phase 2 — JEPA-based world model**  
   Use [EB-JEPA](https://github.com/facebookresearch/eb_jepa) adapted to time series:
   - `TimeSeriesEncoder` + `RNNPredictor` + `VC_IDM_Sim_Regularizer`
   - Train with cosine prediction loss + regularizer

   Run:

   ```bash
   uv run python -m examples.ac_timeseries_jepa.main
   ```

   **Interactive notebook**: `notebooks/ac_timeseries_jepa_explore.py` — explore data, quick train, visualize.

---

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
