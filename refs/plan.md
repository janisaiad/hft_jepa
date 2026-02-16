# HFT World Model Plan

## Goal

Build a **high-frequency world model** for market microstructure: a model that predicts the next state of the market given the current state and control (order flow imbalance).

---

## State Space

At **1-minute scale**, the state is:

| Variable | Description | Source |
|----------|-------------|--------|
| **dPrice** | Price change (return) over the interval | Mid-price from `bid_px_00`, `ask_px_00` |
| **Traded volumes** | Volume exchanged in the interval | Sum of `size` where `action` = Trade |
| **Bid-ask spread** | `ask_px_00 - bid_px_00` at interval end | `bid_px_00`, `ask_px_00` |
| **bid_sz, ask_sz** | Top-of-book sizes at bar end | `bid_sz_00`, `ask_sz_00` |
| **bid_ct, ask_ct** | Top-of-book order counts | `bid_ct_00`, `ask_ct_00` |
| **num_trades** | Number of trades in the bar | Count of Trade events |

---

## Action / Control

| Variable | Description | Formula |
|----------|-------------|---------|
| **Imbalance** | Order flow imbalance over the interval | `(bid_volume - ask_volume) / (bid_volume + ask_volume)` or similar, from `bid_sz_N`, `ask_sz_N`, `side` on Trade events |

---

## Data Format (Databento MBP)

See `refs/format.md` for full schema. Key fields:

- `ts_event`, `ts_recv`: timestamps (nanoseconds since UNIX epoch)
- `price`: int64, 1 unit = 1e-9
- `size`: uint32, order quantity
- `action`: Add, Cancel, Modify, Trade, Fill, cleaR book
- `side`: Ask, Bid, None
- `bid_px_N`, `ask_px_N`: prices at level N
- `bid_sz_N`, `ask_sz_N`: sizes at level N
- `bid_ct_N`, `ask_ct_N`: order counts at level N

---

## Example Raw Data (AAL, 2023-07-18)

```
2023-07-18T08:00:00.030Z  10221  AN  017.88  300  0  1  30163758299423  17.88  300  0  1  0  0  0  0  0  0  0  0  0  0  0  0  AAL  2023-07-18T00:00:00.000Z  2023-07-18T08:00:00.030Z
2023-07-18T08:00:00.031Z  10221  AN  029.71  0   0  1  30164834299474  17.88  29.73  0  0  1  1  0  0  0  0  0  0  0  0  0  0  0  0  AAL  2023-07-18T00:00:00.000Z  2023-07-18T08:00:00.031Z
2023-07-18T08:00:02.063Z  10221  CN  017.88  300  0  1  30164043308351  29.70  1    0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  AAL  2023-07-18T00:00:00.000Z  2023-07-18T08:00:02.064Z
2023-07-18T08:00:02.063Z  10221  ...
```

---

## Roadmap

1. **Phase 1: Simple world model**
   - Aggregate raw MBP ticks to 1-minute bars
   - Compute state: dPrice, traded volume, bid-ask spread
   - Compute control: imbalance
   - Train a simple predictive model (e.g. MLP, LSTM) to predict next state from (state, action)

2. **Phase 2: JEPA-based world model**
   - Use [EB-JEPA](https://github.com/facebookresearch/eb_jepa) (Energy-Based Joint Embedding Predictive Architecture)
   - Adapt [AC-Video-JEPA](https://github.com/facebookresearch/eb_jepa/tree/main/examples/ac_video_jepa) to time series
   - `examples/ac_timeseries_jepa`: TimeSeriesEncoder, RNNPredictor, VC_IDM_Sim_Regularizer
   - Learn joint embeddings for prediction (planning TBD)

---

## References

- Data: Databento MBP (Market-By-Price)
- JEPA: [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa)
- Supervisors: Mathieu Rosenbaum, Charles-Albert Lehalle
