# Combined System: Direction 1 + Direction 2

Implements the combination step from the original technical plan: "feed
the structured-sequence input into the unfolded architecture, and show
the combined system against: (a) the original paper's Deep LAS, (b)
Direction-1-only, (c) Direction-2-only."

## Design

Both Direction 1's pooling net and Direction 2's Option A architecture
are **frozen/reused** (not jointly retrained) -- confirmed choice, for
speed and lower risk given how much debugging each direction alone
needed. The combination point: Direction 1's pooling net's hard-decided
LLR is mapped back to the nearest PAM levels per real dimension, giving
Direction 2's unfolded search a **context-informed initial estimate**
instead of the plain single-subcarrier hard MMSE estimate it was
validated with standalone.

See `simulateBERCombinedFourArms.m`'s header for the full per-arm
breakdown and why reusing Direction 2's i.i.d.-flat-trained temperatures
on multipath-drawn subcarriers is valid (each subcarrier's own channel
is still marginally CN(0,1) regardless of cross-subcarrier correlation
-- the same reasoning Direction 1 already relied on to reuse its own
flat-trained MLP without retraining).

## Files

| File | Purpose |
|---|---|
| `simulateBERCombinedFourArms.m` | The core evaluator -- all four arms on shared correlated-channel realizations (paired comparison) |
| `run_combined_vs_all.m` | Driver: loads frozen Direction 1 net, trains fresh Direction 2 net (same validated recipe) and the original baseline, runs the four-way comparison |

## Setup

Edit the three paths near the top of `run_combined_vs_all.m`: `origDir`,
`dir1Path`, `dir2Path`. Requires `direction1_structured_gru/result/best_pooling_net.mat`
to already exist (run that direction's driver scripts first if not).

## Run

```matlab
run_combined_vs_all.m
```

Trains a fresh Direction 2 model as part of this run (~1 hour at the
validated quickMode scale), plus the original baseline's GRU (fast) --
budget more time than a single-direction run.

## Before trusting the result

**Same rule as every other result in this project:** this is one run.
If arm (c) Direction-2-only and arm (d) Combined come out close, do not
report a winner -- adapt `run_direction2_stability_analysis.m`'s
multi-repeat mean+/-std approach to this four-arm comparison before
writing anything up. Given time constraints, prioritize this check for
the (c) vs (d) comparison specifically -- that's the one that actually
tests the combination's value; (a)/(b) are already independently
validated in their own directions.
