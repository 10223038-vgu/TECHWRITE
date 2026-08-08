# Direction 1: From Structured GRU to Pooled-Context Deep LAS

**Status summary (read this first):** the original "structured GRU"
hypothesis -- that a recurrent net should exploit the *order* of
correlated neighboring subcarriers -- was tested and **falsified** by
its own ablation (Part A below). The correlation-*content* half of the
hypothesis held up, though, so the mechanism was swapped from a GRU to a
permutation-invariant pooling network, which beat the GRU outright but
only reached rough parity with our reproduced Original Deep LAS. The
current iteration adds distance-aware pooling (near/far split + a
distance-weighted mean, instead of one uniform pool) -- run
`run_distance_pooling_vs_original.m` and check its printed SNR-gain
table for the current headline result. See `THEORY_POOLING.md` for the
full theory/reasoning behind every step.

See `THEORY.md` for the original derivation (Sections 1.1-1.6 are still
the foundation) and `THEORY_POOLING.md` for the pooling-net theory that
supersedes Section 1.7's original GRU-order hypothesis.

## The story, in order

1. **`run_direction1_ablation.m`** trains three GRU arms (correlated
   order / shuffled order, same content / i.i.d., no correlation) to
   test whether order matters. Result: correlated vs. shuffled BER gap
   flips sign across the SNR sweep (noise, not a real effect); both
   clearly beat i.i.d. (real, consistent effect). **Order doesn't
   matter; correlated content does.**
2. **`run_direction1_vs_original.m`** compares the correlated GRU against
   our reproduced Original Deep LAS + classical baselines. The GRU
   underperforms the original at every SNR and visibly floors out at
   high SNR -- consistent with (1): a recurrent architecture is a hard,
   sample-inefficient way to learn a function the data says is actually
   order-invariant.
3. **`run_pooling_vs_gru.m`** replaces the GRU with a permutation-
   invariant pooling network (mean/max/std over the same context window,
   see `THEORY_POOLING.md` Section 3) and compares the two directly on
   identical data. The pooling net beats the GRU at every SNR point, by
   a **widening** margin at high SNR where the GRU floors out.
4. **`run_pooling_vs_original.m`** is the test that actually matters: does
   the pooling net beat the Original Deep LAS it's supposed to improve
   on, not just the GRU? Result: **rough parity** -- the two curves
   track closely across the full sweep. Real and honest, but short of a
   clear win. `THEORY_POOLING.md` Section 5-6 lays out why (naive
   pooling is distance-blind -- it weights a barely-correlated far
   neighbor the same as a strongly-correlated near one) and proposes
   distance-aware pooling as the next step.
5. **`run_distance_pooling_vs_original.m`** implements that next step:
   near/far split pooling plus an exponentially-decaying distance-
   weighted mean (`THEORY_POOLING.md` Section 6), and compares it
   three-way against the uniform pooling net and the Original
   reproduction on identical data. This is the current state of the art
   in this folder -- check its printed SNR-gain table's "Distance
   pooling vs Original" row for the actual headline result.
6. **`run_stability_analysis.m`** is required before trusting ANY
   high-SNR (16-20 dB) claim from step 5: repeated single runs showed the
   Original-vs-pooling ordering at high SNR flips from run to run,
   meaning that region is dominated by run-to-run noise (channel draws,
   training-set shuffling, weight initialization), not a real effect.
   This script retrains everything from scratch across multiple
   independent repeats and reports mean +/- std BER curves plus a
   per-SNR significance check, instead of eyeballing single noisy runs.

## Files

| File | Purpose |
|---|---|
| `THEORY.md` | Original derivation (multipath model, Lseq design rule, Sections 1.1-1.6 still current) |
| `THEORY_POOLING.md` | The ablation result, why the GRU was replaced, the pooling theory, and the distance-aware next step (Section 6) |
| `genMultipathChannel.m` | Correlated per-subcarrier channel generator |
| `coherenceBandwidth.m` | `Bc = N/L` -- sets the sequence-length design rule |
| `buildCorrelatedAndShuffledSequences.m` | Paired GRU dataset builder for the core ablation (identical content, different order; also used for the correlated GRU arm generally) |
| `buildIIDSequences.m` | i.i.d. baseline arm (no correlation, matches the original codebase) |
| `trainGRUFromSequences.m` | Shared GRU trainer for all GRU-based arms |
| `buildPooledContextFeatures.m` | Builds the fixed-size, permutation-invariant UNIFORM pooled feature set (supersedes the GRU sequence builder) |
| `buildDistanceWeightedPooledFeatures.m` | Distance-aware extension: near/far split pooling + distance-weighted mean (THEORY_POOLING.md Section 6) |
| `pooledFeaturesForOneWindow.m` | Shared distance-aware feature-construction logic, used identically by the builder above and its BER evaluator so train/test can't drift apart |
| `trainPoolingNet.m` | Trains the plain feedforward network on pooled features (used by both the uniform and distance-aware arms -- it's agnostic to feature layout) |
| `simulateBERDirection1.m` | BER evaluator matching the GRU's sequence-shaped training input (`mode`: correlated / shuffled / iid) |
| `simulateBERPooling.m` | BER evaluator matching the uniform pooling net's fixed-size input |
| `simulateBERDistancePooling.m` | BER evaluator matching the distance-aware pooling net's input |
| `paperReferenceNumbers.m` | Digitized headline numbers from the original paper's text |
| `snrAtTargetBER.m` | Interpolates SNR needed to hit a target BER (the paper's own metric) |
| `run_direction1_ablation.m` | Runs the H1/H2/H3 GRU ablation -- step 1 above |
| `run_direction1_vs_original.m` | GRU vs. Original vs. baselines -- step 2 above |
| `run_pooling_vs_gru.m` | Pooling net vs. GRU, head-to-head -- step 3 above |
| `run_pooling_vs_original.m` | Uniform pooling net vs. Original vs. baselines -- step 4 above |
| `run_distance_pooling_vs_original.m` | Distance-aware pooling vs. uniform pooling vs. Original vs. baselines -- step 5 above |
| `run_stability_analysis.m` | **Run this before trusting any high-SNR claim** -- multi-repeat mean +/- std BER curves and per-SNR significance check -- step 6 above |

## Setup
All driver scripts need the original codebase on the MATLAB path. Each
has an `originalCodePath` / `origDir` variable near the top -- **edit
that line** to point at wherever your `original/deep-las-matlab` (or
equivalent) folder actually lives on your machine. They'll error with a
clear message (instead of failing silently) if the path is wrong.

## Run order
```matlab
run_direction1_ablation.m           % H1/H2/H3 GRU ablation; caches the correlated GRU net
run_direction1_vs_original.m        % GRU vs Original vs baselines (uses cached GRU if present)
run_pooling_vs_gru.m                % Pooling net vs GRU head-to-head; caches the pooling net
run_pooling_vs_original.m           % Pooling net vs Original vs baselines
run_distance_pooling_vs_original.m  % Distance-aware pooling vs uniform pooling vs Original (current best)
run_stability_analysis.m            % REQUIRED before trusting any high-SNR claim from the step above
```

## What we compare, and why

**1. BER vs SNR curves** (primary plot) -- MMSE (hard, sanity floor),
Conv. LAS (hard), our reproduced Original Deep LAS, and our "ours" arm
(GRU or pooling net depending on script), all on one semilog plot. Same
metric/plot style as the paper's own Fig. 9.

**2. SNR required to reach BER=1e-4 and BER=1e-5** -- computed via
`snrAtTargetBER.m` from our own measured curves. We use these specific BER
targets because they are exactly what the paper reports its headline
numbers against ("2.55 dB / 3 dB gain ... maintaining BER of 1e-4"; "0.4
dB / 1.2 dB gap to SD ... to achieve BER of 1e-5").

**3. SNR gain relative to Conv. LAS at each target** -- this is the paper's
own metric, computed several ways so they're all in one table:
   - **Paper's reported gain** (Original Deep LAS vs Conv. LAS, from the text)
   - **Our reproduction's gain** (should land close to the paper's number --
     this is a sanity check on the reproduction itself)
   - **Our "ours" arm's gain** vs Conv. LAS
   - **"Ours" vs our own Original reproduction** -- the cleanest
     apples-to-apples comparison, since both numbers come from the exact
     same codebase/hardware/random seeds and differ only in the
     detector's architecture, isolating our contribution from any
     reproduction-fidelity noise relative to the paper.

We deliberately do NOT try to overlay the paper's raw BER curve point-by-
point, because we don't have their digitized data -- only the summary
numbers they printed in text. Comparing against those reported numbers,
side by side with our own reproduction and improvement, is the honest way
to show "does our baseline match what they claimed, and does our
improvement move further in the same direction" without overstating what
we can verify.

## Next step

Distance-aware pooling (`run_distance_pooling_vs_original.m`) is now
implemented -- run it and check `result/distance_pooling_snr_gain_comparison.csv`:

- **If "DistancePooling vs Orig" is clearly positive:** the distance
  structure was the missing piece -- write this up as the real result.
- **If it's positive but small, or "DistancePooling vs UniformPooling"
  is positive while "vs Orig" still isn't:** distance-awareness helped
  but hasn't closed the gap yet -- try tuning `nearFrac` (currently 0.25)
  and `decayScale` (currently `Lseq/2`) in
  `run_distance_pooling_vs_original.m`'s top-of-script constants before
  concluding the ceiling is real.
- **If it's flat/negative vs. uniform pooling too:** the near/far split
  and decay-weighting didn't capture useful structure either -- at that
  point it's worth seriously entertaining that correlation exploitation,
  at least via pooled per-subcarrier features, has hit its ceiling
  against the original architecture, and a different kind of feature
  entirely (not just a different aggregation of the same `UG` features)
  would be needed to go further.

