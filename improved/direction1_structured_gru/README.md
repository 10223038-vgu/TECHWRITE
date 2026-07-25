# Direction 1: Structured GRU Sequence Input

See `THEORY.md` in this folder for the full mathematical derivation. This
folder implements and tests that theory.

## Files

| File | Purpose |
|---|---|
| `THEORY.md` | Full derivation (copied from repo root) |
| `genMultipathChannel.m` | Correlated per-subcarrier channel generator |
| `coherenceBandwidth.m` | `Bc = N/L` -- sets the sequence-length design rule |
| `buildCorrelatedAndShuffledSequences.m` | Paired dataset builder for the core ablation (identical content, different order) |
| `buildIIDSequences.m` | Baseline arm (no correlation, matches the original codebase) |
| `trainGRUFromSequences.m` | Shared GRU trainer for all three ablation arms |
| `paperReferenceNumbers.m` | Digitized headline numbers from the original paper's text |
| `snrAtTargetBER.m` | Interpolates SNR needed to hit a target BER (the paper's own metric) |
| `run_direction1_ablation.m` | Runs H1/H2/H3 (see THEORY.md Section 1.7-1.8) |
| `run_direction1_vs_original.m` | **The comparison script** -- run this to get the plots/table below |

## Setup
Both driver scripts (`run_direction1_ablation.m`, `run_direction1_vs_original.m`)
need the original codebase on the MATLAB path. Each has an
`originalCodePath` / `origDir` variable near the top -- **edit that line**
to point at wherever your `original/deep-las-matlab` (or equivalent) folder
actually lives on your machine. They'll error with a clear message
(instead of failing silently) if the path is wrong.

## Run order
```matlab
run_direction1_ablation.m        % proves the mechanism (H1/H2/H3), caches best GRU net
run_direction1_vs_original.m     % produces the comparison plot + table (uses cached net if present)
```

## What we compare, and why

**1. BER vs SNR curves** (primary plot) -- MMSE (hard, sanity floor),
Conv. LAS (hard), our reproduced Original Deep LAS, and our Direction 1
Deep LAS, all on one semilog plot. This is the same metric/plot style as
the paper's own Fig. 9.

**2. SNR required to reach BER=1e-4 and BER=1e-5** -- computed via
`snrAtTargetBER.m` from our own measured curves. We use these specific BER
targets because they are exactly what the paper reports its headline
numbers against ("2.55 dB / 3 dB gain ... maintaining BER of 1e-4"; "0.4
dB / 1.2 dB gap to SD ... to achieve BER of 1e-5").

**3. SNR gain relative to Conv. LAS at each target** -- this is the paper's
own metric, computed three ways so they're all in one table:
   - **Paper's reported gain** (Original Deep LAS vs Conv. LAS, from the text)
   - **Our reproduction's gain** (should land close to the paper's number --
     this is a sanity check on the reproduction itself)
   - **Our Direction 1's gain** (the actual novelty claim)
   - **Direction 1 vs our own Original reproduction** -- this is the
     cleanest apples-to-apples comparison, since both numbers come from
     the exact same codebase/hardware/random seeds and differ only in the
     GRU's sequence input, isolating our contribution from any
     reproduction-fidelity noise relative to the paper.

We deliberately do NOT try to overlay the paper's raw BER curve point-by-
point, because we don't have their digitized data -- only the summary
numbers they printed in text. Comparing against those reported numbers,
side by side with our own reproduction and improvement, is the honest way
to show "does our baseline match what they claimed, and does our
improvement move further in the same direction" without overstating what
we can verify.
