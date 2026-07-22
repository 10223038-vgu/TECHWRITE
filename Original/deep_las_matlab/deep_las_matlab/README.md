# MATLAB Package: Soft-Output Deep LAS for Coded MIMO Systems

Implementation scaffold for reproducing the simulations in Ullah et al.,
*"Soft-Output Deep LAS Detection for Coded MIMO Systems: A Learning-Aided
LLR Approximation,"* IEEE TVT, 2024.

**Important:** this code was written directly from the paper's equations
and standard MATLAB/Toolbox APIs, but could not be executed in the
environment that produced it (no MATLAB/Octave available here). Treat it
as a correct-by-construction starting scaffold, not verified output —
run `run_quick_demo.m` first and fix any local syntax/version issues
before trusting larger sweeps.

## Changelog (bug fixes since first delivery)

- **`trainMLP.m`**: `trainParam.max_fail` must be finite in MATLAB
  (was `Inf`, now `max(epochs,1000)`).
- **`trainGRU.m` / `deepLASPredict.m`**: `trainNetwork`/`predict` require
  a plain numeric matrix (not a cell array) for sequence-to-one
  regression responses when using `'OutputMode','last'`. Fixed both
  sides of this interface.
- **`run_fig4_trainingLoss.m`**: `trainInfo.TrainingLoss` from
  `trainNetwork` is logged per mini-batch **iteration**, not per epoch
  — plotting it raw gave a noisy, mislabeled x-axis. Now averaged into
  per-epoch values before plotting, and the MLP panel got a log-scale
  view plus a data-driven zoom instead of a hardcoded, likely-empty
  `ylim`.
- **Root-cause fix for the Figs. 5-8 "Rx symbol imbalance" AND the
  Fig. 9 near-random MLP-only/Deep-LAS behavior**: `initEstimate.m`
  returns a **hard-decided** (quantized) equalizer output. That was
  being fed into the MLP/GRU as a training feature and plotted as
  "Rx symbol" everywhere. Once hard-decided, every noise realization
  that quantizes to the same constellation point becomes identical —
  which (a) explains why the Fig. 5-8 x-axis collapsed onto a few
  discrete points (for 4-QAM, the real part of a hard decision is
  only ever ±1), and (b) removes exactly the information a soft-output
  network needs to estimate LLR confidence, structurally capping
  MLP-only/Deep-LAS performance near a trivial baseline regardless of
  training. Added `initEstimateSoft.m` (same ZF/MMSE math, no
  hard-decision step) and switched every DNN-facing consumer to use
  it: `generateTrainingData.m`, `deepLASPredict.m`,
  `simulateBER.m`'s `'mlp-only'` case, and the Fig. 5-8 scatter script.
  `initEstimate.m` (hard) is still used, correctly, to seed the LAS
  search itself (`las1Hard.m`/`softOutputLAS.m`), which does need an
  integer grid point to start from.
- **`run_fig9_BERcomparison.m`**: added an `'mmse-hard'` sanity
  baseline curve (a detector-free/ML-free lower bound — if a learned
  detector ever sits above this line, something is broken), and raised
  the default block/sample counts.
- **`run_fig11_sweeps.m`**: finer SNR step (1 dB instead of 2 dB) and
  multi-run averaging to smooth the curve.
- **Added `run_fig12_dataDrivenComparison.m` and `trainSeqModel.m`**:
  Fig. 12 (Deep LAS vs. LSTM/Bi-LSTM/GRU/MLP-only baselines) was
  described in the original guide but the driver script was never
  actually written — added now.

## Requirements
- MATLAB R2023a (or similar)
- Communications Toolbox (`qammod`, `qamdemod`, `pammod`, `pamdemod`)
- Deep Learning Toolbox (`trainNetwork`, `gruLayer`, `lstmLayer`,
  `bilstmLayer`, `sequenceInputLayer`)
- Statistics and Machine Learning / Deep Learning Toolbox (`fitnet`, `train`)
- (Optional, for coded BER) Communications Toolbox turbo coding objects

## File Map

| File | Purpose | Paper reference |
|---|---|---|
| `getConfig.m` | Common system parameters | Section V |
| `genChannel.m` | Rayleigh MIMO channel generator | Eq. 2 |
| `qamHelpers.m` | QAM modulation / hard-decision / PAM levels | Section II, Eqs. 5-6 |
| `complexToReal.m` | Complex -> real-valued equivalent MIMO system | (standard LAS representation) |
| `initEstimate.m` | ZF/MMSE **hard-decided** estimate (for LAS seeding only) | Eqs. 5-6 |
| `initEstimateSoft.m` | ZF/MMSE **soft** (unquantized) estimate (for DNN features / plots) | Eqs. 5-6, adapted |
| `lasSearchCore.m` | Shared coordinate-descent LAS search engine | Eqs. 7-13 |
| `las1Hard.m` | Conventional hard-output 1-LAS (Algorithm 1) | Algorithm 1 |
| `pamBitTable.m` | Gray-coded PAM level/bit lookup | (bit mapping for Eq. 4) |
| `modLASCounter.m` | Counter-hypothesis search for one bit (Algorithm 2) | Algorithm 2 |
| `softOutputLAS.m` | Full two-step model-based soft-output LAS + LLR | Section III, Eq. 4b |
| `symbolsToBits.m` | Ground-truth bit matrix from symbols | (bit layout helper) |
| `generateTrainingData.m` | Build Deep LAS training/validation datasets | Eq. 19 |
| `trainMLP.m` | Train MLP block (rough LLR estimator) | Section IV-A, Eqs. 20-21, Alg. 3 |
| `trainGRU.m` | Train GRU block (refines MLP estimate, Deep LAS hyperparams) | Section IV-A, Eqs. 22-23, Fig. 2 |
| `trainSeqModel.m` | Generalized GRU/LSTM/Bi-LSTM trainer (Fig. 12 baselines) | Section V-C |
| `deepLASPredict.m` | Online Deep LAS inference (works with any trained seq net) | Eq. 23 |
| `simulateBER.m` | Configurable Monte-Carlo BER loop | Figs. 9-12 |
| `run_quick_demo.m` | Fast smoke test of the whole pipeline | - |
| `run_fig4_trainingLoss.m` | MLP/GRU layer-count training-loss sweep | Fig. 4 |
| `run_fig5to8_LLRscatter.m` | Actual vs approximated LLR scatter plots | Figs. 5-8 |
| `run_fig9_BERcomparison.m` | Conv. LAS vs Soft-LAS vs Deep LAS vs MLP-only vs MMSE | Fig. 9 |
| `run_fig11_sweeps.m` | BER vs antenna count (and FFT-length notes) | Fig. 11 |
| `run_fig12_dataDrivenComparison.m` | Deep LAS vs LSTM/Bi-LSTM/GRU/MLP-only, 4-QAM & 16-QAM | Fig. 12 |

Not included (left as extensions, see below): DetNet and optimal soft-output
sphere decoding baselines for **Fig. 10**, and full turbo-coded BER for
**Figs. 9-12** absolute numbers (see caveats below).

## Quick Start

```matlab
addpath(genpath(pwd));
run_quick_demo.m          % ~1 minute, sanity-checks the full pipeline
```

Then, for each modulation order you care about:

```matlab
generateTrainingData(4,  0:2:14, 3000, 'train_4QAM.mat');   % raise sample count for real runs
generateTrainingData(16, 0:2:14, 3000, 'train_16QAM.mat');

mlpNet4  = trainMLP('train_4QAM.mat', 2, 10, 300);
gruNet4  = trainGRU('train_4QAM.mat', mlpNet4, 2, 100, 40);
```

Then run any `run_fig*.m` script, or call `simulateBER(...)` / `softOutputLAS(...)`
directly for custom experiments.

## Design Choices / Simplifications (read before trusting numbers)

1. **Real-valued equivalent LAS search.** The paper writes Algorithm 1/2
   in complex notation but LAS-type neighborhood search operates on the
   real PAM grid per dimension; this code converts to the standard
   real-valued equivalent system (`complexToReal.m`) before running the
   search, which is mathematically consistent with Eq. 19's real/imag
   vectorization used for the DNN.

2. **Coordinate-descent LAS core.** Rather than replicating the paper's
   closed-form step-size shortcut (Eq. 9, `λ = 2⌊z/(2q)⌉`) verbatim —
   which is fragile at grid boundaries — `lasSearchCore.m` evaluates the
   cost for every candidate PAM level per dimension directly (Eq. 10) and
   greedily accepts the best decrease. This is the same "single-symbol
   update, monotonically decreasing likelihood" algorithm described in
   the text, just computed robustly.

3. **Algorithm 2's efficiency shortcut.** Rather than re-running a full
   global search for both bit hypotheses, Step-1's global minimum (`F0`)
   is reused as one bit hypothesis's cost, and only the counter
   hypothesis (`F1`) is re-searched per bit — this matches the paper's
   stated complexity savings versus brute force.

4. **GRU/LSTM/Bi-LSTM sequence structure.** The paper describes "2Nt+1
   sequences each of 1×N length" as the GRU input, which is ambiguous
   about what forms a timestep (OFDM subcarriers? Monte-Carlo pool
   index?). `trainGRU.m` and `trainSeqModel.m` (used for the Fig. 12
   LSTM/Bi-LSTM/GRU baselines) both implement this as one timestep per
   training sample (a functioning, but interpretive, simplification).
   If you want literal per-OFDM-block sequences (8 symbols/block,
   `cfg.FFT_len` subcarriers), restructure `XTrainSeq`/`YTrainAll` in
   both files to group `cfg.symbolsPerBlock` consecutive samples into
   one multi-timestep sequence.

5. **Channel coding.** `simulateBER.m` reports **uncoded** uncoded BER at
   the detector's hard-decision output. This preserves the relative SNR
   gaps between detectors that Figs. 9-12 are about, without depending on
   an unspecified turbo puncturing pattern for rate 1/2. To reproduce
   *absolute* coded BER numbers, add rate-1/2 turbo coding, e.g.:

   ```matlab
   trellis = poly2trellis(4, [13 15], 13);
   intrlvr = randperm(frameLen);
   turboEnc = comm.TurboEncoder('TrellisStructure', trellis, ...
                                 'InterleaverIndices', intrlvr);
   turboDec = comm.TurboDecoder('TrellisStructure', trellis, ...
                                 'InterleaverIndices', intrlvr, ...
                                 'NumIterations', 8);
   % encode bits -> map to QAM -> channel -> detector -> LLR
   % -> turboDec(-LLR) [check sign convention against your LLR definition]
   % -> compare decoded bits to source bits
   ```
   You'll need to tune the code rate/puncturing to hit exactly rate 1/2
   (the base `comm.TurboEncoder` trellis above is rate 1/3 before
   puncturing) — the paper doesn't specify its puncturing pattern.

6. **DetNet / optimal soft-output SD (Fig. 10 baselines).** Not
   implemented here — DetNet is an unfolded projected-gradient network
   (see Samuel et al., refs [35],[39]) and the soft-output SD is a
   depth-first sphere search tracking both the ML and counter-hypothesis
   costs (see ref [9]). Both are self-contained additions you can drop
   into `simulateBER.m` as new `detectorType` cases once implemented.

7. **Antenna sweep scale (Fig. 11b).** The paper sweeps up to
   Nt=Nr=256; `run_fig11_sweeps.m` defaults to smaller sizes since the
   LAS search here is O(Nt²) per accepted update in pure MATLAB — profile
   and vectorize `lasSearchCore.m` (or MEX it) before attempting the
   largest antenna counts.

8. **Fig. 12 runtime.** The paper's LSTM/Bi-LSTM/GRU baselines use 800
   hidden units and 500 training epochs each (Section V-C) — three
   such networks per modulation order is substantially heavier than
   the rest of this package. `run_fig12_dataDrivenComparison.m`
   exposes `seqHiddenUnits`/`seqEpochs` at the top so you can turn
   them down for a fast smoke test before committing to the full run.

## Sanity Checks Before Trusting a Figure

- `softOutputLAS.m` on a single high-SNR sample should return `xhat`
  equal (or very close) to the transmitted `x`, and `LLR` values that
  are large in magnitude with the correct sign relative to the true bits
  (`sign(LLR) == +1` when the true bit is 0).
- `simulateBER('mmse-hard', ...)` should be the *worst* curve; `deeplas`
  and `softlas` should sit below `convlas-hard`, consistent with Fig. 9.
