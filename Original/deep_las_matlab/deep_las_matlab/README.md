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

- **CUDA crash fix**: `trainGRU.m` / `trainSeqModel.m` now force
  `'ExecutionEnvironment','cpu'` in `trainingOptions`, since the
  `CUDA_ERROR_UNKNOWN` crash is a local GPU driver/toolbox mismatch,
  not a code bug. Switch back to `'auto'` once your GPU setup is
  confirmed working.
- **Fig. 9 shallow/noisy high-SNR tail fixed**: `simulateBER.m` was
  using a fixed block count, which sees very few actual bit errors at
  high SNR (e.g. 0-2 errors out of a few thousand bits at 12-14 dB) —
  that's almost certainly why the BER drop-off looked worse than the
  paper's. `simulateBER.m` now runs adaptively: it keeps simulating
  until a target minimum number of *actual errors* has been observed
  at every SNR point (default 150), capped by a max block count, which
  is standard practice for reliable BER curves. SNR resolution was
  also finened (1 dB steps).
- **Added Fig. 10** (was never implemented before): `run_fig10_modelBasedComparison.m`,
  plus two new supporting files: `bruteForceSD.m` (exact optimal
  soft-output detector via brute-force search — see below for why this
  is equivalent to sphere decoding here) and `trainDetNet.m` /
  `detNetPredict.m` (DetNet baseline — **this is the most experimental
  part of the whole package**, see the dedicated caveat below).
- **Fig. 11(a) (FFT length) actually implemented**: previously this
  was just a note saying it was out of scope. `trainGRU.m` now accepts
  an optional `seqLen` argument that pools consecutive samples into
  multi-timestep GRU training sequences, and `run_fig11_sweeps.m` maps
  FFT length to `seqLen` as a documented proxy. Fig. 11(b) (antenna
  count) also got the same adaptive-stopping/finer-SNR treatment as
  Fig. 9.
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
| `bruteForceSD.m` | Exact optimal soft-output detector (exhaustive search) | Eq. 3-4, "SD" baseline for Fig. 10 |
| `trainDetNet.m` | DetNet training (custom dlarray loop) **-- experimental** | refs [35]/[39], Fig. 10 |
| `detNetPredict.m` | DetNet inference (hard bits) | refs [35]/[39], Fig. 10 |
| `symbolsToBits.m` | Ground-truth bit matrix from symbols | (bit layout helper) |
| `generateTrainingData.m` | Build Deep LAS training/validation datasets | Eq. 19 |
| `trainMLP.m` | Train MLP block (rough LLR estimator) | Section IV-A, Eqs. 20-21, Alg. 3 |
| `trainGRU.m` | Train GRU block (refines MLP estimate, Deep LAS hyperparams; supports `seqLen` pooling for Fig. 11a) | Section IV-A, Eqs. 22-23, Fig. 2 |
| `trainSeqModel.m` | Generalized GRU/LSTM/Bi-LSTM trainer (Fig. 12 baselines) | Section V-C |
| `deepLASPredict.m` | Online Deep LAS inference (works with any trained seq net) | Eq. 23 |
| `simulateBER.m` | Configurable **adaptive** Monte-Carlo BER loop | Figs. 9-12 |
| `run_quick_demo.m` | Fast smoke test of the whole pipeline | - |
| `run_fig4_trainingLoss.m` | MLP/GRU layer-count training-loss sweep | Fig. 4 |
| `run_fig5to8_LLRscatter.m` | Actual vs approximated LLR scatter plots | Figs. 5-8 |
| `run_fig9_BERcomparison.m` | Conv. LAS vs Soft-LAS vs Deep LAS vs MLP-only vs MMSE | Fig. 9 |
| `run_fig10_modelBasedComparison.m` | + DetNet + optimal SD, ZF and MMSE init | Fig. 10 |
| `run_fig11_sweeps.m` | BER vs FFT length (a) and antenna count (b) | Fig. 11 |
| `run_fig12_dataDrivenComparison.m` | Deep LAS vs LSTM/Bi-LSTM/GRU/MLP-only, 4-QAM & 16-QAM | Fig. 12 |

Not included (left as extensions): full turbo-coded BER for
**Figs. 9-12** absolute numbers (see caveats below) -- Fig. 10's
DetNet and SD baselines ARE now included, see the DetNet caveat below
for how confident to be in that specific piece.

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

6. **DetNet (Fig. 10) is the most experimental file in this package.**
   Unlike everything else here, it can't use MATLAB's standard
   `trainNetwork`/Layer-array API (DetNet has per-layer trainable step
   sizes and a nonstandard skip-connection structure), so
   `trainDetNet.m` implements a manual `dlarray` + custom-Adam training
   loop instead. I could not verify that `dlgradient` correctly
   differentiates through the nested cell-array-of-structs parameter
   container used here (`params.W1{k}`, etc.) — if `trainDetNet.m`
   errors on `dlgradient(loss, params)`, the most likely fix is
   flattening `params` into named top-level fields (`params.W1_1`,
   `params.W1_2`, ... instead of `params.W1{1}`, `params.W1{2}`, ...)
   so there's an unambiguous flat container to differentiate through.
   Everything downstream of a successfully-trained DetNet
   (`detNetPredict.m`, its use in `simulateBER.m`) is plain numeric
   code and should be reliable once training succeeds. If it's not
   worth debugging, set `includeDetNet = false` at the top of
   `run_fig10_modelBasedComparison.m` and get the rest of the figure
   without it.

7. **The "SD (optimal)" baseline is exact brute force, not a real
   sphere-decoding tree search.** For Nt=4, `M^Nt` is at most
   `16^4 = 65536`, small enough to evaluate every candidate directly
   (`bruteForceSD.m`) and get the exact same answer a correctly-tuned
   sphere decoder would give — computing the actual ML/MAP minimum
   IS the definition of what SD approximates efficiently. This will
   not scale past a handful of antennas at 16-QAM; a real bounded-
   radius tree search would be needed for larger configurations.

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
- `simulateBER('sd-optimal', ...)` should be at or near the *best*
  curve in any comparison (it's the exact ML/MAP solution) — if
  anything sits clearly below it, something in that other detector's
  BER accounting is wrong (e.g. a bit-layout mismatch), since nothing
  can legitimately beat the optimal detector on the same channel model.
- DetNet's curve has no guaranteed ordering relative to the others —
  unlike the analytic detectors, its quality depends entirely on
  whether `trainDetNet.m`'s custom training loop actually converged.
  If it's flat/near-random across all SNR, that's a training-failure
  signal (see caveat #6), not necessarily a fundamental limitation.
- `simulateBER.m` now prints the error/bit/block counts for every SNR
  point — if you see very few errors (say, under ~30) at your highest
  SNR points despite adaptive stopping, `maxBlocksFactor` was hit
  before `minErrors`; raise it (at the cost of runtime) for a more
  reliable low-BER tail.
