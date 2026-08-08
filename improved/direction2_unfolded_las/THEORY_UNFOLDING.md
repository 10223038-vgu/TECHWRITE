# Theory: Deep-Unfolded LAS (Direction 2)

## 1. What "Algorithm 1/2" actually are, in this codebase

- **Algorithm 1** (`las1Hard.m`): a coordinate-descent local search over
  the full PAM grid. At each real dimension, it computes the cost change
  `dCost = qnn*lambda^2 - 2*lambda*zn` for every candidate move `lambda`,
  and takes the single best one (a **hard `argmin`**) if it improves the
  cost. Repeated in sweeps until no dimension improves. Produces a
  detected vector `xr_hat` and a final cost `F0` (Eq. 7).
- **Algorithm 2** (`modLASCounter.m`): the same coordinate-descent core,
  but for one target (dimension, bit) pair, the search is restricted so
  that dimension can only move to PAM levels that flip that bit. Run
  once per (dimension, bit), producing a cost `F1`.
- **Combination** (`softOutputLAS.m`): `LLR = (1/sigma^2)*(F1 - F0)`,
  sign depending on the bit Algorithm 1 actually detected.

Both algorithms share one core routine (`lasSearchCore.m`); the only
difference is the candidate-set restriction.

## 2. The central obstacle to "unfolding" this

Classic deep-unfolding (e.g. LISTA unfolding ISTA, or unfolded AMP)
works because the underlying iterative algorithm is already a smooth,
differentiable update (a soft-thresholding function, a linear
projection, etc.) -- unfolding just means turning a fixed number of
iterations into network layers and attaching a learnable scalar (a
step size, a threshold) to each one.

LAS's core operation is a **hard `argmin` over a discrete, data-dependent
candidate set**. There is no meaningful gradient of "which candidate got
picked" with respect to the inputs that determined the pick -- the
choice is piecewise-constant and jumps discontinuously as costs cross
each other. Naively trying to backpropagate through this operation
either gives a zero gradient almost everywhere (useless for training) or
requires a genuinely different relaxation.

## 3. Two ways to handle this (why we're doing B first, A later)

**Option A -- soft relaxation.** Replace the hard `argmin` with a
temperature-weighted softmax over candidate costs, taking an
expectation-weighted move instead of the single best one. Fully
differentiable end-to-end, including through the selection itself. This
is the more standard, more powerful unfolding technique for discrete
search problems -- but it requires a custom differentiable operation at
every layer, careful temperature scheduling (too sharp = vanishing
gradient, too soft = doesn't resemble real LAS), and is harder to
validate against the classical algorithm as a sanity check.

**Option B -- keep the hard decision, learn only the step size (chosen
for the first build).** The `argmin` itself is computed on **detached**
(non-differentiable) values -- exactly the classical algorithm's
decision, unchanged. What's learnable is a per-layer scalar `alpha_k`
that scales how much of the chosen move is actually applied:
```
step = alpha_k * bestLambda        % bestLambda: detached, classical choice
xr   = xr + step * e_n             % differentiable w.r.t. alpha_k
z    = z  - step * Q(:,n)
```
Gradients flow correctly through `alpha_k` and through the chain of
`xr`/`z` across layers, but NOT through the discrete choice of which
candidate was picked. This is simpler to implement, much easier to
validate (`alpha_k = 1` for every layer reproduces the classical
algorithm exactly, which is also the training initialization -- so
training starts AT the known-good classical behavior and only has to
learn whether deviating from it helps), and is a legitimate, established
technique (closely related to how step sizes/damping factors are learned
in many published unfolding papers while keeping a fixed nonlinearity).

The tradeoff: because there's no gradient through the selection itself,
the achievable improvement is bounded by what a per-layer step-size
adjustment alone can capture -- it cannot learn a fundamentally
different search strategy, just a different pace/aggressiveness for the
existing one. If Option B shows a real, statistically-validated
improvement, Option A is a natural, higher-ceiling follow-up. If Option
B shows no improvement at all, that's useful evidence before investing
in Option A's much larger engineering cost.

## 4. Two variants, since both algorithms were requested

**Unfolded-A** (`unfoldedSoftLAS_A.m`): unfolds ONLY Algorithm 1
(`unfoldedLAS1.m`, `K1` layers), then a small **learned linear readout**
maps the final state `[xr_final; z_final]` directly to per-bit LLRs.
This bypasses Algorithm 2 and the classical combination formula
entirely -- a genuinely different design that lets training decide how
to turn the Step-1 state into LLRs, rather than presupposing the
classical two-step structure is the right one. Cheaper (no per-bit
counter search) and a useful point of comparison for whether Algorithm
2's structure is actually earning its computational cost.

**Unfolded-B** (`unfoldedSoftLAS_B.m`): unfolds BOTH algorithms
(`K1` layers for Algorithm 1, `K2` layers for each of Algorithm 2's
per-(dimension,bit) counter searches, sharing one set of `K2` learnable
scalars across all of them to keep the parameter count small), and
combines them via the exact classical `(F1-F0)/sigma^2` formula. Every
step size in both searches is learned, but the algorithmic structure
matches the published method exactly.

## 5. Training objective: end-to-end against true bits, not distilled from classical LAS

The original paper's GRU was trained by **regression** against
`softOutputLAS.m`'s own LLR output (distillation from the classical
algorithm). Direction 2 instead trains by **binary cross-entropy against
the true transmitted bits** (`trainUnfoldedLAS.m`), using the same sign
convention as the rest of the codebase (`hardBits = LLR < 0`, so the
logit for "P(bit=1)" is `-LLR`). This is a meaningfully different
training signal: the network is never told what the classical algorithm
would have predicted, only whether its own final decision was actually
right.

## 6. What to compare against, and why

The correct baseline for isolating "did learning help" is **classical,
fixed `softOutputLAS.m`** (`simulateBERClassicalSoftLAS.m`), not
Direction 1's MLP-GRU -- those answer different questions. Since both
unfolded variants are initialized at `alpha=1` (reproducing classical
behavior exactly), any measured difference from the classical curve is
attributable to what training changed, not to a different starting
point.

**Do not trust a single run for a close call.** Direction 1's
experience (`run_stability_analysis.m`) showed BER differences at
adjacent SNR points can flip sign between independent runs due to
training/data randomness alone. If Unfolded-A/B and classical LAS end up
close together on a first run, that comparison needs the same
multi-repeat mean+/-std treatment before reporting a winner -- not
another single re-run.
