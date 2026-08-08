# Direction 2: Deep-Unfolded LAS

**Status: Option B validated but inconclusive; Option A (soft relaxation)
now implemented, not yet trained to completion.** Option B (learnable
step size, hard decision unchanged) was confirmed correctly implemented
and confirmed to train (parameters moved substantially from init), but
did not clearly beat classical LAS across repeated attempts -- see
`THEORY_UNFOLDING.md` Section 3 for the full diagnostic trail. Option A
replaces the hard decision itself with a differentiable softmax
relaxation. **Run `run_unfoldedB_soft_sanity_check.m` before trusting
any Option A training result** -- same discipline as Option B's sanity
check, and just as important given how much new code this is.

## Files

| File | Purpose |
|---|---|
| `THEORY_UNFOLDING.md` | **Read this first** -- Option B's full diagnostic trail, why Option A was built next |
| **Option B (hard decision + learned step size)** | |
| `unfoldedLASCore.m` | Shared differentiable core: hard candidate selection (detached), learnable per-layer step size |
| `unfoldedLAS1.m`, `unfoldedModLASCounter.m` | Algorithm 1 / Algorithm 2 wrappers |
| `unfoldedSoftLAS_A.m`, `unfoldedSoftLAS_B.m` | Unfolded-A / Unfolded-B detectors |
| `trainUnfoldedLAS.m`, `simulateBERUnfoldedLAS.m` | Training loop / BER evaluator |
| `run_unfoldedB_sanity_check.m` | Correctness check (alpha=1 should match classical) -- **passed** |
| `run_direction2_vs_original.m` | Option B training driver -- result: inconclusive, see THEORY_UNFOLDING.md |
| **Option A (differentiable soft relaxation)** | |
| `unfoldedLASCoreSoft.m` | Shared differentiable core: softmax over candidates, learnable per-layer temperature |
| `unfoldedLAS1Soft.m`, `unfoldedModLASCounterSoft.m` | Algorithm 1 / Algorithm 2 wrappers |
| `unfoldedSoftLAS_A_Soft.m`, `unfoldedSoftLAS_B_Soft.m` | Unfolded-A / Unfolded-B detectors |
| `trainUnfoldedLASSoft.m`, `simulateBERUnfoldedLASSoft.m` | Training loop / BER evaluator |
| `run_unfoldedB_soft_sanity_check.m` | **Run this first** -- correctness check (low temperature should match classical) |
| `run_direction2_soft_vs_hard.m` | Option A training driver -- the actual new result |
| **Shared** | |
| `buildUnfoldedTrainingData.m` | i.i.d. flat-channel training samples with true bits (used by both Option A and B) |
| `simulateBERClassicalSoftLAS.m` | Classical (fixed, no learning) baseline -- the correct comparison for both options |
| `checkClassicalSweepDepth.m` | Measures classical convergence depth to set K1/K2 |

## Setup
Same as Direction 1: edit `origDir` near the top of each `run_*.m` script.

## Run order (if starting fresh)
```matlab
checkClassicalSweepDepth              % measure K1/K2 (already done: max ~4/~5 with correct init)
run_unfoldedB_sanity_check            % Option B correctness check -- passed
run_direction2_vs_original            % Option B result -- inconclusive (see THEORY_UNFOLDING.md)
run_unfoldedB_soft_sanity_check       % Option A correctness check -- RUN THIS NEXT
run_direction2_soft_vs_hard           % Option A result -- the current open question
```

## What to look for in the Option A result

Same discipline as Option B: compare `Unfolded-B-Soft` against
**"Classical softOutputLAS"** specifically. Watch the printed `T1`/`T2`
temperature values each epoch -- shrinking toward 0 means training is
sharpening toward classical-like hard decisions; growing means it's
deliberately blending candidates more than classical ever would. If the
result is close to classical, don't trust a single run -- use
`../direction1_structured_gru/run_stability_analysis.m` as a template
for the multi-repeat treatment before reporting a winner.

