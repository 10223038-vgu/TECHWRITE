# Direction 1 Writing Plan: Correlation-Aware and Permutation-Invariant Deep LAS

This is a plan for the Direction 1 section of the paper: what actually
happened, in order, what you can honestly claim, and exactly which
figures/tables back each claim. Organize the section around the
**findings** below (each is a clean, defensible statement), using the
**process** narrative only where it explains *why* you built what you
built -- reviewers care about the findings, not the debugging history.

---

## The five findings, in the order they occurred

### Finding 1 — Reproduction baseline is credible
You reproduced MMSE (hard), Conv. LAS (hard), and the original paper's
Deep LAS (MLP+GRU on flat/i.i.d.-pooled input) as baselines. This isn't
a "finding" you'll state as a contribution, but it's the sanity check
every downstream claim depends on -- your reproduced Deep LAS should
track the paper's own reported trend (better than Conv. LAS, worse than
soft-decision) even if the exact dB gain differs.

**Where this appears in text:** one paragraph in Methods/Setup, not a
separate finding. **Figure:** folded into Finding 4/5's main comparison
figure, not shown alone.

### Finding 2 — Correlated context helps; sequence order does not (the core ablation)
The H2 ablation (correlated-order vs. shuffled-order-same-content vs.
i.i.d. context, GRU architecture held fixed) showed:
- Correlated vs. shuffled BER gap flips sign across the SNR sweep, small
  magnitude (0.003-0.02) -- statistically indistinguishable, i.e. **order
  does not matter**.
- Correlated/shuffled vs. i.i.d. gap is large (>2x at multiple SNRs) and
  consistent in sign at every point -- **correlated context matters a
  lot**.

This is your first clean, positive result, and it's also what
*motivates* replacing the GRU (a sequence-order model) with a
permutation-invariant pooling network in Finding 3.

**Figure to use:** the H2 ablation plot from `run_direction1_ablation.m`
Part A -- BER vs. SNR, three curves (correlated/shuffled/i.i.d.).
**Table to use:** the numeric table (SNR, corr, shuf, iid, gap) -- this
is worth showing as an actual table, not just a plot, because the
"gap flips sign" point is much clearer in numbers than in a crowded plot.

### Finding 3 — A permutation-invariant pooling network matches (and later, at high SNR, beats) the recurrent architecture, while training faster/more stably
Given Finding 2, you replaced the GRU with a feedforward network over
mean/max/std-pooled context (`buildPooledContextFeatures.m` +
`trainPoolingNet.m`). Head-to-head against the GRU
(`run_pooling_vs_gru.m`), the pooling net beat the GRU at every SNR
point, with a widening margin at high SNR where the GRU visibly floored
out (a sign of BPTT-related training difficulty over the 64-step
sequences).

**Figure to use:** the pooling-vs-GRU plot. This is a good "process"
figure to include -- it's the direct evidence for *why* you moved away
from the GRU, which a reviewer familiar with sequence models will ask
about if you only show the pooling net's final numbers.

### Finding 4 — Near/far distance-weighted pooling does not improve on uniform pooling (a negative result worth reporting)
Motivated by the theory that correlation decays with subcarrier
distance, you built a distance-aware pooling variant (near/far split +
exponentially-decaying weighted mean). Tested against uniform pooling,
it did **not** improve results, and was mildly worse. Explanation: `Lseq`
was already set to the coherence bandwidth `Bc` by design, so the whole
window is already "coherent" -- there wasn't much of a within-window
decay gradient left to exploit, and the extra parameters likely just
added overfitting risk on a modest training set.

**Why report a negative result:** it forecloses an obvious reviewer
question ("did you try weighting by distance?") and demonstrates the
uniform-pooling result wasn't reached by accident -- you tested a more
sophisticated alternative and it didn't help, which is itself evidence
uniform pooling was already capturing most of the available signal.

**Figure/table to use:** a compact table (uniform vs. distance-aware,
BER at a few representative SNRs, or the SNR-gain-vs-uniform row from
`run_distance_pooling_vs_original.m`'s output) -- does not need its own
full BER-vs-SNR figure; can be one row in Finding 5's summary table or a
half-column supplementary figure.

### Finding 5 — The real result: a three-regime SNR crossover (THE headline finding)
This is what actually survived proper statistical treatment
(`run_stability_analysis.m`, mean +/- std over 5 independent full
retrains, plus a per-SNR significance check) after single noisy runs
gave contradictory, flip-flopping impressions:

| SNR regime | Result | Statistical support |
|---|---|---|
| **0-6 dB** | Original Deep LAS significantly **outperforms** pooling | \|z\| up to ~13, clearly significant |
| **~8-14 dB** | **Statistical tie** -- no significant difference | \|z\| < 1-2, genuinely ambiguous |
| **16-20 dB** | Pooling nets significantly **outperform** Original | \|z\| ~2.2-2.4, consistent across both pooling variants |

You also tested (and ruled out) the obvious "maybe it's just overfitting"
explanation for the low-SNR loss: a smaller, more heavily regularized
pooling net (`run_lowsnr_regularization_test.m`) did **not** close the
gap -- at most SNR points it was mildly *worse*, not better. That means
the low-SNR underperformance is not a tuning artifact; it's a property
of the correlated-context representation itself (plausible explanation:
at low SNR, neighboring subcarriers' own estimates are themselves too
noisy to be a useful aggregate signal, so pooling over them adds
variance rather than information -- this only flips once neighbor
estimates become reliable enough at higher SNR).

**This finding is more precise and more interesting than your
submitted abstract's claim** ("matches or exceeds... across a wide SNR
range"). The full paper should state the honest, sharper version: *the
correlation-aware pooling detector significantly outperforms the
original reproduction above ~16 dB SNR, is statistically tied with it
in the 8-14 dB range, and underperforms it below ~6 dB -- a genuine,
SNR-dependent crossover rather than a uniform improvement.*

**Figure to use — THIS IS YOUR MAIN RESULTS FIGURE:** the
`run_stability_analysis.m` mean +/- 1 std shaded-band plot (MMSE, Conv.
LAS, Original Deep LAS, Uniform pooling, Distance-aware pooling, all
five curves, bands visible). This is the only figure in the whole
Direction 1 arc that's actually statistically defensible rather than a
single noisy run -- every other BER-vs-SNR plot you've generated should
be considered a diagnostic step, not a result to publish as-is.

**Table to use:** the per-SNR significance table (SNR, Orig mean, Unif
mean, Dist mean, z-scores, verdict) -- condense to the 3 regimes above
for the main text; the full row-by-row table can go in an appendix.

---

## Recommended figure list for the paper (final answer to "what graphs")

In order of appearance, this is the minimal, sufficient figure set --
everything else you generated along the way (single-run BER plots, the
various debugging screenshots) should NOT appear in the paper itself:

1. **Fig 1 (Methods/motivation, optional):** diagram or short description
   of the coherence-bandwidth/correlation model (can be a diagram, not a
   result plot) -- sets up *why* correlated context should matter before
   you show that it does.
2. **Fig 2 — H2 ablation:** correlated vs. shuffled vs. i.i.d. BER curves
   (Finding 2). Table alongside it.
3. **Fig 3 — Pooling vs. GRU:** head-to-head (Finding 3), justifies the
   architecture swap.
4. **Fig 4 (optional, can be a table instead) — Uniform vs.
   distance-aware pooling:** the negative result (Finding 4).
5. **Fig 5 — THE headline figure:** mean +/- std shaded-band plot, all
   five detectors, full SNR sweep (Finding 5). This is the figure your
   abstract, intro, and conclusion should all be written to support.
6. **Table — per-SNR significance summary:** the 3-regime table
   (Finding 5), condensed.

## Suggested section flow (Direction 1 write-up)

1. **Setup:** reproduce baseline, state the correlation hypothesis.
2. **Ablation (Finding 2):** structured-GRU hypothesis, H1/H2/H3 design,
   result -- content matters, order doesn't. This *justifies* the
   architecture choice you're about to present, rather than presenting
   the GRU as if it were your final method.
3. **Method (Finding 3 + 4):** permutation-invariant pooling network,
   brief mention of the GRU comparison (Fig 3) and the distance-aware
   variant you tried and ruled out (Fig 4/table) -- keep this section
   tight, it's supporting evidence, not the headline.
4. **Results (Finding 5):** the three-regime crossover, Fig 5, the
   significance table, and the ruled-out overfitting explanation for the
   low-SNR regime.
5. **Discussion/Limitations:** be upfront that the low-SNR
   underperformance is unresolved (not just untried -- you tested and
   ruled out the obvious fix), and that this is a natural direction for
   Direction 2 (unfolded LAS) to potentially address, since it optimizes
   end-to-end against bits rather than distilling from a fixed
   correlated-context representation.

## One more thing before you write this

Your submitted abstract's claim ("matches or exceeds... across a wide
SNR range") does not match Finding 5 as written. For the full paper you
have two honest options:
- **(a)** Report the sharper, three-regime finding as-is -- it's a
  *better*, more specific result than the abstract's vaguer claim, and
  reviewers tend to reward precision over a blanket "we win" statement.
- **(b)** If you want the full paper's headline to still read as an
  unambiguous win, that would require either extending Direction 2 to
  fix the low-SNR regime, or reframing Direction 1's contribution as
  "identifying exactly where and why correlation-based pooling helps" --
  which is (a) again, just phrased as the intended contribution rather
  than an unresolved gap.

I'd recommend (a). It's not a weaker paper -- "we found X works above 16
dB and precisely characterized why it doesn't below 6 dB, ruling out the
obvious confound" is a more citable, harder-to-dismiss claim than "we
made it better," and it sets up Direction 2 as a natural next step
rather than a totally separate contribution.
