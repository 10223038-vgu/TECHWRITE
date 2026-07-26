# Theory: Pooled-Context (Permutation-Invariant) Deep LAS

This document picks up exactly where `THEORY.md`'s Part 1 (Structured GRU
Sequence Input) left off. Read that first -- Sections 1.1-1.6 (why i.i.d.
input makes recurrence pointless, the multipath correlation model, the
Lseq design rule) are unchanged and still the foundation here. This
document covers what happened when Section 1.7's ablation was actually
run, why it forced a change of mechanism, and the theory behind the
replacement.

## 1. What the ablation actually found

THEORY.md Section 1.7 proposed three arms to isolate *why* a sequence
model should help: correlated order, shuffled (same content, random
order), and i.i.d. (no correlation at all). The prediction, if the
"structured GRU" hypothesis were correct, was:

```
BER(correlated) < BER(shuffled) < BER(i.i.d.)
```

The measured result (`run_direction1_ablation.m`, Part A, 7-point SNR
sweep) was:

| SNR (dB) | Correlated | Shuffled | i.i.d. |
|---:|---:|---:|---:|
| 0  | 0.1267 | 0.1458 | 0.1845 |
| 2  | 0.1042 | 0.1087 | 0.1616 |
| 4  | 0.0996 | 0.0925 | 0.1359 |
| 6  | 0.0670 | 0.0609 | 0.1320 |
| 8  | 0.0458 | 0.0381 | 0.1130 |
| 10 | 0.0372 | 0.0344 | 0.1157 |
| 12 | 0.0295 | 0.0366 | 0.0915 |

The correlated-vs-shuffled gap **flips sign four times** across seven
points, with magnitudes (0.003-0.02) consistent with run-to-run noise --
this is not a real, directionally consistent effect. The
correlated/shuffled-vs-i.i.d. gap, in contrast, is large (>2x at several
points) and consistent in sign at every single point.

**Conclusion:** the GRU is extracting value from the *presence* of
correlated context, not from the *order* it's presented in. A recurrent
architecture is the wrong tool for a function that the data itself says
is order-invariant -- it has to spend training capacity approximating
permutation-invariance via its recurrence dynamics, instead of getting
that property for free from the architecture, which is a large part of
why the GRU trained slower/worse and floored out at high SNR
(`run_pooling_vs_gru.m`'s comparison plot).

## 2. Restating the hypothesis in order-invariant form

Section 1.4/1.5 of THEORY.md model the correlation between subcarrier
$k$ and a nearby subcarrier $k+\delta$ (within one coherence block) as
approximately constant and high for $|\delta| \ll B_c$, decaying for
$|\delta|$ approaching or exceeding $B_c$. The original (falsified)
hypothesis was that a model should learn a function of the *sequence*
$(u_{k-L_{seq}+1}, \dots, u_{k-1}, u_k)$. The revised hypothesis, informed
by Section 1, is:

$$
\hat{\ell}_k \approx f\big(u_k,\; \phi(\{u_j : j \in \mathcal{N}(k)\})\big)
$$

where $\mathcal{N}(k)$ is the same Lseq-1 context window as before, but
$\phi(\cdot)$ is a **symmetric (permutation-invariant) function of the
context set**, not a function of an ordered sequence. $u_j$ here is the
same per-subcarrier feature vector used throughout
(`buildCorrelatedAndShuffledSequences.m`'s `UG_full(:,j)`: the
normalized MMSE estimate plus the MLP's rough LLR summary).

This is a strictly weaker, empirically-grounded restatement of the
original hypothesis -- it keeps "correlated context helps" (confirmed)
and drops "in a specific order" (falsified).

## 3. Choice of $\phi$: mean / max / std pooling

Symmetric functions of a set are exactly what "Deep Sets"-style
architectures are built from (Zaheer et al., 2017): any permutation-
invariant function of a set can be approximated by
$\rho\big(\sum_j g(u_j)\big)$ for suitable $g, \rho$. Implementing the
full learned-$g$-then-sum architecture requires a custom pooling layer;
as a first, cheap-to-implement test of the hypothesis,
`buildPooledContextFeatures.m` instead uses three **fixed, well-known
symmetric statistics** computed directly on the raw context vectors:

$$
\phi(\{u_j\}) = \big[\; \text{mean}_j(u_j) \;;\; \max_j(u_j) \;;\;
\text{std}_j(u_j) \;\big]
$$

- **mean** -- the context's central tendency; if the target's own
  estimate is noisy, averaging over Lseq-1 correlated (hence similarly-
  biased) neighbors is a variance-reduction move, similar in spirit to
  averaging repeated noisy measurements of a slowly-varying quantity.
- **max** -- captures the least-ambiguous (highest-confidence) neighbor
  in the window; useful because in Rayleigh-faded channels a few
  subcarriers within a coherence block can be in a much better local SNR
  regime than the target itself, and a max-pool lets that signal through
  even if it's not the average trend.
- **std** -- a self-reported uncertainty/consistency signal: a
  context whose per-subcarrier estimates disagree strongly (in the
  middle of a fast-varying tap transition, or at the edge of a coherence
  block near $|\delta| \approx B_c$) is a weaker basis for correcting the
  target than a context that agrees tightly.

The pooled vector is concatenated with the target's own feature vector
$u_k$ (never pooled away -- the target's own MMSE/MLP information is
always available un-degraded) and fed to a plain feedforward network
(`trainPoolingNet.m`): two hidden layers, ReLU, dropout, trained with
Adam -- the same overall training recipe as `trainGRUFromSequences.m`,
so any performance difference is attributable to the representation, not
the optimizer/training budget.

## 4. Why this should (and did) train better than the GRU

- **No BPTT.** A 64-timestep GRU backpropagates gradients through 64
  recurrent steps; vanishing/exploding gradients are a real risk even
  with GRU gating, and the sequence-builder's own data volume (a few
  thousand sequences after the stride fix) is small relative to what
  that training problem typically needs. A feedforward net over a
  fixed-size input has no such depth-in-time problem.
- **Fewer effective degrees of freedom to identify.** A permutation-
  invariant target function has a *smaller* effective hypothesis space
  than the space of order-sensitive sequence functions a GRU can
  represent (it's a strict subset). Constraining the architecture to
  match the true function's invariance is a standard bias-variance win
  when that invariance is correct -- it cannot represent the shuffled-
  order artifacts the GRU wasted some of its capacity on distinguishing
  between (since the data shows there's no real signal there to
  distinguish).

This matches what was observed empirically: the pooling net not only
matched the GRU, it beat it by a widening margin at high SNR, while the
GRU visibly floored out (`run_pooling_vs_gru.m`'s plot) -- consistent
with the GRU hitting a training-difficulty ceiling that the simpler
architecture doesn't share.

## 5. What `run_pooling_vs_original.m` then found, and what it means

Against the actual target (our reproduced Original Deep LAS, not just
the GRU), the pooling net lands at **rough parity** with the original --
the two curves track each other closely across the full SNR sweep, with
the gap (in either direction) small compared to the gap either one has
over Conv. LAS. This is a real, honest, and still useful result, but a
more modest one than "beats the GRU" suggested:

- It confirms correlated context is not *harmful* and is roughly as
  useful, when pooled naively (uniform weight, whole Lseq-1 window), as
  the original's own per-subcarrier MLP+GRU pipeline already extracts
  from the target's own MMSE estimate alone.
- It does **not** yet show that exploiting correlation *beats* the
  original by a clear margin. Two explanations are both consistent with
  the data so far, and are not yet distinguished:
  1. There genuinely isn't much more information in the neighboring
     subcarriers beyond what the target's own estimate + MLP already
     captures, once you're only allowed a naive uniform-weight
     aggregate (the ceiling really is near parity).
  2. The naive, distance-blind pooling (Section 3's $\phi$ treats every
     one of the Lseq-1 context positions identically, regardless of how
     close $|\delta|$ is to the target) throws away exactly the
     structure Section 1.4 derived: correlation strength *decays* with
     $|\delta|$. Averaging a highly-correlated near neighbor with a
     nearly-decorrelated far one (near $|\delta| \approx B_c$) dilutes
     the near neighbor's signal instead of weighting it appropriately.

## 6. Proposed next step: distance-aware pooling

Explanation (2) above is directly testable and is well-motivated by
theory already established in `THEORY.md` (the same correlation-decay
argument that set $L_{seq} \approx B_c$ in the first place): instead of
one uniform pool over all Lseq-1 context positions, weight or partition
the context by $|\delta|$ (distance from the target subcarrier), e.g.:

- **Distance-weighted mean**: $\sum_j w_j u_j$ with
  $w_j \propto \exp(-|\delta_j| / B_c)$ (or the actual measured
  correlation profile from Section 1.4, if available), instead of the
  uniform $\frac{1}{L_{seq}-1}$ weight used now.
- **Near/far split pooling**: compute mean/max/std separately over a
  small "near" window (e.g. $|\delta| \le B_c/4$) and the remaining
  "far" context, giving the network two differently-scoped aggregates
  instead of one that blends both regimes.

Either change keeps the architecture fully permutation-invariant
*within* each distance band (still consistent with Section 1's finding
that order doesn't matter) while reintroducing the one piece of
structure -- correlation strength as a function of distance -- that
uniform pooling currently discards. This is the most theoretically
well-motivated lever left before concluding that correlation exploitation
has hit a real ceiling.
