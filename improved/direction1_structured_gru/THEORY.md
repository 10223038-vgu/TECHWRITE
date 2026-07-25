# Theory Backbone: Deep-LAS Extensions

This document derives the mathematical justification for our two proposed
extensions to the Deep LAS detector. Part 1 (structured GRU sequence input)
is fully derived below. Part 2 (deep-unfolded LAS) is scoped but not yet
derived -- see the placeholder at the end.

---

## Part 1: Structured GRU Sequence Input

### 1.1 The gap in the original system model

The paper's per-subcarrier system model (its Eq. 1) is

  y = Hx + n

with H drawn i.i.d. Rayleigh, and the subcarrier index k omitted "for
notational brevity." Read literally, this means each subcarrier's channel
realization is independent of every other subcarrier's. The paper describes
the GRU block as consuming "2Nt+1 sequences each of 1xN length" without
specifying what forms a timestep, and separately claims longer FFT length
improves training -- but under an i.i.d.-per-subcarrier channel model there
is no mechanism connecting FFT length to anything the GRU could exploit.

### 1.2 Why an i.i.d. input makes the recurrence pointless

A GRU's hidden state update is, schematically,

  h_k = GRU(h_{k-1}, u_k)

where u_k is the input at timestep k. The only reason to prefer a GRU over a
memoryless per-timestep mapping (i.e. the MLP block alone) is that h_{k-1}
carries information useful for producing the output at time k. If the inputs
u_k across timesteps are i.i.d. (as they are if timesteps are arbitrary
pooled samples, or subcarriers under an i.i.d. channel model), there is no
statistical dependency for h_{k-1} to carry, and the loss-minimizing solution
degenerates to a mapping that ignores h_{k-1} -- i.e. the GRU collapses to
the same function class as the MLP. Any empirical gap between GRU and
MLP-only performance under these conditions cannot be attributed to genuine
sequential modeling.

**Claim:** for the GRU to be doing something the MLP structurally cannot,
consecutive timesteps must be built from a channel process that is
correlated across the sequence axis.

### 1.3 A channel model with real correlation: frequency-selective (multipath) fading

Let the time-domain channel between Tx antenna j and Rx antenna i have L
resolvable taps, h_ij[l] for l = 0, ..., L-1, each complex Gaussian with
power following a power delay profile (PDP). A standard exponential PDP:

  sigma_l^2 = exp(-l/tau) / sum_{m=0}^{L-1} exp(-m/tau)

normalized so that sum_l sigma_l^2 = 1 (unit total channel energy), with tau
a decay-rate parameter (larger tau = more spread-out delay profile = more
frequency-selective).

The frequency-domain channel on OFDM subcarrier k (k = 0, ..., N-1, N = FFT
length) is the DFT of the taps:

  H_k[i,j] = sum_{l=0}^{L-1} h_ij[l] * exp(-i*2*pi*k*l/N)

### 1.4 Subcarrier correlation (the key derived quantity)

Because every H_k is a linear combination of the *same* L time-domain taps,
subcarriers are correlated. The correlation between subcarriers separated by
Delta_k is:

  R(Delta_k) = E[H_k * conj(H_{k+Delta_k})]
             = sum_{l=0}^{L-1} sigma_l^2 * exp(-i*2*pi*Delta_k*l/N)

This is exactly the DFT of the power delay profile. It equals 1 at
Delta_k = 0 and decays as Delta_k grows; the decay rate is governed
entirely by L and the PDP shape (tau).

**Coherence bandwidth** (in units of subcarriers), the range over which the
channel remains meaningfully correlated:

  B_c ~= N / L

(more taps L -> more frequency-selective -> faster decorrelation across
subcarriers -> smaller B_c).

### 1.5 Reframing LLR estimation as sequential filtering

Treat the channel state s_k = vec(H_k) as a hidden process correlated across
k per R(Delta_k) above, with noisy observations y_k = H_k(s_k) x_k + n_k. The
MLP block estimates LLR_k from y_k alone -- a memoryless estimator. Since s_k
is correlated with neighboring s_{k-1}, s_{k+1}, ..., the *statistically
optimal* estimator should incorporate neighboring subcarriers' observations
too. This is a nonlinear filtering/smoothing problem (Kalman-flavored, but
nonlinear due to the x_k multiplication), and a GRU trained end-to-end over a
**correctly ordered** sequence is a learned approximation to that filter:

  h_k = GRU(h_{k-1}, y_k, LLR_MLP,k)
  LLR_k = g(h_k)

This construction is only meaningful if the timestep axis follows the true
correlation axis (consecutive subcarriers within a coherence block) rather
than an arbitrary pooling order.

### 1.6 Design rule: choosing sequence length

This replaces the original ad hoc choice (`seqLen ~= FFT_len/256`, an
unjustified proxy) with a principled one: set the GRU's sequence length to
match the coherence bandwidth,

  L_seq ~= B_c = N / L

so each training/inference sequence spans approximately one coherence block:
long enough to contain real correlation, short enough to avoid extending
into decorrelated territory (which would just look like noise to the
recurrence). This also gives a first-principles explanation for the paper's
own empirical claim that longer FFT length improves GRU training -- larger N
(for fixed L) increases B_c, allowing longer *useful* sequences.

### 1.7 The critical ablation

The experiment that distinguishes "the GRU is exploiting real correlation"
from "the GRU just has more parameters/context and would do this with any
sequence": train three variants at the *same* sequence length L_seq, on the
*same underlying samples*:

1. **Correctly ordered**: consecutive subcarriers within one coherence block
2. **Shuffled control**: the identical set of samples, randomly reordered
   within each sequence
3. **Original i.i.d. pooling**: current baseline (arbitrary pooling, no
   correlation structure at all)

- If (1) beats (2) despite identical content, that is direct evidence the
  GRU is using the *order* (i.e. the correlation structure), not just
  benefiting from a longer receptive field or more parameters.
- If (1) is approximately equal to (2), the correlation at the chosen L_seq
  is too weak to matter (or the theory needs revisiting) -- report this
  honestly rather than force a positive result.

### 1.8 Testable hypotheses for the results section

- **H1**: BER improves as L_seq approaches B_c, and degrades for
  L_seq << B_c (too short to capture correlation) or L_seq >> B_c (sequence
  extends into decorrelated subcarriers, diluting useful signal with noise)
- **H2**: correlation-ordered sequences outperform shuffled sequences of
  identical length and content (isolates the mechanism, per 1.7)
- **H3**: the improvement from (H1)/(H2) should shrink as L (number of
  channel taps) increases, since more taps -> smaller B_c -> less
  correlation available to exploit for any fixed FFT length -- this gives a
  second, independent way to confirm the mechanism is really coherence
  bandwidth and not some other confound

---

## Part 2: Deep-Unfolded LAS

**Status: not yet derived.**

Scope for next session: formalize LAS's Algorithm 1 (hard detection) and
Algorithm 2 (counter-hypothesis search) as a fixed number of unfolded
layers, replace the closed-form step size (the paper's Eq. 9,
`lambda = 2*round(z/(2q))`) with a learnable per-layer step size or
projection function, and define an end-to-end loss against true transmitted
bits (rather than the current label-distillation setup, which trains a
black-box MLP+GRU to imitate classical LAS's output). Needs:

- A precise unfolded-layer definition analogous to DetNet's structure but
  specific to LAS's coordinate-descent update rule
- A loss function across layers (e.g. weighted sum over unfolding depth,
  similar to DetNet's `log(k+1)` weighting)
- A justification for why training against true bits end-to-end should be
  able to *exceed* the classical LAS teacher's performance, not just match it
- An experiment plan showing performance vs. number of unfolded layers K,
  and complexity/parameter-count comparison against both classical LAS and
  the original paper's Deep LAS

To be filled in once we work through it together.
