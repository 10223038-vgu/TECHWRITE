# Fixes and Improvements (Figures 5–12)

This document summarizes current issues identified in the simulation results and proposed fixes for improving performance, visualization, and consistency across figures.

---

## 1. Figures 5–8: Rx Symbol Imbalance

### Issue

The received (Rx) symbols are not uniformly distributed. A large portion of samples is concentrated around specific constellation points (e.g., `+1` and `-1`), leading to biased learning and poor generalization.

### Cause

* Dataset imbalance in symbol generation
* Channel/noise effects not sufficiently randomized
* Possible bug in symbol mapping or sampling process

### Fix

* Ensure **uniform symbol generation** across the constellation:

  * For QPSK: equal probability of all 4 symbols
  * For 16-QAM: uniform sampling of all 16 constellation points
* Increase dataset size to improve statistical diversity
* Add **randomization in channel conditions** (e.g., SNR variation, fading)
* Verify symbol mapping logic (modulation/demodulation correctness)

---

## 2. Figure 9: Random Guessing Behavior

### Issue

The first two algorithms perform close to random guessing (accuracy near theoretical random baseline).

### Possible Causes

* Model not trained properly (underfitting)
* Learning rate too high/low
* Insufficient training epochs
* Poor feature representation (input not informative enough)
* Incorrect label alignment

### Fix

* Verify **training pipeline**:

  * Check input-output matching
  * Ensure correct labels
* Tune hyperparameters:

  * Learning rate
  * Batch size
  * Number of epochs
* Normalize input data
* Add **baseline sanity check**:

  * Compare with simple detector (ZF/MMSE)
* Initialize model weights properly
* Monitor **loss curve** (should decrease over time)

---

## 3. Figure 11: Discontinuity in Results

### Issue

The plot is not smooth/continuous, making it difficult to interpret system performance trends.

### Cause

* Discrete or sparse SNR sampling
* Missing interpolation between points
* Simulation inconsistency across runs

### Fix

* Use **finer SNR steps** (e.g., 1 dB instead of large gaps)
* Increase number of simulation samples per SNR point
* Apply **averaging over multiple runs**
* Optionally use interpolation for smoother curves

---

## 4. Figure 12: Performance Comparison (NEW)

### Description

Add a new figure comparing the performance of the proposed **Deep LAS** model with other data-driven techniques.

### Requirements

* Two subplots:

  * (a) 4-QAM
  * (b) 16-QAM
* Metrics:

  * BER vs SNR (recommended)
* Include:

  * Deep LAS (proposed)
  * Other learning-based methods
  * Optional: classical baselines (ZF, MMSE)

### Goal

* Demonstrate performance gain of Deep LAS
* Highlight robustness across modulation schemes

---

## Summary of Actions

* [ ] Fix symbol imbalance in dataset (Figs. 5–8)
* [ ] Debug and retrain weak models (Fig. 9)
* [ ] Improve SNR resolution and averaging (Fig. 11)
* [ ] Add new comparison figure (Fig. 12)

---

## Notes

* Always validate results against known baselines (e.g., ZF/MMSE).
* Ensure reproducibility by fixing random seeds when needed.
* Log all parameters used for each simulation.

---
