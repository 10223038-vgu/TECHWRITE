# TECHWRITE
# Deep-LAS Simulation Project

This repository contains the implementation and experimentation pipeline for reproducing and extending the results of the **Deep LAS (Likelihood Ascent Search)** detector in MIMO systems.

---

## 📌 Project Overview

The project is divided into two main phases:

1. **Reproduction Phase (Original Work)**

   * Recreate the simulation results from the reference paper
   * Validate correctness of implementation
   * Ensure consistency with reported performance

2. **Extension Phase (Our Work)**

   * Apply fixes and improvements identified during reproduction
   * Introduce new ideas for novelty
   * Develop our own enhanced simulation and results for a new paper

---

## 📂 Repository Structure

```
.
├── original/
│   ├── result/
│   └── deep-las-matlab/
```

### `original/`

This folder contains all resources required to **reproduce the original paper results**.

#### `original/result/`

* Stores output data and figures generated from MATLAB simulations
* Includes BER curves, performance plots, and intermediate results
* Used for validation against the reference paper

#### `original/deep-las-matlab/`

* Contains all MATLAB scripts and functions for simulation
* Implements:

  * MIMO system model
  * Modulation schemes (e.g., QAM)
  * Deep LAS detection algorithm
  * Baseline methods (if included)
* Serves as the **core simulation engine**

---

## ⚙️ Workflow

### Step 1: Reproduce Original Results

* Run MATLAB scripts in `deep-las-matlab/`
* Generate outputs in `result/`
* Compare results with the reference paper

### Step 2: Identify Issues

* Analyze inconsistencies in:

  * BER performance
  * Constellation distribution
  * Model convergence
* Refer to documented fixes (e.g., `fix.md`)

### Step 3: Apply Improvements

* Fix dataset imbalance
* Improve training stability
* Refine simulation parameters (SNR range, averaging, etc.)

### Step 4: Develop Novel Contributions

* Modify Deep LAS architecture or training strategy
* Introduce new detection techniques or hybrid models
* Evaluate performance on:

  * Different modulation schemes
  * More challenging channel conditions

---

## 🎯 Goals

* ✅ Accurately reproduce published results
* 🔧 Debug and improve weak components
* 🚀 Develop a novel contribution for future publication

---

## 📝 Notes

* MATLAB is required to run simulations
* Ensure all dependencies are correctly set before execution
* Results may vary depending on random initialization—use fixed seeds when necessary

---

## 📌 Future Work

* Add extended simulation folder (e.g., `improved/` or `proposed/`)
* Compare Deep LAS with AI-based detectors
* Explore higher-order MIMO systems and realistic channels

