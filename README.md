# BCI Practical SS26 - Team 4

Welcome to the repository for Team 4's Brain-Computer Interface (BCI) Practical Course project at TUM. This repository documents our progress in building a complete BCI pipeline, using the `baseline-bci-26` repository as our foundational blueprint and tutorial.

## 👥 Team Members
* **Daniel Cortez de Oliveira Marche Barros**
* **Jens von Greiff**
* **Martin Waxenberger**

---

## ⚙️ Environment Setup & Installation

**⚠️ Important Notice Regarding OS Compatibility:** We are currently in the process of migrating our development environment from Ubuntu to a fully Windows-based setup. As a result, the current build may not be completely stable on Windows. Specifically, building `PyQt5` using `uv` currently fails on Windows machines. The pyproject.toml has been updated to work on windows, with certain scripts requiring the pyproject_linux.toml environment to function.

*If reviewing this specific commit on Windows, please be aware of this limitation. A fully functional Windows implementation will be provided starting with Submission 3.*

### Local Installation (Ubuntu/Linux Preferred for Current Build)
To set up the local repository path and install dependencies using `uv`:

```bash
# Clone the repository
git clone <repository-url>
cd practical-ss26-team4

# Sync dependencies using uv
uv sync

```

## Project Roadmap & Submissions

This project is structured around iterative submissions, moving from theoretical foundations to online BCI deployment.
## Submission 1: State of the Art Paper Review

For our initial theoretical foundation, our team analyzed and presented three recent papers focusing on advanced deep learning architectures and error potential detection for BCI applications:

#### TCACNet: 
Explored Temporal and Channel Attention Convolutional Networks for Motor Imagery (MI) classification, noting its lightweight model advantages (<10 ms inference).

#### ErrP Detection:
 Reviewed feature-based detection of Error-Related Potentials for Human-Robot Interaction.

####   IFNet: 
Analyzed the Interactive Frequency Convolutional Neural Network for robust EEG decoding.

## Submission 2: Signal Preprocessing & Characterization

This phase focused on cleaning and analyzing raw EEG data to prepare it for feature extraction and classification.
Files are suffixed according to their **Week 3 Exercise** for ease of grading.

### Filtering Implementation: 
Evaluated multiple filter candidates, ultimately selecting a 4th-order IIR Butterworth filter over higher-tap FIR filters for optimal magnitude response and latency tradeoffs.

### Artifact Removal:
 Applied Independent Component Analysis (ICA) to identify and remove ocular and muscular artifacts from the raw signal. The pyproject_linux.toml environment must be used to run this task.

### Signal Characterization:
Analyzed Event-Related Desynchronization/Synchronization (ERD/ERS) in Mu (8-13 Hz) and Beta (13-30 Hz) bands over motor channels (C3, Cz, C4) during left/right-hand motor imagery tasks. Generated ERD/ERS time courses, power spectral density plots, ERP averages, and time-frequency representations. Running the script requires entering the subject number, session number, run number, and task in the same format as the data .xdf file.

## Submission 3:

tbd

## Local Repository Structure

```
practical-ss26-team4/  (project root)
├── data/                   # Raw EEG data (.xdf files)
│   └── sub-P{NNN}/
│       └── eeg/
│           └── sub-P{NNN}_ses-S{NNN}_task-{TASK}_run-{RRR}_eeg.xdf
├── plots/                  # Generated visual assets for reports
│   ├── artifact_removal/   # ICA topomaps and signal comparisons
│   └── filtering/          # Frequency response and latency plots
├── plots_MI/               # MI-specific plots post analysis
├── tests/                  # Unit tests for pipeline compenents
├── artifact_removal_A2.py  # ICA and artifact rejection logic
├── filtering_A1.py         # Implementation of IIR/FIR filters
├── loading_helpers.py      # XDF loading utilities
├── mi_evaluate_plots.py    # Plotting functions
├── signal_analysis_A3.py   # Main analysis script
├── pyproject.toml          # Project configuration and dependencies
├── uv.lock                 # Locked dependencies
└── README.md               # This file
```
 