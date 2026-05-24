# BCI Practical SS26 - Team 4

Welcome to the repository for Team 4's Brain-Computer Interface (BCI) Practical Course project at TUM. This repository documents our progress in building a complete BCI pipeline, using the `baseline-bci-26` repository as our foundational blueprint and tutorial.

## 👥 Team Members
* **Daniel Cortez de Oliveira Marche Barros**
* **Jens von Greiff**
* **Martin Waxenberger**

---

## ⚙️ Environment Setup & Installation

**⚠️ Important Notice Regarding OS Compatibility:** 

This repository is primarily Windows-compatible. For different operating systems, full functionality was not tested.

### Local Installation 
To set up the local repository path and install dependencies using `uv`:

```bash
# Clone the repository
git clone <repository-url>
cd practical-ss26-team4

# Sync dependencies using uv
uv sync

```

### Running files for assignments
This is the most reliable way to run files in the current project format:
```bash
# Run the respective files in this form
uv run python -m weekX.filename

# Example
uv run python -m week4.feature_extraction

```

### Project Roadmap & Submissions

This project is structured around iterative submissions, moving from theoretical foundations to online BCI deployment.
#### Submission 1: State of the Art Paper Review

For our initial theoretical foundation, our team analyzed and presented three recent papers focusing on advanced deep learning architectures and error potential detection for BCI applications:

    TCACNet: Explored Temporal and Channel Attention Convolutional Networks for Motor Imagery (MI) classification, noting its lightweight model advantages (<10 ms inference).

    ErrP Detection: Reviewed feature-based detection of Error-Related Potentials for Human-Robot Interaction.

    IFNet: Analyzed the Interactive Frequency Convolutional Neural Network for robust EEG decoding.

#### Submission 2: Signal Preprocessing & Characterization (Week 3)

This phase focused on cleaning and analyzing raw EEG data to prepare it for feature extraction and classification.

    Filtering Implementation: 
    Evaluated multiple filter candidates, ultimately selecting a 4th-order IIR Butterworth filter over higher-tap FIR filters for optimal magnitude response and latency tradeoffs.

    Artifact Removal: 
    Applied Independent Component Analysis (ICA) to identify and remove ocular and muscular artifacts from the raw signal.

    Signal Characterization: 
    Analyzed Event-Related Desynchronization/Synchronization (ERD/ERS) in Mu (8-13 Hz) and Beta (13-30 Hz) bands over motor channels (C3, Cz, C4) during left/right-hand motor imagery tasks. Generated ERD/ERS time courses, power spectral density plots, ERP averages, and time-frequency representations.

#### Submission 3: Feature Extraction & Classification (Week 4)

In this phase, we implemented and evaluated four distinct feature extraction methods to classify Left vs. Right hand Motor Imagery:

    Band-power PSD: 
    Log-variance in subject-specific Mu and Beta bands.

    CSP (Common Spatial Pattern): 
    4 filters applied to the Mu+Beta bandpass.

    Morlet Wavelets: 
    Log-amplitude at 8/10/12/20/24 Hz.

    Riemannian MDM (Custom Method): 
    Classification directly on spatial covariance matrices using Riemannian geometry.

Key Insights:

    While CSP achieved the highest within-session accuracy (72.1%), it suffered from a severe transfer gap (+19.4% drop) when tested cross-session due to over-optimizing to specific spatial patterns.

    Riemannian MDM emerged as the most robust architecture for real-world deployment, achieving the highest cross-session accuracy (64.0%) with minimal transfer gap (+1.0%), successfully addressing the non-stationarity limitations of standard CSP.

#### Submission 4: Evaluation Framework & Data Augmentation (Week 5)

Moving beyond standard accuracy metrics, this phase introduces a strict 4-Pillar evaluation framework to test model viability for online BCI deployment, alongside data augmentation to improve robustness.
Evaluation Framework (eval.py)

We built a comprehensive CLI tool evaluating models across four key pillars:

    Accuracy: 
    Macro-F1, Balanced Accuracy, and Confusion Matrices.

    Reliability: 
    Expected Calibration Error (ECE), Brier Score, and Calibration Curves.

    Efficiency: 
    Computational latency and Information Transfer Rate (ITR).

    Generalization: 
    Performance degradation across Noisy, Cross-Session, and Cross-Subject settings.

Data Augmentation

(Placeholder for teammate's work: Include a brief summary here of the input-space and/or feature-space augmentation techniques chosen (e.g., Mixup, Channel Dropout) and their impact on reducing the domain gap and improving ECE.)
## Local Repository Structure

```
practical-ss26-team4/       # Project root
├── data/                   # Raw EEG data (.xdf files) and BCI2a datasets (.mat)
├── features/               # Extracted feature binaries (e.g., all_features.pkl)
├── plots/                  # Generated visual assets for reports (filtering, ICA)
├── plots_MI/               # MI-specific plots post analysis
├── week3/                  # Signal Preprocessing & Characterization
│   ├── artifact_removal_A2.py
│   ├── filtering_A1.py
│   ├── loading_helpers.py
│   ├── mi_evaluate_plots.py
│   ├── signal_analysis_A3.py
│   └── tests/
├── week4/                  # Feature Extraction & Classification
│   ├── classification.py
│   ├── feature_extraction.py
│   ├── loading.py
│   ├── plotting.py
│   └── preprocessing.py
├── week5/                  # Evaluation Framework & Augmentation
│   └── eval.py             # 4-Pillar Evaluation script
├── pyproject.toml          # Project configuration and dependencies (Windows)
├── pyproject_linux.toml    # Project configuration and dependencies (Linux)
├── uv.lock                 # Locked dependencies
└── README.md               # This file
```
 