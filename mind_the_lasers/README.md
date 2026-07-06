# Mind The Lasers

**Technical University of Munich (TUM) - BCI Practical Course SS26**
**Team 4:** Daniel Cortez de Oliveira Marche Barros, Jens von Greiff, Martin Waxenberger

## 1. Project Overview

*Mind The Lasers* is a continuous three-lane runner game controlled entirely via Motor Imagery (MI) Brain-Computer Interfaces (BCI). The objective is to navigate a character through a procedurally generated track, collecting targets and avoiding laser obstacles. As the game progresses, the difficulty and speed increase, requiring precise timing and utilization of the boost mechanic.

The project focuses on building a robust, real-time data processing pipeline that can handle continuous EEG streams, process them asynchronously, and map them to game commands while accounting for the inherent noise and misclassification rates of non-invasive EEG.

## 2. Command Mapping & Game Strategy

The game utilizes a strictly three-class paradigm mapped to discrete movements:

* **Left Motor Imagery:** Steer Left
* **Right Motor Imagery:** Steer Right
* **Rest (Idle):** Break / Charge Boost

### Coping with Imperfect BCI

To ensure the game remains engaging despite classifier inaccuracies, we implemented a fail-forward mechanical design:

* **Strategic Misclassification:** Because lasers act as binary obstacles in specific lanes, an incorrect lateral classification might inadvertently move the player out of danger.
* **The Rest/Boost Mechanic:** Deliberately resting (maintaining the "Idle" class) charges a movement boost. If a player is resting outside of laser range, they are rewarded with a speed boost for their next movement. If they are resting inside laser range, the accumulated boost allows them to rapidly escape the situation upon their next directional command.

## 3. System Architecture & Pipeline

The software architecture strictly separates the synchronous Pygame rendering loop from the heavy asynchronous numerical computations required for EEG processing. This is achieved via Python's `multiprocessing` library.

### 3.1 Data Ingestion (`signal.py`)

* **Live Mode:** Utilizes Lab Streaming Layer (LSL) to connect to the EEG amplifier. It maintains a rolling buffer, extracting a predefined window (e.g., 1 second of data at 250Hz) and sliding forward with a specific stride (e.g., 100ms) to ensure continuous, high-frequency control updates.
* **Offline Simulation Mode:** Parses the BCI Competition IV 2a dataset, synthesizes the "Rest" class from the pre-cue fixation periods, splits the data 50/50 for calibration/testing, and streams the matrices through the pipeline mimicking exact hardware latency.

### 3.2 Signal Processing (`filtering.py` & `artifact_removal.py`)

* **IIR Bandpass:** A causal 8-30Hz Butterworth filter (`scipy.signal.lfilter`) is applied to isolate the Mu and Beta bands necessary for MI detection without looking into the "future" of the signal window.
* **Spatial Filtering (ICA):** An MNE-based Independent Component Analysis module fits spatial unmixing matrices during the calibration phase and applies them via fast matrix multiplication during the live game loop to reject artifact components (e.g., blinks).

### 3.3 Classification Baselines (`model.py` & `feature_extraction.py`)

The orchestrator supports dynamic swapping of the classification backbone:

1. **End-to-End Deep Learning:** A native PyTorch implementation of `EEGNet`.
2. **Traditional Machine Learning:** A Common Spatial Pattern (CSP) feature extractor coupled with Linear Discriminant Analysis (LDA) via `mne` and `scikit-learn`.

### 3.4 Command Smoothing (`smoothing.py`)

Raw probabilities generated every 100ms are too volatile for continuous runner mechanics. The `SmoothingController` maintains a state buffer of size $N$. It applies a confidence threshold (default 60%); if the maximum probability falls below this, the output defaults to "Rest." If it passes, the class is appended to the buffer, and a majority vote dictates the final game command.

## 4. Repository Structure

```text
mind_the_lasers/
│
├── src/
│   ├── main.py                     # Entry point and CLI argument parser
│   │
│   ├── game/                       # Pygame Mechanics
│   │   ├── game.py                 # Core rendering and state loop
│   │   ├── input_controller.py     # Keyboard vs BCI state arbitration & offline metric tracking
│   │   ├── player.py               # Player kinematics
│   │   ├── laser.py                # Obstacle generation logic
│   │   ├── level.py                # Track progression mapping
│   │   └── settings.py             # Global constants
│   │
│   └── pipeline/                   # BCI Processing Core
│       ├── signal.py               # LSL, Offline Streamers, and EEGWindow dataclass
│       ├── filtering.py            # Causal IIR Bandpass implementations
│       ├── artifact_removal.py     # MNE ICA abstraction
│       ├── feature_extraction.py   # CSP wrapper
│       ├── model.py                # EEGNet architecture and LDA wrapper
│       ├── smoothing.py            # Stabilizer (Thresholding + Majority Vote)
│       ├── pipeline_constructor.py # Multiprocessing worker and pipeline orchestration
│       └── weights/                # Serialized .pt and .pkl subject-specific weights

```

## 5. Execution Logic

The system is controlled via a command-line interface in `main.py`.

**1. Live Hardware Session:**
Executes the game using the LSL stream. Requires pre-calibrated weights in the `pipeline/weights/` directory.

```bash
python mind_the_lasers/src/main.py --mode live --baseline eegnet

```

**2. Automated Offline Validation:**
Performs an end-to-end simulation. It parses the provided `.mat` dataset, splits it 50/50, dynamically fits the chosen baseline to the training half, and streams the test half into the Pygame window. Upon completion, it outputs a detailed `classification_report` and `confusion_matrix`.

```bash
python mind_the_lasers/src/main.py --mode offline --dataset /path/to/A01T.mat --baseline csp-lda

```