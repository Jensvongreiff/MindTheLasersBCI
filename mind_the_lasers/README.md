# Mind The Lasers

**Technical University of Munich (TUM) - BCI Practical Course SS26**
**Team 4:** Daniel Cortez de Oliveira Marche Barros, Jens von Greiff, Martin Waxenberger

## 1. Project Overview

Mind The Lasers is a timing-based BCI game designed to evaluate motor imagery EEG decoders in an interactive setting. The player controls a colored ball that moves horizontally across the screen while avoiding vertical laser beams. The decoder outputs one of three commands: left, right, or rest. Movement commands start the ball moving in the corresponding direction, while rest stops the ball.

The game includes two modes. Training Mode presents structured left, right, and rest trials with clear ground-truth commands and logs decoder performance. Play Mode uses the same command interface in a dynamic game environment with levels, lives, lasers, boost mechanics, and gameplay metrics.

The project combines a real-time EEG processing pipeline with a Pygame-based game. EEG data can come from live LSL streams or replayed XDF recordings. The pipeline processes EEG windows, produces decoded predictions, optionally smooths them, and sends commands to the game over UDP. This separation allows the game, classifier, stream replay, keyboard simulation, and evaluation tools to be tested independently.

## 2. Command Mapping & Game Strategy

The game utilizes a strictly three-class paradigm mapped to discrete movements:

* **Left Motor Imagery:** Move Left
* **Right Motor Imagery:** Move Right
* **Rest (Idle):** Stop And Charge Boost

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
│   ├── main.py                          # Launches streamer, pipeline and/or game
│   │
│   ├── game/
│   │   ├── run_game.py                  # Game entry point
│   │   ├── game.py                      # Main game loop
│   │   ├── input_controller.py          # UDP controller and command definitions
│   │   ├── player.py                    # Player movement, boost and lives
│   │   ├── laser.py                     # Sweeping laser obstacle implementation
│   │   ├── level.py                     # Level class and progression logic
│   │   ├── levels.py                    # Collection of game levels
│   │   ├── game_metrics.py              # Gameplay metrics and game-session logging
│   │   ├── training_mode.py             # Training mode implementation
│   │   ├── training_logger.py           # Training trial logging
│   │   ├── training_trial.py            # Training trial implementation
│   │   ├── mode_select.py               # Training/play mode selection screen
│   │   └── settings.py                  # Global game constants
│   │
│   ├── pipeline/
│   │   ├── run_pipeline.py              # Pipeline entry point
│   │   ├── pipeline_config.py           # Pipeline configuration and model loading
│   │   ├── pipeline_constructor.py      # BCIPipeline and worker process
│   │   ├── prediction_sender.py         # Sends decoded predictions over UDP
│   │   ├── signal.py                    # LSL streamers, offline loader and EEGWindow
│   │   ├── filtering.py                 # EEG filtering implementations
│   │   ├── artifact_removal.py          # Artifact removal implementations
│   │   ├── feature_extraction.py        # CSP and feature extraction implementations
│   │   ├── model.py                     # EEGNet and LDA model wrappers
│   │   ├── smoothing.py                 # Prediction rejection and smoothing logic
│   │   ├── decoder_metrics.py           # Decoder evaluation metrics
│   │   └── weights/                     # Serialized fitted pipeline weights
│   │       ├── csp.pkl
│   │       ├── lda.pkl
│   │       ├── ica.pkl
│   │       └── eegnet.pt
│   │
│   ├── stream/
│   │   ├── streamer.py                  # Replays XDF recordings as LSL streams
│   │   ├── marker_sender.py             # Publishes game/training markers over LSL
│   │   └── key_press.py                 # Keyboard prediction sender over UDP
│   │
│   ├── logs/
│   │   ├── game_logs/                   # CSV logs from play-mode sessions
│   │   └── training_logs/               # CSV logs from training-mode sessions
│   │
│   ├── visualization/
│   │   ├── visualize_decoder_metrics.py # Visualizes decoder metrics from CSV logs
│   │   ├── visualize_play.py            # Visualizes gameplay metrics from CSV logs
│   │   └── visualize_training.py        # Visualizes training metrics from CSV logs
│   │
│   └── tests/
│       └── ...                      # Pipeline evaluation and automated tests
│
└── README.md
```


## 5. Execution Logic

The project can be launched either through a single entry point (`src/main.py`) or by starting each component independently for debugging and development.

### Main Entry Point

The entire system is launched through:

```bash
python -m mind_the_lasers.src.main
```

The following command-line arguments are available:

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Data source: `live` (LSL stream) or `offline` (XDF replay) | `live` |
| `--controller` | Input source: `pipeline`, `keyboard`, or `both` | `pipeline` |
| `--baseline` | Classifier backend: `csp-lda` or `eegnet` | `csp-lda` |
| `--xdf` | Path to an XDF recording (required only in offline mode) | — |
| `--ip` | UDP destination IP | `127.0.0.1` |
| `--port` | UDP destination port | `5005` |

### Example Usage

**Live BCI session**

Uses an existing LSL EEG stream.

```bash
python -m mind_the_lasers.src.main \
    --mode live \
    --controller pipeline \
    --baseline csp-lda
```

**Offline replay from an XDF recording**

Replays a previously recorded session while running the classifier and the game.

```bash
python -m mind_the_lasers.src.main \
    --mode offline \
    --xdf /path/to/recording.xdf
```

**Keyboard-only control**

Useful for testing gameplay without running the pipeline.

```bash
python -m mind_the_lasers.src.main \
    --controller keyboard
```

**Keyboard and pipeline simultaneously**

Useful for debugging while allowing manual override.

```bash
python -m mind_the_lasers.src.main \
    --controller both
```

---

### Running Individual Processes

For debugging, each component can also be started independently.

#### 1. Replay an XDF recording as an LSL stream

```bash
python -m mind_the_lasers.src.stream.streamer /path/to/recording.xdf
```

#### 2. Run the BCI pipeline

```bash
python -m mind_the_lasers.src.pipeline.run_pipeline \
    --mode live \
    --baseline zp-csp-lda
```

#### 3. Launch the game

```bash
python -m mind_the_lasers.src.game.run_game
```

#### 4. Send manual keyboard predictions

```bash
python -m mind_the_lasers.src.stream.key_press
```

This modular execution allows each subsystem (streaming, classification, communication, and gameplay) to be tested independently before performing a full end-to-end experiment.

#### 5. Visualize decoding metrics

```bash
python -m mind_the_lasers.src.visualization.visualize_decoder_metrics --json mind_the_lasers/reports/decoder_summary_eegnet_0001.json mind_the_lasers/reports/decoder_summary_zp-csp-lda_0001.json
```

