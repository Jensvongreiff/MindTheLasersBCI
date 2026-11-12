# Mind The Lasers

**Technical University of Munich (TUM) - BCI Practical Course SS26**

Authors: Jens von Greiff, Daniel Cortez de Oliveira Marche Barros, Martin Waxenberger

## 1. Project Overview

Mind The Lasers is a timing-based BCI game designed to evaluate motor imagery EEG decoders in an interactive setting. The player controls a colored ball that moves horizontally across the screen while avoiding vertical laser beams. The decoder outputs one of three commands: left, right, or rest. Movement commands start the ball moving in the corresponding direction, while rest stops the ball.

The game includes three modes. Training Mode presents structured left, right, and rest trials with clear ground-truth commands and logs decoder performance. Play Mode uses the same command interface in a dynamic game environment with levels, lives, lasers, boost mechanics, and gameplay metrics. ErrP Experiment Mode introduces controlled correct/error movements for investigating EEG error-related potentials (ErrPs); see Section 7 for instructions and configuration.

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
│   │   ├── errp_mode.py                 # ErrP experiment logic and inter-trial screen
│   │   ├── errp_trial.py                # ErrP trial definition and target direction
│   │   ├── errp_logger.py               # ErrP CSV decision logging
│   │   ├── mode_select.py               # Training/play/ErrP mode-selection screens
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
│   │   ├── marker_sender.py             # Publishes trial markers over LSL or UDP
│   │   └── key_press.py                 # Keyboard prediction sender over UDP
│   │
│   ├── logs/
│   │   ├── game_logs/                   # CSV logs from play-mode sessions
│   │   ├── training_logs/               # CSV logs from training-mode sessions
│   │   └── errp_logs/                   # CSV logs from ErrP-mode sessions
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


## 5. Marker Transmission

Marker transport is selected in `src/game/settings.py`. **The current configuration uses UDP** (not LSL); set `MARKER_TRANSPORT = "lsl"` to use an LSL marker stream instead:

```python
MARKER_TRANSPORT = "udp"  # "udp" (current) or "lsl"
MARKER_UDP_IP = "127.0.0.1"
MARKER_UDP_PORT = 1000
```

LSL mode continues to send the original string markers on the `MindTheLasersMarkers` stream. UDP mode sends each code as an ASCII decimal string to the configured IP and port. The UDP codes are centralized in `MARKER_UDP_CODES` in `src/stream/marker_sender.py`:

| Marker | UDP code | Marker | UDP code |
|---|---:|---|---:|
| `trial_start:left` | 1 | `trial_start:right` | 2 |
| `trial_start:rest` | 3 | `trial_start:correct:left` | 10 |
| `trial_start:correct:right` | 11 | `trial_start:error:left` | 12 |
| `trial_start:error:right` | 13 | `trial_input:correct:left` | 14 |
| `trial_input:correct:right` | 15 | `trial_input:error:left` | 16 |
| `trial_input:error:right` | 17 | `trial_end:correct:left` | 18 |
| `trial_end:correct:right` | 19 | `trial_end:error:left` | 20 |
| `trial_end:error:right` | 21 | | |

In Unicorn Suite / BCI Core Recorder, enable external UDP trigger reception and configure the trigger UDP port to match `MARKER_UDP_PORT` (`1000` by default). Use `127.0.0.1` when Core Recorder runs on the same computer and select ASCII text trigger decoding. Keep the port separate from the game's classifier prediction port, which remains `5005`.

## 6. Execution Logic

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

#### 5. Visualize game metrics

```bash
python -m mind_the_lasers.src.visualization.visualize_training
```

```bash
python -m mind_the_lasers.src.visualization.visualize_play
```

## 7. ErrP Experimental Paradigm

ErrP mode is a separate, **Pygame-based error-induction task** intended for recording EEG responses to correct and unexpected movements. A ball starts in the middle of the screen and a target appears on the left or right. On scheduled **correct** trials, the incoming movement command is executed as received; on scheduled **error** trials, the game swaps LEFT and RIGHT to produce a movement mismatch. These labels describe the *scheduled manipulation*, not necessarily the movement or error the participant actually experiences. EEG acquisition and ErrP classification are **not** performed by this game mode: record EEG separately and synchronize using its event markers.

### 7.1 Launch and controls

Run the paradigm from the directory where the `mind_the_lasers` Python package is importable, using the project's `uv` environment:

```bash
uv run python -m mind_the_lasers.src.game.run_game --controller local_keyboard
```

This starts the game with **local keyboard input**, so no external decoder is needed to operate the ErrP task. At the main menu, press **3 — ERRP EXPERIMENT MODE**, then select an ErrP variant using **1**, **2**, or **3**. The first trial starts automatically. After each trial except the last, an inter-trial screen requires **Space** to begin the next trial (after the minimum interval has elapsed).

| Key | Function |
|---|---|
| `1` / `2` / `3` (main menu) | Select Training / Play / ErrP mode |
| `1` / `2` / `3` (ErrP menu) | Select Observation / Interactive / Interactive Continuous |
| `Left` / `Right` arrows | Directional input in interactive variants; hold to move in continuous mode |
| `Down` arrow | REST in other game modes; **not accepted as a trial decision** in ErrP mode |
| `Space` | Continue from the inter-trial screen |
| `F11` | Toggle fullscreen; the 1000 × 600 logical game canvas scales proportionally to the display |
| `Esc` | Back to the menu (or exit from the main menu); ends the active ErrP session |
| `Enter` | Return to the main menu from the completed-session screen |

The display window can also be resized. For actual EEG sessions, choose the screen size/fullscreen state before recording and keep it consistent within a session.

### 7.2 ErrP variants

| Variant | How it works |
|---|---|
| **Observation** (`observation`) | The player observes an automatically generated movement; the game waits `ERRP_COMMAND_DELAY` before issuing the target-directed command (or its reversed version). No participant directional input is required. |
| **Interactive** (`interactive`) | One new LEFT/RIGHT press determines the command for the trial. On an error trial, that command is reversed before moving the ball. Further input is not used once the decision is made. |
| **Interactive Continuous** (`interactive_continuous`) | The participant steers the ball with ongoing LEFT/RIGHT input. On error trials, each active direction is reversed. The trial is finalized when the ball reaches the target or crosses the wrong-direction distance threshold. Releasing the arrows stops movement. |

Only LEFT and RIGHT count as directional decisions. In continuous mode, no direction/REST stops the ball rather than completing a trial. Trial feedback and movement timing are governed by the settings below; explicit correctness/debug text is shown only when `DEBUG_ERRP = True`.

### 7.3 Trial and session flow

1. `ErrPMode` creates any initial **calibration trials** (all scheduled correct), then a randomized block of `2 * ERRP_TRIALS_PER_COMMAND` trials. Each randomized trial independently samples its left/right target using `LR_PROBABILITY` and its scheduled error condition using `ERROR_PROBABILITY`.
2. `load_trial()` centers the ball, resets the controller and timers, starts the trial clock, and sends `trial_start:<condition>:<target>`.
3. The selected ErrP variant supplies the command. On scheduled error trials, `_apply_trial_error()` swaps LEFT and RIGHT; `trial_input:<condition>:<target>` is emitted when the decision is registered in Observation/Interactive mode or on the first directional movement update in Interactive Continuous mode (only once per trial).
4. The game applies movement and its completion timing, records the finalized decision in the CSV, and sends `trial_end:<condition>:<target>` when the trial finishes.
5. Between trials, the game shows **TRIAL COMPLETE** and requires **Space**, after `ERRP_INTER_TRIAL_MIN_TIME` seconds, to load the next trial. The EEG stream itself is **not paused or reset**. After the last trial, the session-complete screen appears.

**Important:** `trial_type = "error"` signifies a planned command reversal, not proof that the ball traveled away from the target or that the participant perceived an error. For example, a participant pressing the opposite key on a scheduled error trial may produce a correct movement. Check the CSV's `input_correct` and `command_correct` fields when selecting EEG epochs. The markers encode the scheduled condition and target side, **not** actual behavioral correctness.

### 7.4 Important settings (`src/game/settings.py`)

The current values in the uploaded code are:

| Setting | Current value | Purpose |
|---|---|---|
| `CALIBRATION_TRIALS` | `0` | Number of initial, error-free familiarization trials; increase to enable them |
| `ERRP_TRIALS_PER_COMMAND` | `25` | Historical name: randomized trials = **2 × this value**, currently 50 total; does *not* guarantee exactly 25 trials per direction |
| `ERROR_PROBABILITY` | `0.2` | Probability that each randomized trial has its LEFT/RIGHT command flipped |
| `LR_PROBABILITY` | `0.5` | Probability the target appears on the left (otherwise right) |
| `ERRP_COMMAND_DELAY` | `3.0` seconds | Preparation delay before automatic movement in Observation mode |
| `ERRP_INCORRECT_FEEDBACK_TIME` | `1.3` seconds | Incorrect-movement interval in Observation and Interactive modes before the final feedback interval |
| `ERRP_INCORRECT_FEEDBACK_TIME_CONTINUOUS` | `0.0` seconds | Equivalent additional movement interval in Interactive Continuous |
| `ERRP_CORRECT_TRIAL_DELAY` | `0.3` seconds | Delay at the end of a correct movement |
| `ERRP_INTER_TRIAL_MIN_TIME` | `1.0` seconds | Minimum inter-trial wait before Space can start the next trial |
| `INCORRECT_DISTANCE_THRESHOLD_FRACTION` | `3` | Wrong-direction completion threshold for Interactive Continuous, as a fraction of half-screen width |
| `DEBUG_ERRP` | `False` | Displays trial condition/target and correctness feedback for debugging; leave off for participant-facing sessions |
| `MARKER_TRANSPORT` | `"udp"` | Select `"udp"` for recorder trigger codes or `"lsl"` for string markers |
| `MARKER_UDP_IP`, `MARKER_UDP_PORT` | `"127.0.0.1"`, `1000` | UDP trigger destination, used only for `MARKER_TRANSPORT = "udp"` |

In addition, `ErrPMode.TRIAL_FEEDBACK_TIME` is currently `0.5` seconds (defined in `src/game/errp_mode.py`, not `settings.py`). The probabilities are sampled independently, so the exact counts of error and left/right trials vary between sessions. Change the settings **before** starting a session and record the chosen values with your experimental protocol.

### 7.5 Code organization

The principal files are:

| File | Responsibility |
|---|---|
| `src/game/run_game.py` | Main event/render loop, `--controller` argument, menu navigation, controller selection, display scaling, and creation/closing of ErrP sessions |
| `src/game/mode_select.py` | Main mode and ErrP sub-mode selection screens |
| `src/game/errp_mode.py` | Trial randomization, state transitions, input handling, command inversion, movement, feedback timing, inter-trial screen, marker emission, and CSV logging calls |
| `src/game/errp_trial.py` | One trial's target side, scheduled condition, initial/goal position; builds on `TrainingTrial` |
| `src/game/errp_logger.py` | Opens one CSV per session and saves one row per finalized trial decision; all column definitions are commented next to `FIELDNAMES` |
| `src/game/input_controller.py` | Local keyboard and UDP prediction controllers, including the LEFT/RIGHT/REST command definitions |
| `src/game/settings.py` | Experiment parameters and marker transport configuration |
| `src/stream/marker_sender.py` | LSL string marker publisher or UDP numeric trigger sender; contains the `MARKER_UDP_CODES` mapping |

### 7.6 Event markers and behavioral CSV

**Marker output is separate from controller input.** With `--controller local_keyboard`, arrow keys control the paradigm directly; `MARKER_TRANSPORT` independently determines where trial markers are sent:

- `"lsl"`: publishes string events on the **`MindTheLasersMarkers`** LSL stream (`Markers` type).
- `"udp"` (**current setting**): sends ASCII decimal trigger codes to `MARKER_UDP_IP:MARKER_UDP_PORT`, intended for a compatible recorder (e.g., Unicorn BCI Core Recorder). The code mapping and recorder setup are documented in [Section 5](#5-marker-transmission).

The separate controller-prediction UDP port is **5005** (`UDPController`), not the marker port **1000**. The game constructs a UDP prediction controller on startup even when `local_keyboard` is selected, but external predictions are not needed for ErrP control with this launch command. Confirm that the selected marker receiver is running and actually receives `trial_start`, `trial_input`, and `trial_end` before collecting participant data.

Each ErrP session creates `src/logs/errp_logs/errp_session_YYYYMMDD_HHMMSS.csv`. Each row is logged when the decision is finalized, **not** once per EEG sample. Relevant columns include `trial_type` (scheduled condition), `scenario` (target side), `expected_command` (target-directed response), `input_command`/`input_correct` (raw response before error injection, when applicable), `received_command`/`command_correct` (executed response and its correctness), and `decision_time` (time since trial start). In Observation mode, the original input columns are blank. In Interactive Continuous, they represent the input **at the moment the decision is finalized**, not a full keypress history. The local CSV `timestamp` is wall-clock time and should not be treated as an EEG-synchronized event timestamp; use recording markers for EEG alignment.
