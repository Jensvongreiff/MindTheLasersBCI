# Online EEG Adaptation Experiments

This branch contains implementations and experiments for online adaptation methods for EEG motor imagery classification using EEGNet on the BCI Competition IV 2a dataset.

The focus of this work is evaluating how online adaptation methods handle EEG domain shift under:

* Cross-session evaluation
* Cross-subject evaluation

using 4-class motor imagery classification.

Implemented methods:

* Static EEGNet baseline
* AdaBN (Adaptive Batch Normalization)
* Rolling-buffer AdaBN
* TENT (Test-time Entropy Minimization)

---

# Repository Structure

```text
week6/
├── eval.py
├── data_augmentation.py
├── models.py
├── online_adaptation.py
├── online_adaptation_eval.py
├── online_adaptation_visualization.py
└── results/
```

---

# Methods

## 1. Baseline EEGNet

Standard EEGNet trained offline and evaluated without adaptation.

Implemented in:

* `week6/eval.py`

The script supports:

* cross-session evaluation
* cross-subject evaluation
* LOSO evaluation
* model weight saving

---

## 2. AdaBN

Adaptive Batch Normalization updates BatchNorm running statistics online using incoming target EEG data.

No labels or gradients are used.

Updated BatchNorm statistics:

* `running_mean`
* `running_var`

The method adapts the normalization statistics to the target EEG distribution online.

---

## 3. Rolling-Buffer AdaBN

Instead of adapting using a batch of samples, adaptation is performed using a rolling buffer of recent EEG trials.

Example:

```text
[t1, t2, ..., t16]
↓
new trial arrives
↓
[t2, t3, ..., t17]
```


---

## 4. TENT

TENT (Test-time Entropy Minimization) performs online adaptation using entropy minimization on unlabeled target EEG data.

Unlike AdaBN, TENT updates:

* BatchNorm affine parameters (γ and β)

using gradient descent.

Entropy objective:

```text
H(p) = -Σ p log p
```

The goal is to encourage confident predictions on the target domain.

---

# Scripts

## 1. Baseline Evaluation

```bash
python week6/eval.py \
    --model eegnet \
    --protocol cross-session \
    --classes 4 \
    --save_weights \
    --out ./week6/results/eegnet/cross_session_baseline
```

---

## 2. Online Adaptation Evaluation

```bash
python week6/online_adaptation_eval.py \
    --method adabn \
    --protocol cross-session \
    --classes 4 \
    --buffer_size 16 \
    --bn_momentum 0.1 \
    --weights_dir ./week6/results/eegnet/cross_session_baseline/weights \
    --out ./week6/results/eegnet/cross_session_adabn
```

---

# online_adaptation_eval.py Usage

## Required Arguments

### `--method`

Online adaptation method.

Choices:

* `adabn`
* `tent`

Example:

```bash
--method adabn
```

---

### `--protocol`

Evaluation protocol.

Choices:

* `cross-session`
* `cross-subject`

Example:

```bash
--protocol cross-session
```

---

### `--weights_dir`

Directory containing pretrained EEGNet weights.

Example:

```bash
--weights_dir ./week6/results/eegnet/cross_session_baseline/weights
```

---

### `--out`

Output directory for metrics and plots.

Example:

```bash
--out ./week6/results/eegnet/cross_session_adabn
```

---

# Optional Arguments

### `--classes`

Number of classes.

Choices:

* `2`
* `4`

Default:

```bash
--classes 4
```

---

### `--seeds`

Number of random seeds.

Default:

```bash
--seeds 3
```

---

### `--buffer_size`

Rolling buffer size for online adaptation.

Controls the number of recent EEG trials used for adaptation.

Example:

```bash
--buffer_size 16
```

---

### `--bn_momentum`

BatchNorm momentum for AdaBN.

Controls adaptation speed.

Typical values:

* `0.01`
* `0.1`
* `0.5`

Example:

```bash
--bn_momentum 0.1
```

---

### `--tent_lr`

Learning rate for TENT online optimization.

Typical values:

* `1e-4`
* `1e-3`

Example:

```bash
--tent_lr 1e-4
```

---

# Example Commands

## AdaBN

```bash
python week6/online_adaptation_eval.py \
    --method adabn \
    --protocol cross-session \
    --classes 4 \
    --buffer_size 16 \
    --bn_momentum 0.1 \
    --weights_dir ./week6/results/eegnet/cross_session_baseline/weights \
    --out ./week6/results/eegnet/cross_session_adabn
```

---

## Rolling-Buffer AdaBN

```bash
python week6/online_adaptation_eval.py \
    --method adabn \
    --protocol cross-session \
    --classes 4 \
    --buffer_size 32 \
    --bn_momentum 0.1 \
    --weights_dir ./week6/results/eegnet/cross_session_baseline/weights \
    --out ./week6/results/eegnet/cross_session_adabn_buffer_32
```

---

## TENT

```bash
python week6/online_adaptation_eval.py \
    --method tent \
    --protocol cross-session \
    --classes 4 \
    --buffer_size 16 \
    --tent_lr 1e-4 \
    --weights_dir ./week6/results/eegnet/cross_session_baseline/weights \
    --out ./week6/results/eegnet/cross_session_tent
```

---

# Visualization

Plots are generated using:

```bash
python week6/online_adaptation_visualization.py
```

Generated plots include:

* Main method comparison
* Subject-wise improvement
* Calibration comparison
* Momentum ablation
* Buffer-size ablation

Figures are saved to:

```text
week6/results/figures/
```

---

# Ablation Studies

## Momentum Ablation

Tested:

* 0.01
* 0.1
* 0.5

to study:

* stability vs adaptability

---

## Buffer-Size Ablation

Tested:

* 4
* 8
* 16
* 32

to study:

* short-term vs long-term online adaptation memory

--> Not much difference found 
