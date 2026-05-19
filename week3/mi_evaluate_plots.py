import sys
from pathlib import Path

import autoreject
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.time_frequency import tfr_multitaper
from scipy.stats import ttest_ind, wilcoxon

sys.path.append(str(Path.cwd().parent))
from week3.loading_helpers import get_epochs, get_raw_offline

mne.set_log_level("WARNING")

# === CONFIG ===
MOVEMENT_LEFT_EVENT = "ARROW LEFT ONSET"
MOVEMENT_RIGHT_EVENT = "ARROW RIGHT ONSET"
REST_EVENT = "CIRCLE ONSET"

T_ONSET = 1.5
T_OFFSET = 4.5
T_CUE = 0.0

MOTOR_CHANNELS = ["Cz", "C3", "C4", "Pz", "Fz"]
USED_CHANNELS = ["Cz", "C3", "C4", "F4", "P3", "Pz", "Fz"]
TARGET_CHANNELS = ["C3", "Cz", "C4", "Pz", "Fz"]

FMIN, FMAX = 1, 40
MU_BAND = (8, 13)
BETA_BAND = (13, 30)
EPOCH_TMIN, EPOCH_TMAX = -1.5, 5.5


def ask_with_default(prompt: str, default: str) -> str:
    value = input(prompt).strip()
    return value if value else default


def get_runtime_inputs() -> tuple[int, int, int, str]:
    subj_nr = int(ask_with_default("Enter subject number: ", "999"))
    session_nr = int(ask_with_default("Enter session number: ", "2"))
    run_nr = int(ask_with_default("Enter run number: ", "1"))
    task = ask_with_default("Enter task (Default, Arrow): ", "arrow")
    return subj_nr, session_nr, run_nr, task


def build_paths(subj_nr: int, session_nr: int, run_nr: int, task: str) -> tuple[Path, Path]:
    cwd = Path.cwd()
    data_file = (
        cwd
        / "data"
        / f"sub-P{subj_nr:03d}"
        / "eeg"
        / f"sub-P{subj_nr:03d}_ses-S{session_nr:03d}_task-{task}_run-{run_nr:03d}_eeg.xdf"
    )
    plots_folder = (
        cwd
        / "plots_MI"
        / f"sub-P{subj_nr:03d}_ses-S{session_nr:03d}_task-{task}_run-{run_nr:03d}"
    )
    plots_folder.mkdir(parents=True, exist_ok=True)
    return data_file, plots_folder


def save_raw_diagnostics(raw: mne.io.BaseRaw, plots_folder: Path) -> None:
    raw.plot(duration=10, start=0, show=False)
    plt.savefig(plots_folder / "raw_data_before.png", dpi=300)
    plt.close()

    raw.filter(FMIN, FMAX)
    raw.notch_filter(np.arange(50, 251, 50), method="spectrum_fit", verbose=False)
    raw.set_eeg_reference(ref_channels="average")

    raw.plot(duration=10, start=0, show=False)
    plt.savefig(plots_folder / "raw_data.png", dpi=300)
    plt.close()


def apply_autoreject(epochs: mne.Epochs) -> mne.Epochs:
    ar = autoreject.AutoReject(n_interpolate=[1, 2], random_state=11, n_jobs=1, verbose=True)
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    cleaned = ar.transform(epochs, reject_log=reject_log)
    print(
        f"Number of epochs originally: {len(epochs.events[:, -1])}, after autoreject: {len(epochs_ar)}"
    )
    return cleaned


def get_band_power(epochs: mne.Epochs, band: tuple[float, float]) -> np.ndarray:
    psd = epochs.compute_psd(method="welch", fmin=band[0], fmax=band[1])
    return psd.get_data().mean(axis=-1).mean(axis=0)


def get_band_power_per_epoch(epochs: mne.Epochs, band: tuple[float, float]) -> np.ndarray:
    psd = epochs.compute_psd(method="welch", fmin=band[0], fmax=band[1])
    return psd.get_data().mean(axis=2)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    pooled_var = ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2)
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled_var)


def get_hilbert_envelope(epochs: mne.Epochs, band: tuple[float, float]) -> np.ndarray:
    filtered = epochs.copy().filter(band[0], band[1], fir_design="firwin")
    envelope = filtered.copy().apply_hilbert(envelope=True)
    return envelope.get_data() ** 2


def get_avg_psd(epochs: mne.Epochs, channels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    psd = epochs.copy().pick(channels).compute_psd(fmin=FMIN, fmax=FMAX)
    psd_data = psd.get_data()
    mean_psd = psd_data.mean(axis=(0, 1))
    sem_psd = psd_data.std(axis=(0, 1)) / np.sqrt(psd_data.shape[0])
    return psd.freqs, mean_psd, sem_psd


def plot_psd_overview(
    epochs_move: mne.Epochs, epochs_rest: mne.Epochs, epochs_right: mne.Epochs, plots_folder: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, (ep, title) in enumerate(
        zip([epochs_move, epochs_rest, epochs_right], ["Left", "Resting", "Right"])
    ):
        psd = ep.compute_psd(fmin=FMIN, fmax=FMAX)
        psd.plot(average=True, picks=MOTOR_CHANNELS, axes=axes[i], show=False)
        axes[i].set_title(f"{title} - PSD")

    plt.suptitle("Power Spectral Density: Left vs Rest vs Right")
    plt.tight_layout()
    plt.savefig(plots_folder / "psd_movement_vs_rest.png", dpi=300)
    plt.close(fig)


def plot_erd(erd: np.ndarray, channels: list[str], title: str, output_file: Path, ylabel: str) -> None:
    fig, _ = plt.subplots(figsize=(10, 5))
    plt.bar(channels, erd)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_effect_sizes(
    left_effects: list[float], right_effects: list[float], channels: list[str], plots_folder: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(channels, left_effects, color="darkorange")
    axes[0].axhline(-0.8, color="green", linestyle="--", label="Large effect")
    axes[0].axhline(-0.5, color="blue", linestyle="--", label="Medium effect")
    axes[0].axhline(-0.2, color="gray", linestyle="--", label="Small effect")
    axes[0].set_title("Mu-band ERD Effect Size (Cohen's d)")
    axes[0].set_ylabel("Cohen's d (Movement vs Rest)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].bar(channels, right_effects, color="darkorange")
    axes[1].axhline(-0.8, color="green", linestyle="--", label="Large effect")
    axes[1].axhline(-0.5, color="blue", linestyle="--", label="Medium effect")
    axes[1].axhline(-0.2, color="gray", linestyle="--", label="Small effect")
    axes[1].set_title("Mu-band ERD Effect Size (Cohen's d) - RIGHT")
    axes[1].set_ylabel("Cohen's d (Movement vs Rest)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(plots_folder / "mu_erd_effect_size.png", dpi=300)
    plt.close(fig)


def plot_tfr(
    epochs_move_long: mne.Epochs,
    epochs_rest_long: mne.Epochs,
    epochs_right_long: mne.Epochs,
    plots_folder: Path,
) -> None:
    freqs = np.linspace(5, 30, 60)
    n_cycles = freqs / 2.0

    tfr_move = tfr_multitaper(
        epochs_move_long,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=3.0,
        return_itc=False,
        average=True,
    )
    tfr_rest = tfr_multitaper(
        epochs_rest_long,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=3.0,
        return_itc=False,
        average=True,
    )
    tfr_right = tfr_multitaper(
        epochs_right_long,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=3.0,
        return_itc=False,
        average=True,
    )

    for ch in MOTOR_CHANNELS:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        tfr_move.plot([tfr_move.ch_names.index(ch)], title=f"TFR {ch}", mode="logratio", axes=axes[0], show=False)
        tfr_rest.plot([tfr_rest.ch_names.index(ch)], title=f"TFR {ch}", mode="logratio", axes=axes[1], show=False)
        tfr_right.plot([tfr_right.ch_names.index(ch)], title=f"TFR {ch}", mode="logratio", axes=axes[2], show=False)

        for ax in axes:
            ax.axvline(T_ONSET, color="green", linestyle="--", label="Onset")
            ax.axvline(T_OFFSET, color="blue", linestyle="--", label="Offset")
            ax.axvline(T_CUE, color="red", linestyle="--", label="Cue")

        fig.suptitle(f"TFR - {ch} - Left vs Rest vs Right")
        plt.legend()
        plt.savefig(plots_folder / f"tfr_movement_vs_rest_{ch}.png", dpi=300)
        plt.close(fig)



def plot_mu_time_series(
    epochs_move_long: mne.Epochs,
    epochs_rest_long: mne.Epochs,
    epochs_right_long: mne.Epochs,
    plots_folder: Path,
) -> None:
    for target_channel in ["C3", "C4", "Cz"]:
        mu_power_move = get_hilbert_envelope(epochs_move_long, MU_BAND)
        mu_power_rest = get_hilbert_envelope(epochs_rest_long, MU_BAND)
        mu_power_right = get_hilbert_envelope(epochs_right_long, MU_BAND)

        ch_idx = epochs_move_long.ch_names.index(target_channel)
        times = epochs_move_long.times

        mean_move = mu_power_move[:, ch_idx, :].mean(axis=0)
        mean_rest = mu_power_rest[:, ch_idx, :].mean(axis=0)
        mean_right = mu_power_right[:, ch_idx, :].mean(axis=0)

        sem_move = mu_power_move[:, ch_idx, :].std(axis=0) / np.sqrt(mu_power_move.shape[0])
        sem_rest = mu_power_rest[:, ch_idx, :].std(axis=0) / np.sqrt(mu_power_rest.shape[0])
        sem_right = mu_power_right[:, ch_idx, :].std(axis=0) / np.sqrt(mu_power_right.shape[0])

        plt.figure(figsize=(10, 5))
        plt.plot(times, mean_move, label="Movement", color="darkred")
        plt.fill_between(times, mean_move - sem_move, mean_move + sem_move, alpha=0.3, color="darkred")

        plt.plot(times, mean_rest, label="Rest", color="navy")
        plt.fill_between(times, mean_rest - sem_rest, mean_rest + sem_rest, alpha=0.3, color="navy")

        plt.plot(times, mean_right, label="Right", color="darkorange")
        plt.fill_between(times, mean_right - sem_right, mean_right + sem_right, alpha=0.3, color="darkorange")

        plt.axvline(0, color="black", linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Mu Power (Envelope)")
        plt.title(f"Mu-band ERD at {target_channel}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_folder / f"mu_power_time_series-{target_channel}.png", dpi=300)
        plt.close()

        baseline_idx = (times > -1.5) & (times < 0.0)
        post_idx = (times > 1.5) & (times < 4.5)

        baseline_power = mu_power_move[:, ch_idx, baseline_idx].mean(axis=1)
        post_power = mu_power_move[:, ch_idx, post_idx].mean(axis=1)
        erd_percent = 100 * (baseline_power - post_power) / baseline_power

        plt.figure(figsize=(8, 4))
        plt.hist(erd_percent, bins=20, color="darkorange", edgecolor="black")
        plt.axvline(0, color="black", linestyle="--")
        plt.title(f"Mu-band ERD per Trial (% Drop) - {target_channel}")
        plt.xlabel("ERD (%)")
        plt.ylabel("Number of Trials")
        plt.tight_layout()
        plt.savefig(plots_folder / f"erd_histogram-{target_channel}.png", dpi=300)
        plt.close()

        stat, p = wilcoxon(erd_percent)
        print("Is the ERD statistically robust across trials?")
        print(f"Wilcoxon signed-rank test on ERD%: p = {p:.4f}")
        if p < 0.05:
            print("Statistically significant ERD across trials. Bc it is below 0.05")
        else:
            print("No statistically significant ERD across trials. Bc it is above 0.05")



def plot_channel_psd_comparison(
    epochs_move: mne.Epochs, epochs_rest: mne.Epochs, epochs_right: mne.Epochs, plots_folder: Path
) -> None:
    for ch in TARGET_CHANNELS:
        print(f"\nChannel: {ch}")

        freqs, psd_left, err_left = get_avg_psd(epochs_move, [ch])
        _, psd_right, err_right = get_avg_psd(epochs_right, [ch])
        _, psd_rest, err_rest = get_avg_psd(epochs_rest, [ch])

        plt.figure(figsize=(10, 5))
        plt.plot(freqs, psd_left, label="Left Hand", color="blue")
        plt.fill_between(freqs, psd_left - err_left, psd_left + err_left, color="blue", alpha=0.2)

        plt.plot(freqs, psd_right, label="Right Hand", color="green")
        plt.fill_between(freqs, psd_right - err_right, psd_right + err_right, color="green", alpha=0.2)

        plt.plot(freqs, psd_rest, label="Rest", color="gray")
        plt.fill_between(freqs, psd_rest - err_rest, psd_rest + err_rest, color="gray", alpha=0.2)

        plt.title(f"PSD - Left vs Right vs Rest ({ch})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (uV^2/Hz)")
        plt.legend()
        plt.grid(True)
        plt.xlim(FMIN, FMAX)
        plt.tight_layout()
        plt.savefig(plots_folder / f"psd_movement_vs_rest_avg_{ch}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(freqs, psd_left - psd_rest, label="Left - Rest", color="blue")
        plt.plot(freqs, psd_right - psd_rest, label="Right - Rest", color="green")
        plt.axhline(0, color="black", linestyle="--")
        plt.title(f"PSD Difference from Rest - {ch}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Delta Power (uV^2/Hz)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_folder / f"psd_diff_from_rest_{ch}.png", dpi=300)
        plt.close()


def main() -> None:
    subj_nr, session_nr, run_nr, task = get_runtime_inputs()
    file_path, plots_folder = build_paths(subj_nr, session_nr, run_nr, task)

    raw, markers, channel_labels = get_raw_offline(file_path)
    print(channel_labels)

    raw.pick(USED_CHANNELS)
    save_raw_diagnostics(raw, plots_folder)
    raw.pick(MOTOR_CHANNELS)

    events, event_id = mne.events_from_annotations(raw)
    print("Event ID mapping:", event_id)

    epochs, _ = get_epochs(raw, markers, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX)
    print("epoch events:", epochs.event_id)

    epochs = apply_autoreject(epochs)

    epochs_witherp = epochs.copy()
    epochs = epochs.copy().crop(tmin=1.5, tmax=4.5)

    epochs_move = epochs[MOVEMENT_LEFT_EVENT]
    epochs_rest = epochs[REST_EVENT]
    epochs_right = epochs[MOVEMENT_RIGHT_EVENT]

    mu_power_move = get_band_power(epochs_move, MU_BAND)
    mu_power_rest = get_band_power(epochs_rest, MU_BAND)
    mu_power_right = get_band_power(epochs_right, MU_BAND)

    beta_power_move = get_band_power(epochs_move, BETA_BAND)
    beta_power_rest = get_band_power(epochs_rest, BETA_BAND)
    beta_power_right = get_band_power(epochs_right, BETA_BAND)

    mu_snr_left = mu_power_move / (mu_power_rest + mu_power_right)
    beta_snr_left = beta_power_move / (beta_power_rest + beta_power_right)
    mu_snr_right = mu_power_right / (mu_power_rest + mu_power_move)
    beta_snr_right = beta_power_right / (beta_power_rest + beta_power_move)

    for ch_idx, ch in enumerate(MOTOR_CHANNELS):
        print(f"\nChannel: {ch}")
        print(f"  Mu-band SNR (8-13 Hz) left: {mu_snr_left[ch_idx]:.2f}")
        print(f"  Beta-band SNR (13-30 Hz) left: {beta_snr_left[ch_idx]:.2f}")
        print(f"  Mu-band SNR (8-13 Hz) right: {mu_snr_right[ch_idx]:.2f}")
        print(f"  Beta-band SNR (13-30 Hz) right: {beta_snr_right[ch_idx]:.2f}")

    plot_psd_overview(epochs_move, epochs_rest, epochs_right, plots_folder)

    mu_move_epochs = get_band_power_per_epoch(epochs_move, MU_BAND)
    mu_rest_epochs = get_band_power_per_epoch(epochs_rest, MU_BAND)
    mu_right_epochs = get_band_power_per_epoch(epochs_right, MU_BAND)

    _, p_vals = ttest_ind(mu_move_epochs, mu_rest_epochs, axis=0, equal_var=False)
    ttest_ind(mu_move_epochs, mu_right_epochs, axis=0, equal_var=False)

    for ch_idx, ch in enumerate(MOTOR_CHANNELS):
        print(f"Mu-band: {ch} - p = {p_vals[ch_idx]:.4f}")

    erd_left = (mu_rest_epochs.mean(axis=0) - mu_move_epochs.mean(axis=0)) / mu_rest_epochs.mean(axis=0)
    erd_right = (mu_rest_epochs.mean(axis=0) - mu_right_epochs.mean(axis=0)) / mu_rest_epochs.mean(axis=0)

    plot_erd(
        erd_left,
        MOTOR_CHANNELS,
        "Mu-band ERD per Channel",
        plots_folder / "mu_erd.png",
        "Mu ERD (Fractional Power Drop)",
    )
    plot_erd(
        erd_right,
        MOTOR_CHANNELS,
        "Mu-band ERD per Channel - RIGHT",
        plots_folder / "mu_erd_right.png",
        "Mu ERD (Fractional Power Drop)",
    )

    left_effects: list[float] = []
    right_effects: list[float] = []
    for ch_idx, ch in enumerate(MOTOR_CHANNELS):
        d_left = cohen_d(mu_move_epochs[:, ch_idx], mu_rest_epochs[:, ch_idx])
        d_right = cohen_d(mu_right_epochs[:, ch_idx], mu_rest_epochs[:, ch_idx])
        print(f"Mu-band: {ch} - Cohen's d = {d_left:.2f}")
        print(f"Mu-band: {ch} - Cohen's d (right) = {d_right:.2f}")
        left_effects.append(d_left)
        right_effects.append(d_right)

    plot_effect_sizes(left_effects, right_effects, MOTOR_CHANNELS, plots_folder)

    epochs_move_long = epochs_witherp[MOVEMENT_LEFT_EVENT]
    epochs_rest_long = epochs_witherp[REST_EVENT]
    epochs_right_long = epochs_witherp[MOVEMENT_RIGHT_EVENT]

    plot_tfr(epochs_move_long, epochs_rest_long, epochs_right_long, plots_folder)

    evoked_move_avg = epochs_move_long.average()
    evoked_move_avg.plot_topomap(times=np.linspace(0.0, 5.0, 12), ch_type="eeg", show=False)
    plt.title("Evoked - Movement")
    plt.savefig(plots_folder / "evoked_movement.png", dpi=300)
    plt.close()

    plot_mu_time_series(epochs_move_long, epochs_rest_long, epochs_right_long, plots_folder)
    plot_channel_psd_comparison(epochs_move, epochs_rest, epochs_right, plots_folder)


if __name__ == "__main__":
    main()
