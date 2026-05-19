from week3.mi_evaluate_plots import (
    build_paths,
    get_epochs,
    get_hilbert_envelope,
    get_raw_offline,
    get_runtime_inputs,
    plot_channel_psd_comparison,
    plot_psd_overview,
    plot_tfr,
    save_raw_diagnostics,
)
import mne
from mne.preprocessing import ICA
from week3.filtering_A1 import apply_causal_filter_iir
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import seaborn as sns


#---------------------Move this later to a config or yaml file------------------------
MOVEMENT_LEFT_EVENT = "ARROW LEFT ONSET"
MOVEMENT_RIGHT_EVENT = "ARROW RIGHT ONSET"
REST_EVENT = "CIRCLE ONSET"

T_ONSET = 1.5
T_OFFSET = 4.5
T_CUE = 0.0

MOTOR_CHANNELS = ["Cz", "C3", "C4", "Pz", "Fz"]
USED_CHANNELS = ["Cz", "C3", "C4", "F4", "P3", "Pz", "Fz"]
TARGET_CHANNELS = ["C3", "Cz", "C4"]

FMIN, FMAX = 1, 40
MU_BAND = (8, 13)
BETA_BAND = (13, 30)
EPOCH_TMIN, EPOCH_TMAX = -1.5, 5.5
#---------------------Move this later to a config or yaml file------------------------


def use_ICA(raw: mne.io.RawArray, n_components=20, random_state=97, max_iter="auto", method="fastica"):
    """
    !!!THIS FUNCTION WILL BE IMPORTED FROM A SEPARATE MODULE IN THE FUTURE!!! (version control must be resolved)

    Fit ICA on raw EEG Data
    Parameters
    ----------
    raw : mne.io.RawArray
        raw data RawArray
    Rest: ICA params

    Returns
    -------
    raw_ica : mne.io.RawArray
        ICA processed RawArray data object.
    ica : ICA object
        For data analysis
    """
    raw_data = raw
    ica = ICA(n_components=n_components, random_state=random_state, max_iter=max_iter, method=method)
    # Fit ICA on bandpass-filtered data (1-40 Hz) for cleaner component detection
    ica.fit(raw_data.copy().filter(1., 40., phase="zero"))
    raw_ica = raw_data.copy()
    return raw_ica, ica


def preprocess_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Preprocess the raw data by applying a bandpass filter and ICA for artifact removal.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object.

    Returns
    -------
    raw : mne.io.Raw
        Preprocessed raw data object.
    """

    raw = apply_causal_filter_iir(raw)
    # Extract ICA-processed data (discard ICA object)
    raw = use_ICA(raw)[0]
    return raw


def plot_erd_ers_timecourse(
    epochs_by_condition: dict[str, mne.Epochs],
    band: tuple[float, float],
    band_name: str,
    channels: list[str],
    baseline: tuple[float, float],
    plots_folder: Path,
    baseline_condition: str | None = None,
    plot_baseline_condition: bool = False,
) -> None:
    """
    Plot trial-averaged ERD/ERS time course per condition.

    ERD/ERS is computed as:
        100 * (power(t) - baseline_power) / baseline_power

    Negative values = ERD
    Positive values = ERS

    Parameters
    ----------
    epochs_by_condition:
        Dictionary mapping condition names to MNE Epochs.

    band:
        Frequency band, e.g. (8, 13) for mu.

    band_name:
        Name used in plot title and filename.

    channels:
        Channels to plot.

    baseline:
        Time-window baseline used when baseline_condition is None.
        Example: (-1.0, 0.0)

    plots_folder:
        Folder where plots are saved.

    baseline_condition:
        Optional condition name to use as the baseline, e.g. "Rest".
        If None, uses the time-window baseline from each trial.

    plot_baseline_condition:
        If False, the baseline condition itself is not plotted.
        Usually keep this False when baseline_condition="Rest".
    """

    # ------------------------------------------------------------
    # Optional: compute condition-based baseline, e.g. rest baseline
    # ------------------------------------------------------------
    condition_baseline_power = None

    if baseline_condition is not None:
        if baseline_condition not in epochs_by_condition:
            raise ValueError(
                f"baseline_condition='{baseline_condition}' not found in epochs_by_condition. "
                f"Available conditions: {list(epochs_by_condition.keys())}"
            )

        baseline_epochs = epochs_by_condition[baseline_condition]

        # Compute amplitude envelope via Hilbert transform, then square for power
        baseline_power = get_hilbert_envelope(
            baseline_epochs, band
        )  # shape: trials x channels x times

        # Average over rest trials and time.
        # Result shape: channels
        condition_baseline_power = baseline_power.mean(axis=(0, 2))

        # Reshape for broadcasting against power:
        # power shape = trials x channels x times
        # baseline shape = 1 x channels x 1
        condition_baseline_power = condition_baseline_power.reshape(1, -1, 1)

    # ------------------------------------------------------------
    # Plot each requested channel
    # ------------------------------------------------------------
    for ch in channels:
        plt.figure(figsize=(10, 5))

        for condition_name, epochs in epochs_by_condition.items():
            if (
                baseline_condition is not None
                and condition_name == baseline_condition
                and not plot_baseline_condition
            ):
                continue

            # Bandpass + Hilbert envelope squared = time-resolved band power
            power = get_hilbert_envelope(epochs, band)  # shape: trials x channels x times

            times = epochs.times

            if ch not in epochs.ch_names:
                raise ValueError(
                    f"Channel '{ch}' not found in condition '{condition_name}'. "
                    f"Available channels: {epochs.ch_names}"
                )

            ch_idx = epochs.ch_names.index(ch)

            # ------------------------------------------------------------
            # Choose baseline mode
            # ------------------------------------------------------------
            if baseline_condition is None:
                baseline_idx = (times >= baseline[0]) & (times <= baseline[1])

                if not baseline_idx.any():
                    raise ValueError(
                        f"No baseline samples found for baseline={baseline}. "
                        f"Epoch time range is {times[0]:.2f} to {times[-1]:.2f} s."
                    )

                # Baseline power per trial and channel
                # shape: trials x channels x 1
                baseline_power = power[:, :, baseline_idx].mean(axis=2, keepdims=True)

                ylabel = "% change from baseline"
                title_suffix = ""
                filename_suffix = ""

            else:
                # Same condition baseline for all trials/conditions
                # shape: 1 x channels x 1
                baseline_power = condition_baseline_power

                ylabel = f"% change from {baseline_condition}"
                title_suffix = f" vs {baseline_condition}"
                filename_suffix = f"_vs_{baseline_condition.lower().replace(' ', '_')}"

            # Avoid division by zero
            eps = np.finfo(float).eps

            # Percent change from baseline
            erd_ers = 100 * (power - baseline_power) / (baseline_power + eps)

            # Select channel and average across trials
            ch_erd_ers = erd_ers[:, ch_idx, :]

            mean_erd_ers = ch_erd_ers.mean(axis=0)
            sem_erd_ers = ch_erd_ers.std(axis=0) / np.sqrt(ch_erd_ers.shape[0])

            plt.plot(times, mean_erd_ers, label=condition_name)
            plt.fill_between(
                times,
                mean_erd_ers - sem_erd_ers,
                mean_erd_ers + sem_erd_ers,
                alpha=0.25,
            )

        plt.axhline(0, color="black", linewidth=0.8)
        plt.axvline(0, color="black", linestyle="--", label="Cue")
        plt.axvline(T_ONSET, color="green", linestyle="--", label="MI onset")
        plt.axvline(T_OFFSET, color="blue", linestyle="--", label="MI offset")

        plt.xlabel("Time relative to cue (s)")
        plt.ylabel(ylabel)
        plt.title(f"{band_name} ERD/ERS Time Course{title_suffix} - {ch}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        safe_band_name = band_name.lower().replace(" ", "_")
        safe_ch_name = ch.replace(" ", "_")
        plt.savefig(
            plots_folder / f"{safe_band_name}_erd_ers_timecourse_{safe_ch_name}{filename_suffix}.png",
            dpi=300,
        )
        plt.close()


def plot_erp_average(
    epochs_by_condition: dict[str, mne.Epochs],
    channels: list[str],
    plots_folder: Path,
    tmin: float | None = None,
    tmax: float | None = None,
    separate_conditions: bool = False,
) -> None:
    """
    Plot ERP average across trials for selected channels.

    ERP = trial-averaged time-domain EEG signal.
    This plots the evoked response for each condition and channel.
    If `separate_conditions` is True, create one plot per input condition.
    """

    if separate_conditions:
        for condition_name, epochs in epochs_by_condition.items():
            for ch in channels:
                plt.figure(figsize=(10, 5))

                if ch not in epochs.ch_names:
                    raise ValueError(
                        f"Channel {ch} not found in epochs for condition {condition_name}. "
                        f"Available channels: {epochs.ch_names}"
                    )

                evoked = epochs.average()
                ch_idx = evoked.ch_names.index(ch)

                times = evoked.times
                signal = evoked.data[ch_idx, :] * 1e6  # convert V to µV

                plt.plot(times, signal, label=condition_name)

                plt.axhline(0, color="black", linewidth=0.8)
                plt.axvline(0, color="black", linestyle="--", label="Cue")
                plt.axvline(T_ONSET, color="green", linestyle="--", label="MI onset")
                plt.axvline(T_OFFSET, color="blue", linestyle="--", label="MI offset")

                if tmin is not None or tmax is not None:
                    plt.xlim(tmin, tmax)

                plt.xlabel("Time relative to cue (s)")
                plt.ylabel("Amplitude (µV)")
                plt.title(f"ERP Average - {condition_name} - {ch}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                safe_condition = condition_name.lower().replace(" ", "_")
                plt.savefig(
                    plots_folder / f"erp_average_{safe_condition}_{ch}.png",
                    dpi=300,
                )
                plt.close()
    else:
        for ch in channels:
            plt.figure(figsize=(10, 5))

            for condition_name, epochs in epochs_by_condition.items():
                if ch not in epochs.ch_names:
                    raise ValueError(
                        f"Channel {ch} not found in epochs for condition {condition_name}. "
                        f"Available channels: {epochs.ch_names}"
                    )

                evoked = epochs.average()
                ch_idx = evoked.ch_names.index(ch)

                times = evoked.times
                signal = evoked.data[ch_idx, :] * 1e6  # convert V to µV

                plt.plot(times, signal, label=condition_name)

            plt.axhline(0, color="black", linewidth=0.8)
            plt.axvline(0, color="black", linestyle="--", label="Cue")
            plt.axvline(T_ONSET, color="green", linestyle="--", label="MI onset")
            plt.axvline(T_OFFSET, color="blue", linestyle="--", label="MI offset")

            if tmin is not None or tmax is not None:
                plt.xlim(tmin, tmax)

            plt.xlabel("Time relative to cue (s)")
            plt.ylabel("Amplitude (µV)")
            plt.title(f"ERP Average - {ch}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(plots_folder / f"erp_average_{ch}.png", dpi=300)
            plt.close()


def plot_tfr_multiplt():
    """
    Placeholder for MNE-style TFR plotting function based on https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html
    """
    pass


def plot_erds_mu_beta_timecourses_mne_style(
    epochs_by_condition: dict[str, mne.Epochs],
    channels: list[str],
    plots_folder: Path,
    baseline: tuple[float, float] = (-1.0, 0.0),
    freqs: np.ndarray | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    decim: int = 2,
    n_boot: int = 10,
) -> None:
    """
    !!!THIS FUNCTION WILL BE DEPRECATED IN FAVOR OF plot_tfr_multiplt()!!!

    Plot MNE-style ERD/ERS band time courses for mu and beta.

    Process:
        1. Combine conditions into one Epochs object
        2. Compute single-trial TFR with multitaper
        3. Apply baseline correction with mode='percent'
        4. Convert TFR to dataframe
        5. Average frequencies into mu and beta bands
        6. Plot ERD/ERS time courses with seaborn FacetGrid

    Notes:
        MNE mode='percent' returns fractional change:
            -0.5 = -50%
             0.5 = +50%

        This function multiplies by 100, so the plotted values are true percent.
    """
    plots_folder.mkdir(parents=True, exist_ok=True)

    if freqs is None:
        freqs = np.arange(8, 31)  # 8–30 Hz: mu + beta

    n_cycles = freqs

    # ------------------------------------------------------------
    # Combine conditions into one Epochs object
    # ------------------------------------------------------------
    epochs_list = []
    next_event_code = 1

    for condition_name, epochs in epochs_by_condition.items():
        available_channels = [ch for ch in channels if ch in epochs.ch_names]

        if not available_channels:
            raise ValueError(
                f"No requested channels found for condition '{condition_name}'. "
                f"Requested: {channels}, available: {epochs.ch_names}"
            )

        epochs_cond = epochs.copy().pick(available_channels)

        # Recode events so each input condition becomes one clean condition label
        epochs_cond.events[:, 2] = next_event_code
        epochs_cond.event_id = {condition_name: next_event_code}

        epochs_list.append(epochs_cond)
        next_event_code += 1

    epochs_all = mne.concatenate_epochs(epochs_list)

    if tmin is not None or tmax is not None:
        epochs_all = epochs_all.copy().crop(tmin=tmin, tmax=tmax)

    # ------------------------------------------------------------
    # Compute TFR
    # ------------------------------------------------------------
    tfr = epochs_all.compute_tfr(
        method="multitaper",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=decim,
    )

    tfr.apply_baseline(
        baseline=baseline,
        mode="percent",
    )

    # ------------------------------------------------------------
    # Convert to dataframe
    # ------------------------------------------------------------
    df = tfr.to_data_frame(
        time_format=None,
        long_format=True,
    )

    # Verify condition column exists (required for FacetGrid grouping)
    if "condition" not in df.columns:
        raise RuntimeError(
            "No 'condition' column found in TFR dataframe. "
            "Check that epochs_all.event_id contains condition names."
        )

    # Convert MNE fractional percent to true percent
    df["value"] *= 100.0

    # -------- Map frequencies into mu (8-13 Hz) and beta (13-30 Hz) bands --------
    band_edges = [8, 13, 30]
    band_labels = ["mu", "beta"]

    df["band"] = pd.cut(
        df["freq"],
        bins=band_edges,
        labels=band_labels,
        include_lowest=True,
    )

    df = df[df["band"].isin(["mu", "beta"])].copy()
    df["band"] = df["band"].cat.remove_unused_categories()

    # ------------------------------------------------------------
    # Order channels
    # ------------------------------------------------------------
    available_channels = [ch for ch in channels if ch in df["channel"].unique()]

    df["channel"] = df["channel"].cat.reorder_categories(
        available_channels,
        ordered=True,
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    g = sns.FacetGrid(
        df,
        row="band",
        col="channel",
        margin_titles=True,
        sharex=True,
        sharey=True,
    )

    g.map(
        sns.lineplot,
        "time",
        "value",
        "condition",
        n_boot=n_boot,
    )

    axline_kw = dict(
        color="black",
        linestyle="dashed",
        linewidth=0.5,
        alpha=0.5,
    )

    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)

    for ax in g.axes.flat:
        ax.axvline(T_ONSET, color="green", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.axvline(T_OFFSET, color="blue", linestyle="--", linewidth=0.7, alpha=0.7)

    g.set_axis_labels("Time (s)", "% change from baseline")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set(ylim=(-100, 150))

    g.add_legend(ncol=2, loc="lower center")

    g.fig.suptitle("Mu/Beta ERD/ERS Time Courses")
    g.fig.subplots_adjust(
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.08,
    )

    output_path = plots_folder / "erds_mu_beta_timecourses_mne_style.png"
    g.fig.savefig(output_path, dpi=300)
    plt.close(g.fig)


def plot_erds_tfr_percent_maps(
    epochs_by_condition: dict[str, mne.Epochs],
    channels: list[str],
    plots_folder: Path,
    freqs: np.ndarray | None = None,
    baseline: tuple[float, float] = (-1.0, 0.0),
    tmin: float = -1.0,
    tmax: float = 4.5,
    method: str = "multitaper",
    decim: int = 2,
    vmin: float = -100.0,
    vmax: float = 150.0,
    show: bool = False,
) -> None:
    """
    !!!THIS FUNCTION WILL BE DEPRECATED IN FAVOR OF plot_tfr_multiplt()!!!

    Plot ERD/ERS time-frequency maps as percent change from baseline.

    This follows the MNE ERDS example style:
        epochs.compute_tfr(method="multitaper", average=False)
        tfr.crop(...).apply_baseline(..., mode="percent")

    Important:
        MNE's mode="percent" returns fractional change:
            -1.0 = -100%
             0.5 = +50%

        This function multiplies by 100 before plotting, so the colorbar
        and plot values are actual percent values.

    Parameters
    ----------
    epochs_by_condition:
        Dict mapping condition name to Epochs.
        Example:
            {
                "Left hand MI": epochs_left,
                "Right hand MI": epochs_right,
            }

    channels:
        Channels to plot, e.g. ["C3", "Cz", "C4"].

    plots_folder:
        Output folder.

    freqs:
        Frequencies for TFR. Default is 2-35 Hz.

    baseline:
        Baseline interval in seconds, usually (-1.0, 0.0).

    tmin, tmax:
        Time range to crop and plot.

    method:
        TFR method. Default "multitaper", matching the MNE example.

    decim:
        Decimation factor for speed.

    vmin, vmax:
        Color limits in percent. For example:
            -100 = -100% ERD
             150 = +150% ERS

    show:
        Whether to display figures interactively.
    """

    plots_folder.mkdir(parents=True, exist_ok=True)

    if freqs is None:
        freqs = np.arange(2, 36)  # 2–35 Hz, matching MNE example

    n_cycles = freqs

    # Center colormap at zero to visually distinguish ERD (negative, blue) from ERS (positive, red)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    for condition_name, epochs in epochs_by_condition.items():
        available_channels = [ch for ch in channels if ch in epochs.ch_names]

        if not available_channels:
            raise ValueError(
                f"None of the requested channels {channels} were found in "
                f"condition '{condition_name}'. Available channels: {epochs.ch_names}"
            )

        # Work on selected channels only
        epochs_sel = epochs.copy().pick(available_channels)

        # Compute single-trial TFR
        tfr = epochs_sel.compute_tfr(
            method=method,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            average=False,
            decim=decim,
        )

        # Crop and baseline-correct.
        # mode="percent" gives fractional percent change:
        # -1.0 = -100%, +0.5 = +50%.
        tfr.crop(tmin=tmin, tmax=tmax)
        tfr.apply_baseline(baseline=baseline, mode="percent")

        # Average across trials, then convert fraction -> real percent
        tfr_avg = tfr.average()
        tfr_avg.data *= 100.0

        fig, axes = plt.subplots(
            1,
            len(available_channels) + 1,
            figsize=(4 * len(available_channels) + 1.2, 4),
            gridspec_kw={"width_ratios": [10] * len(available_channels) + [1]},
        )

        if len(available_channels) == 1:
            axes = np.array([axes[0], axes[1]])

        image = None

        for ch_idx, ch_name in enumerate(available_channels):
            ax = axes[ch_idx]

            tfr_avg.plot(
                picks=[ch_name],
                axes=ax,
                cmap="RdBu_r",
                cnorm=cnorm,
                colorbar=False,
                show=False,
            )

            ax.set_title(ch_name)
            ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Cue")
            ax.axvline(T_ONSET, color="green", linestyle="--", linewidth=1, label="MI onset")
            ax.axvline(T_OFFSET, color="blue", linestyle="--", linewidth=1, label="MI offset")

            if ch_idx != 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            # Grab image for shared colorbar
            if ax.images:
                image = ax.images[-1]

        if image is not None:
            cbar = fig.colorbar(image, cax=axes[-1])
            cbar.set_label("% change from baseline")
        else:
            axes[-1].axis("off")

        fig.suptitle(f"ERD/ERS Time-Frequency Map - {condition_name}")
        fig.tight_layout()

        safe_condition = (
            condition_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        fig.savefig(
            plots_folder / f"erds_tfr_percent_{safe_condition}.png",
            dpi=300,
        )

        if show:
            plt.show()

        plt.close(fig)


def main():
    subj_nr, session_nr, run_nr, task = get_runtime_inputs()
    file_path, plots_folder = build_paths(subj_nr, session_nr, run_nr, task)

    raw, markers, channel_labels = get_raw_offline(file_path)
    print(channel_labels)

    # Preprocess raw data (filtering, ICA) A3.1
    raw = preprocess_raw(raw)

    save_raw_diagnostics(raw, plots_folder) # Come back and change this to better reflect preprocessing, incorperate its filtering into preprocessing function
    
    raw.pick(MOTOR_CHANNELS)

    events, event_id = mne.events_from_annotations(raw)
    print("Event ID mapping:", event_id)

    epochs, _ = get_epochs(raw, markers, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX)
    print("epoch events:", epochs.event_id)

    # Keep full epochs for ERP analysis, create cropped version for power analysis
    epochs_witherp = epochs.copy()
    epochs = epochs.copy().crop(tmin=1.5, tmax=4.5)

    # Split cropped epochs by condition
    epochs_move = epochs[MOVEMENT_LEFT_EVENT]
    epochs_rest = epochs[REST_EVENT]
    epochs_right = epochs[MOVEMENT_RIGHT_EVENT]

    # Extract same conditions from full-length epochs
    epochs_move_long = epochs_witherp[MOVEMENT_LEFT_EVENT]
    epochs_rest_long = epochs_witherp[REST_EVENT]
    epochs_right_long = epochs_witherp[MOVEMENT_RIGHT_EVENT]


    # ERD/ERS Plots A3.2
    # Plotting with rest as baseline condition
    conditions_vs_rest = {
        "Left hand MI": epochs_move_long,
        "Right hand MI": epochs_right_long,
        "Rest": epochs_rest_long,
    }
    plot_erd_ers_timecourse(
        epochs_by_condition=conditions_vs_rest,
        band=MU_BAND,
        band_name="Mu band",
        channels=TARGET_CHANNELS,
        baseline=(-1.0, 0.0),
        plots_folder=plots_folder,
        baseline_condition="Rest",
    )
    plot_erd_ers_timecourse(
        epochs_by_condition=conditions_vs_rest,
        band=BETA_BAND,
        band_name="Beta band",
        channels=TARGET_CHANNELS,
        baseline=(-1.0, 0.0),
        plots_folder=plots_folder,
        baseline_condition="Rest",
    )

    # Plotting with no baseline condition, using time-window baseline instead (standard practice for ERD/ERS)
    conditions = {
        "Left hand MI": epochs_move_long,
        "Right hand MI": epochs_right_long,
    }
    plot_erd_ers_timecourse(
        epochs_by_condition=conditions,
        band=MU_BAND,
        band_name="Mu band",
        channels=TARGET_CHANNELS,
        baseline=(-1.0, 0.0),
        plots_folder=plots_folder,
        baseline_condition=None,
    )
    plot_erd_ers_timecourse(
        epochs_by_condition=conditions,
        band=BETA_BAND,
        band_name="Beta band",
        channels=TARGET_CHANNELS,
        baseline=(-1.0, 0.0),
        plots_folder=plots_folder,
        baseline_condition=None,
    )

    plot_tfr(epochs_move_long, epochs_rest_long, epochs_right_long, plots_folder)

    # PSD Plots: A3.3
    plot_psd_overview(epochs_move, epochs_rest, epochs_right, plots_folder)
    plot_channel_psd_comparison(epochs_move, epochs_rest, epochs_right, plots_folder)

    # ERP average Plots A3.4. 
    # NOTE: If separate_conditions=True, ERP plots will be generated per condition. False will plot all conditions on the same plot.
    erp_conditions = {
        "Left hand MI": epochs_move_long,
        "Right hand MI": epochs_right_long,
        "Rest": epochs_rest_long,
    }
    plot_erp_average(
        epochs_by_condition=erp_conditions,
        separate_conditions=True,
        channels=TARGET_CHANNELS,
        plots_folder=plots_folder,
        # Limit to motor task window for clarity
        tmin=T_ONSET,
        tmax=T_OFFSET,
    )

if __name__ == "__main__":
    main()