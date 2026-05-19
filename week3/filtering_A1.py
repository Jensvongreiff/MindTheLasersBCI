import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from week3.loading_helpers import load_one_channel_data
import mne

# Function to apply causal 4th order IIR filter (for online MI)

def apply_causal_filter_iir(raw: mne.io.RawArray) -> mne.io.RawArray:
    """
    Apply a causal 4th-order Butterworth IIR band-pass filter from 8–30 Hz
    to an MNE RawArray.

    This uses sosfilt, not sosfiltfilt, so it is suitable for online/causal
    filtering. The returned RawArray is a filtered copy of the input.
    """
    raw_filt = raw.copy()

    fs = raw.info["sfreq"]

    sos = sig.butter(
        N=4,
        Wn=[8, 30],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    data = raw_filt.get_data()  # shape: (n_channels, n_samples)

    filtered_data = sig.sosfilt(
        sos,
        data,
        axis=1,  # filter along time axis
    )

    raw_filt._data = filtered_data

    return raw_filt


def apply_causal_filter_iir_one_channel(x):
    sos = sig.butter(
        N=4,
        Wn=[8,30],
        btype="bandpass",
        fs=250,
        output="sos",
    )
    return sig.sosfilt(sos, x)


# ============================================================
# Helper functions
# ============================================================

def causal_filter_fir(b, x):
    """Apply causal FIR filtering."""
    y = sig.lfilter(b, [1.0], x)
    return y


def causal_filter_iir_sos(sos, x):
    """Apply causal IIR filtering using SOS."""
    y = sig.sosfilt(sos, x)
    return y


def db(x, floor=1e-12):
    """Convert magnitude to dB safely."""
    return 20 * np.log10(np.maximum(np.abs(x), floor))


def frequency_response_fir(b, fs, n=8192):
    """Return frequency response for FIR filter."""
    f, h = sig.freqz(b, worN=n, fs=fs)
    return f, h


def frequency_response_iir_sos(sos, fs, n=8192):
    """Return frequency response for SOS IIR filter."""
    f, h = sig.sosfreqz(sos, worN=n, fs=fs)
    return f, h


def group_delay_from_phase(f, h, fs):
    """
    Estimate group delay from unwrapped phase.

    Group delay in samples:
        gd = -d(phi) / d(omega)

    where omega is digital rad/sample:
        omega = 2*pi*f/fs
    """
    phase = np.unwrap(np.angle(h))
    omega = 2 * np.pi * f / fs

    gd_samples = -np.gradient(phase, omega)
    gd_ms = 1000 * gd_samples / fs

    return gd_samples, gd_ms


def passband_mask(f, bp):
    return (f >= bp[0]) & (f <= bp[1])


def stopband_mask(f):
    """
    Stopband regions used for rough comparison.
    We ignore the transition bands near 8 Hz and 30 Hz.
    """
    return ((f >= 0.5) & (f <= 5)) | ((f >= 40) & (f <= 120))


def summarize_filter(name, f, h, gd_ms, bp):
    """
    Compute rough summary metrics:
    - passband ripple
    - median passband group delay
    - maximum passband group delay
    - approximate stopband attenuation
    """
    mag_db = db(h)
    pb = passband_mask(f, bp)
    sb = stopband_mask(f)

    pass_mag = mag_db[pb]
    stop_mag = mag_db[sb]
    pass_gd = gd_ms[pb]

    passband_ripple_db = np.max(pass_mag) - np.min(pass_mag)
    stopband_max_db = np.max(stop_mag)
    stopband_attenuation_db = -stopband_max_db

    return {
        "name": name,
        "passband_ripple_db": passband_ripple_db,
        "stopband_attenuation_db": stopband_attenuation_db,
        "median_group_delay_ms": np.median(pass_gd),
        "max_group_delay_ms": np.max(pass_gd),
    }


def benchmark_fir_filter(b, x, block_size, repeats):
    """
    Benchmark causal block-based FIR filtering.
    This simulates online processing using lfilter with state.
    """
    zi = np.zeros(len(b) - 1)
    n = len(x)

    start = time.perf_counter()

    for _ in range(repeats):
        zi_run = zi.copy()
        for i in range(0, n, block_size):
            block = x[i:i + block_size]
            _, zi_run = sig.lfilter(b, [1.0], block, zi=zi_run)

    end = time.perf_counter()

    total_blocks = repeats * int(np.ceil(n / block_size))
    mean_processing_time_per_block_s = (end - start) / total_blocks

    return mean_processing_time_per_block_s


def benchmark_iir_sos_filter(sos, x, block_size, repeats):
    """
    Benchmark causal block-based IIR SOS filtering.
    This simulates online processing using sosfilt with state.
    """
    zi = sig.sosfilt_zi(sos)
    n = len(x)

    start = time.perf_counter()

    for _ in range(repeats):
        zi_run = zi.copy()
        for i in range(0, n, block_size):
            block = x[i:i + block_size]
            _, zi_run = sig.sosfilt(sos, block, zi=zi_run)

    end = time.perf_counter()

    total_blocks = repeats * int(np.ceil(n / block_size))
    mean_processing_time_per_block_s = (end - start) / total_blocks

    return mean_processing_time_per_block_s



if __name__ == "__main__":
    # ============================================================
    # Configuration
    # ============================================================

    fs = 250  # EEG sampling frequency in Hz
    bp = [8, 30]  # motor imagery band-pass range in Hz


    output_dir = "plots/filtering"
    os.makedirs(output_dir, exist_ok=True)

    # Online buffering configuration
    # Example: process EEG in blocks of 64 samples.
    # At fs = 500 Hz, this block is 128 ms long.
    block_size = 64

    # Benchmark configuration
    benchmark_repeats = 500


    # ============================================================
    # Load EEG signal
    # ============================================================

    x = load_one_channel_data("/home/dani/Documents/TUM/3.Semester/BCI/baseline-bci-26/data/sub-P999/eeg/sub-P999_ses-S002_task-arrow_run-001_eeg.xdf", 0)
    x = np.asarray(x).squeeze()

    # Create matching time vector
    t = np.arange(len(x)) / fs




    # ============================================================
    # Design candidate filters
    # ============================================================

    fir_numtaps_candidates = [101, 201, 401]
    iir_order_candidates = [2, 4, 6]

    fir_filters = {}
    iir_filters = {}

    for numtaps in fir_numtaps_candidates:
        b = sig.firwin(
            numtaps=numtaps,
            cutoff=bp,
            fs=fs,
            pass_zero=False,
            window="hamming",
        )
        fir_filters[numtaps] = b

    for order in iir_order_candidates:
        sos = sig.butter(
            N=order,
            Wn=bp,
            btype="bandpass",
            fs=fs,
            output="sos",
        )
        iir_filters[order] = sos


    # ============================================================
    # Analyze all candidates
    # ============================================================

    summaries = []

    for numtaps, b in fir_filters.items():
        f, h = frequency_response_fir(b, fs)
        gd_samples, gd_ms = group_delay_from_phase(f, h, fs)

        summary = summarize_filter(
            name=f"FIR numtaps={numtaps}",
            f=f,
            h=h,
            gd_ms=gd_ms,
            bp=bp,
        )
        summaries.append(summary)

    for order, sos in iir_filters.items():
        f, h = frequency_response_iir_sos(sos, fs)
        gd_samples, gd_ms = group_delay_from_phase(f, h, fs)

        summary = summarize_filter(
            name=f"IIR Butterworth order={order}",
            f=f,
            h=h,
            gd_ms=gd_ms,
            bp=bp,
        )
        summaries.append(summary)


    print("\nCandidate filter summaries:")
    print("-" * 90)
    for s in summaries:
        print(
            f"{s['name']:<30} | "
            f"Ripple: {s['passband_ripple_db']:6.2f} dB | "
            f"Stop attenuation: {s['stopband_attenuation_db']:6.2f} dB | "
            f"Median GD: {s['median_group_delay_ms']:7.2f} ms | "
            f"Max GD: {s['max_group_delay_ms']:7.2f} ms"
        )


    # ============================================================
    # Choose final filters for detailed comparison
    # ============================================================

    # FIR candidate:
    # 401 taps gives a clearer transition but high latency.
    chosen_fir_numtaps = 401
    b_fir = fir_filters[chosen_fir_numtaps]

    # IIR candidate:
    # 4th order Butterworth is a good latency/attenuation compromise.
    chosen_iir_order = 4
    sos_iir = iir_filters[chosen_iir_order]

    # Also get b, a form for plotting with scipy if needed, but SOS is preferred for implementation.
    b_iir, a_iir = sig.butter(
        N=chosen_iir_order,
        Wn=bp,
        btype="bandpass",
        fs=fs,
        output="ba",
    )


    # ============================================================
    # Frequency, phase, and group delay of chosen filters
    # ============================================================

    f_fir, h_fir = frequency_response_fir(b_fir, fs)
    gd_fir_samples, gd_fir_ms = group_delay_from_phase(f_fir, h_fir, fs)

    f_iir, h_iir = frequency_response_iir_sos(sos_iir, fs)
    gd_iir_samples, gd_iir_ms = group_delay_from_phase(f_iir, h_iir, fs)

    phase_fir = np.unwrap(np.angle(h_fir))
    phase_iir = np.unwrap(np.angle(h_iir))


    # ============================================================
    # Apply causal filters
    # ============================================================

    y_fir = causal_filter_fir(b_fir, x)
    y_iir = causal_filter_iir_sos(sos_iir, x)


    # ============================================================
    # Benchmark processing time
    # ============================================================

    fir_proc_time_s = benchmark_fir_filter(
        b=b_fir,
        x=x,
        block_size=block_size,
        repeats=benchmark_repeats,
    )

    iir_proc_time_s = benchmark_iir_sos_filter(
        sos=sos_iir,
        x=x,
        block_size=block_size,
        repeats=benchmark_repeats,
    )

    fir_proc_time_ms = 1000 * fir_proc_time_s
    iir_proc_time_ms = 1000 * iir_proc_time_s


    # ============================================================
    # Latency calculation
    # ============================================================

    # Buffering latency:
    # If the online system waits until a full block arrives before processing,
    # then the first sample in a block waits block_size/fs seconds.
    # This is a conservative worst-case buffering latency.
    buffer_latency_ms = 1000 * block_size / fs

    # FIR causal group delay:
    fir_group_delay_samples = (chosen_fir_numtaps - 1) / 2
    fir_group_delay_ms_exact = 1000 * fir_group_delay_samples / fs

    # IIR group delay:
    # Since IIR group delay is frequency-dependent, report median and maximum
    # inside the passband.
    pb_iir = passband_mask(f_iir, bp)
    iir_group_delay_median_ms = np.median(gd_iir_ms[pb_iir])
    iir_group_delay_max_ms = np.max(gd_iir_ms[pb_iir])

    # Conservative full latency estimates
    fir_full_latency_ms = buffer_latency_ms + fir_group_delay_ms_exact + fir_proc_time_ms
    iir_full_latency_median_ms = buffer_latency_ms + iir_group_delay_median_ms + iir_proc_time_ms
    iir_full_latency_max_ms = buffer_latency_ms + iir_group_delay_max_ms + iir_proc_time_ms

    print("\nLatency report:")
    print("-" * 90)
    print(f"Sampling frequency: {fs} Hz")
    print(f"Block size: {block_size} samples")
    print(f"Buffering latency: {buffer_latency_ms:.3f} ms")
    print()
    print(f"Chosen FIR: numtaps={chosen_fir_numtaps}")
    print(f"FIR exact group delay: {fir_group_delay_ms_exact:.3f} ms")
    print(f"FIR processing time per block: {fir_proc_time_ms:.6f} ms")
    print(f"FIR full latency estimate: {fir_full_latency_ms:.3f} ms")
    print()
    print(f"Chosen IIR: Butterworth order={chosen_iir_order}, SOS implementation")
    print(f"IIR median passband group delay: {iir_group_delay_median_ms:.3f} ms")
    print(f"IIR max passband group delay: {iir_group_delay_max_ms:.3f} ms")
    print(f"IIR processing time per block: {iir_proc_time_ms:.6f} ms")
    print(f"IIR full latency estimate, median GD: {iir_full_latency_median_ms:.3f} ms")
    print(f"IIR full latency estimate, max GD: {iir_full_latency_max_ms:.3f} ms")


    # ============================================================
    # Plot 1: Frequency response of all candidates
    # ============================================================

    plt.figure(figsize=(10, 6))

    for numtaps, b in fir_filters.items():
        f, h = frequency_response_fir(b, fs)
        plt.plot(f, db(h), label=f"FIR {numtaps} taps")

    for order, sos in iir_filters.items():
        f, h = frequency_response_iir_sos(sos, fs)
        plt.plot(f, db(h), linestyle="--", label=f"IIR Butter order {order}")

    plt.axvline(bp[0], linestyle=":", linewidth=1)
    plt.axvline(bp[1], linestyle=":", linewidth=1)
    plt.xlim(0, 80)
    plt.ylim(-80, 5)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.title("Frequency response: FIR and IIR candidates")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_frequency_response_candidates.png"), dpi=200)
    plt.close()


    # ============================================================
    # Plot 2: Chosen filter frequency response
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.plot(f_fir, db(h_fir), label=f"FIR {chosen_fir_numtaps} taps")
    plt.plot(f_iir, db(h_iir), label=f"IIR Butterworth order {chosen_iir_order}")
    plt.axvline(bp[0], linestyle=":", linewidth=1)
    plt.axvline(bp[1], linestyle=":", linewidth=1)
    plt.xlim(0, 80)
    plt.ylim(-80, 5)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.title("Chosen filters: frequency response")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_frequency_response_chosen.png"), dpi=200)
    plt.close()


    # ============================================================
    # Plot 3: Phase response
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.plot(f_fir, phase_fir, label=f"FIR {chosen_fir_numtaps} taps")
    plt.plot(f_iir, phase_iir, label=f"IIR Butterworth order {chosen_iir_order}")
    plt.axvline(bp[0], linestyle=":", linewidth=1)
    plt.axvline(bp[1], linestyle=":", linewidth=1)
    plt.xlim(0, 80)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Unwrapped phase [rad]")
    plt.title("Phase response")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_phase_response.png"), dpi=200)
    plt.close()


    # ============================================================
    # Plot 4: Group delay
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.plot(f_fir, gd_fir_ms, label=f"FIR {chosen_fir_numtaps} taps")
    plt.plot(f_iir, gd_iir_ms, label=f"IIR Butterworth order {chosen_iir_order}")
    plt.axvline(bp[0], linestyle=":", linewidth=1)
    plt.axvline(bp[1], linestyle=":", linewidth=1)
    plt.xlim(0, 80)
    plt.ylim(0, min(600, np.nanmax(gd_fir_ms[(f_fir > 1) & (f_fir < 80)]) * 1.1))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Group delay [ms]")
    plt.title("Group delay")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_group_delay.png"), dpi=200)
    plt.close()


    # ============================================================
    # Plot 5: Raw and filtered signal
    # ============================================================

    plot_start_s = 5.0
    plot_end_s = 7.0
    scale = 1e-5

    idx = (t >= plot_start_s) & (t <= plot_end_s)

    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axs[0].plot(t[idx], x[idx] / scale)
    axs[0].set_title("Raw signal")
    axs[0].set_ylabel("Amplitude / 1e-5")
    axs[0].grid(True)

    axs[1].plot(t[idx], y_fir[idx] / scale)
    axs[1].set_title(f"Causal FIR filtered signal, {chosen_fir_numtaps} taps")
    axs[1].set_ylabel("Amplitude / 1e-5")
    axs[1].grid(True)

    axs[2].plot(t[idx], y_iir[idx] / scale)
    axs[2].set_title(f"Causal IIR filtered signal, Butterworth order {chosen_iir_order}")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Amplitude / 1e-5")
    axs[2].grid(True)

    plt.suptitle("Signal before and after filtering, 5s to 7s")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_signal_5s_to_7s_scaled_1e-5.png"), dpi=200)
    plt.close()


    # ============================================================
    # Plot 6: Zoomed signal after transient
    # ============================================================

    # Causal filters have startup transients. Plot later part.
    plot_start_s = 5.0
    plot_end_s = 7.0
    idx = (t >= plot_start_s) & (t <= plot_end_s)

    plt.figure(figsize=(12, 6))
    plt.plot(t[idx], x[idx] * 1e6, label="Raw signal")
    plt.plot(t[idx], y_fir[idx] * 1e6, label=f"Causal FIR {chosen_fir_numtaps} taps")
    plt.plot(t[idx], y_iir[idx] * 1e6, label=f"Causal IIR Butterworth order {chosen_iir_order}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [µV]")
    plt.title("Signal before and after filtering, zoomed segment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_signal_zoomed.png"), dpi=200)
    plt.close()


    # ============================================================
    # Plot 7: Latency comparison bar plot
    # ============================================================

    labels = [
        "FIR full latency",
        "IIR full latency\nmedian GD",
        "IIR full latency\nmax GD",
    ]

    latencies = [
        fir_full_latency_ms,
        iir_full_latency_median_ms,
        iir_full_latency_max_ms,
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, latencies)
    plt.ylabel("Latency [ms]")
    plt.title("Estimated full filtering latency")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_latency_comparison.png"), dpi=200)
    plt.close()


    # ============================================================
    # Save text report
    # ============================================================

    report_path = os.path.join(output_dir, "latency_and_design_report.txt")

    with open(report_path, "w") as f:
        f.write("A1 Online Motor Imagery Filtering Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Sampling frequency: {fs} Hz\n")
        f.write(f"Band-pass range: {bp[0]}-{bp[1]} Hz\n")
        f.write(f"Block size: {block_size} samples\n")
        f.write(f"Buffering latency: {buffer_latency_ms:.3f} ms\n\n")

        f.write("Candidate filter summaries:\n")
        f.write("-" * 60 + "\n")
        for s in summaries:
            f.write(
                f"{s['name']:<30} | "
                f"Ripple: {s['passband_ripple_db']:6.2f} dB | "
                f"Stop attenuation: {s['stopband_attenuation_db']:6.2f} dB | "
                f"Median GD: {s['median_group_delay_ms']:7.2f} ms | "
                f"Max GD: {s['max_group_delay_ms']:7.2f} ms\n"
            )

        f.write("\nChosen FIR filter:\n")
        f.write(f"Type: causal linear-phase FIR windowed-sinc band-pass\n")
        f.write(f"Window: Hamming\n")
        f.write(f"Number of taps: {chosen_fir_numtaps}\n")
        f.write(f"Cutoff frequencies: {bp[0]}-{bp[1]} Hz\n")
        f.write(f"Exact group delay: {fir_group_delay_ms_exact:.3f} ms\n")
        f.write(f"Processing time per block: {fir_proc_time_ms:.6f} ms\n")
        f.write(f"Full latency estimate: {fir_full_latency_ms:.3f} ms\n\n")

        f.write("Chosen IIR filter:\n")
        f.write(f"Type: causal Butterworth band-pass\n")
        f.write(f"Order: {chosen_iir_order}\n")
        f.write(f"Implementation: second-order sections, SOS\n")
        f.write(f"Cutoff frequencies: {bp[0]}-{bp[1]} Hz\n")
        f.write(f"Median passband group delay: {iir_group_delay_median_ms:.3f} ms\n")
        f.write(f"Max passband group delay: {iir_group_delay_max_ms:.3f} ms\n")
        f.write(f"Processing time per block: {iir_proc_time_ms:.6f} ms\n")
        f.write(f"Full latency estimate, median GD: {iir_full_latency_median_ms:.3f} ms\n")
        f.write(f"Full latency estimate, max GD: {iir_full_latency_max_ms:.3f} ms\n\n")

        f.write("Final design choice:\n")
        f.write(
            "For online motor imagery filtering, the causal IIR Butterworth filter "
            "is selected because it provides a useful 8-30 Hz band-pass response "
            "with much lower latency than the long FIR filter. Although the FIR "
            "filter has linear phase and therefore preserves waveform shape better, "
            "its group delay is too large for online processing. In motor imagery "
            "BCI, the main feature is usually band power in the mu and beta range, "
            "so the nonlinear phase of the IIR filter is acceptable. The IIR filter "
            "should be implemented using second-order sections for numerical stability.\n"
        )

    print(f"\nPlots and report saved to: {output_dir}")