# Please give me your results in a dictionary like this

# features[subject][method]["T"]["X"]  # session 1 features
# features[subject][method]["T"]["y"]

# features[subject][method]["E"]["X"]  # session 2 features
# features[subject][method]["E"]["y"]

# Example: X_train = features["A01"]["CSP"]["T"]["X"]

# Give the ability to save and load the features to avoid recomputing them every time
from mne.decoding import CSP
from mne.time_frequency import tfr_array_morlet
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pathlib import Path
import pickle
from pyriemann.tangentspace import TangentSpace


from loading import get_bci2a_mat_path, load_bci2a_dataset, load_bci2a_rest_runs
from preprocessing import BCI2A_CHANNEL_NAMES, preprocess_bci2a_dataset
# X = epochs.get_data()        # shape: (n_epochs, n_channels, n_times)
# y = epochs.events[:, 2]      # integer label per epoch


MOTOR_CHANNELS = ["C3", "Cz", "C4"]


def get_channel_indices(ch_names, wanted_channels):
    return [ch_names.index(ch) for ch in wanted_channels]


def bandpass_epochs(X, sfreq, band, order=4):
    """
    X shape: n_trials, n_channels, n_times
    """
    low, high = band
    nyq = sfreq / 2

    if low <= 0:
        raise ValueError("Low cutoff must be > 0 Hz.")
    if high >= nyq:
        raise ValueError(f"High cutoff must be below Nyquist frequency: {nyq} Hz.")

    sos = butter(
        order,
        [low / nyq, high / nyq],
        btype="bandpass",
        output="sos",
    )

    return sosfiltfilt(sos, X, axis=-1)


def logvar(X, eps=1e-12):
    """
    X shape: n_trials, n_channels, n_times
    returns: n_trials, n_channels
    """
    return np.log(np.var(X, axis=-1) + eps)


def estimate_iaf_from_rest(
    X_rest,
    sfreq,
    ch_indices,
    fmin=7.0,
    fmax=13.0,
):
    """
    Estimate individual alpha frequency from resting EEG.

    X_rest shape: n_channels, n_samples
    """
    X_rest_motor = X_rest[ch_indices, :]

    freqs, psd = welch(
        X_rest_motor,
        fs=sfreq,
        nperseg=int(sfreq * 2),
        axis=-1,
    )

    psd_mean = psd.mean(axis=0)

    alpha_mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(alpha_mask):
        raise ValueError("No frequency bins found in alpha search range.")

    iaf = freqs[alpha_mask][np.argmax(psd_mean[alpha_mask])]

    return float(iaf)

def create_bci2a_mne_info(info_dict):
    n_channels = info_dict["n_channels"]
    sfreq = info_dict["fs"]

    ch_names = BCI2A_CHANNEL_NAMES[:n_channels]

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=["eeg"] * n_channels,
    )

    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")

    return info

def extract_CSP(dataset, subject, train_or_eval, plot: bool = True):
    """Extract CSP features for a given subject and session (train or eval).
    Parameters
    ----------
    subject : str
    train_or_eval : bool, True for training session, False for evaluation session
    plot : bool, whether to show CSP pattern/filter plots
    """

    # Fit CSP on epochs and plot components
    if subject not in dataset:
        raise ValueError(f"Subject {subject} not found in dataset.")
    if train_or_eval:
        X = dataset[subject]["T"]["X"]
        y = dataset[subject]["T"]["y"]
        info = dataset[subject]["T"]["info"]
        info = create_bci2a_mne_info(info)
    else:
        X = dataset[subject]["E"]["X"]
        y = dataset[subject]["E"]["y"]
        info = dataset[subject]["E"]["info"]
        info = create_bci2a_mne_info(info)

    # Ensure labels are 0/1 for CSP
    y01 = y - y.min()

    # Configure CSP
    n_components = 4
    csp = CSP(
        n_components=n_components,
        reg='ledoit_wolf',   # robust covariance regularization
        log=True,
        norm_trace=False,
    )

    # Fit and transform
    X_csp = csp.fit_transform(X, y01)
    print("CSP-transformed shape:", X_csp.shape)

    # Optional: map numeric labels back to annotation names for legend titles
    try:
        inv_event = {v: k for k, v in event_id.items()}
        class_names = [inv_event.get(1, 'class 0'), inv_event.get(2, 'class 1')]
    except Exception:
        class_names = ['class 0', 'class 1']

    if plot:
        # Plot CSP patterns (topomaps)
        fig_patterns = csp.plot_patterns(
            info=info,
            ch_type='eeg',
            components=list(range(n_components)),
            show=True,
        )
        fig_patterns.suptitle("CSP Patterns")

        # Plot CSP filters (topomaps)
        fig_filters = csp.plot_filters(
            info=info,
            ch_type='eeg',
            components=list(range(n_components)),
            show=True,
        )
        fig_filters.suptitle("CSP Filters")

        # Keep all figures open until manually closed
        plt.show(block=True)

    # Quick variance by class (sanity check)
    import numpy as np
    var0 = X_csp[y01 == 0].var(axis=0)
    var1 = X_csp[y01 == 1].var(axis=0)
    print("Per-component variance, class0:", np.round(var0, 3))
    print("Per-component variance, class1:", np.round(var1, 3))

    return X_csp

def extract_bandpower_features(dataset, subject, train_or_eval, data_path=None):
    """
    Extract subject-specific mu and beta log-bandpower features.

    Parameters
    ----------
    subject : str
        Example: "A01"
    train_or_eval : bool
        True for training session, False for evaluation session.

    Returns
    -------
    features : np.ndarray
        Shape: n_trials, 6
        Columns:
        C3_mu, Cz_mu, C4_mu, C3_beta, Cz_beta, C4_beta

    y01 : np.ndarray
        Labels as 0/1.

    feature_info : dict
        Metadata about extracted features.
    """

    session = "T" if train_or_eval else "E"

    if subject not in dataset:
        raise ValueError(f"Subject {subject} not found in dataset.")

    X = dataset[subject][session]["X"]
    y = dataset[subject][session]["y"]
    info_dict = dataset[subject][session]["info"]
    sfreq = info_dict["fs"]

    # Ensure labels are 0/1
    y01 = y - y.min()

    # Select C3, Cz, C4
    motor_idx = get_channel_indices(BCI2A_CHANNEL_NAMES, MOTOR_CHANNELS)
    X_motor = X[:, motor_idx, :]

    # Load resting-state data from same session
    X_rest, fs_rest, rest_info = load_bci2a_rest_runs(data_path)

    if fs_rest != sfreq:
        raise ValueError(
            f"Rest fs={fs_rest}, trial fs={sfreq}. Sampling rates do not match."
        )

    # Estimate subject-specific IAF from rest data
    iaf = estimate_iaf_from_rest(
        X_rest=X_rest,
        sfreq=sfreq,
        ch_indices=motor_idx,
        fmin=7.0,
        fmax=13.0,
    )

    mu_band = (iaf - 2.0, iaf + 2.0)
    beta_band = (13.0, 30.0)

    print(f"Estimated IAF: {iaf:.2f} Hz")
    print(f"Mu band: {mu_band[0]:.2f}-{mu_band[1]:.2f} Hz")
    print(f"Beta band: {beta_band[0]:.2f}-{beta_band[1]:.2f} Hz")

    # Bandpass into mu and beta bands
    X_mu = bandpass_epochs(X_motor, sfreq, mu_band)
    X_beta = bandpass_epochs(X_motor, sfreq, beta_band)

    # Log-variance features
    feat_mu = logvar(X_mu)
    feat_beta = logvar(X_beta)

    # Final feature matrix: n_trials x 6
    features = np.concatenate([feat_mu, feat_beta], axis=1)

    feature_names = [
        "C3_mu", "Cz_mu", "C4_mu",
        "C3_beta", "Cz_beta", "C4_beta",
    ]

    feature_info = {
        "subject": subject,
        "session": session,
        "sfreq": sfreq,
        "iaf": iaf,
        "mu_band": mu_band,
        "beta_band": beta_band,
        "channels": MOTOR_CHANNELS,
        "feature_names": feature_names,
        "n_trials": features.shape[0],
        "n_features": features.shape[1],
    }

    print("Feature shape:", features.shape)

    return features, y01, feature_info


def extract_morlet_wavelet_features(dataset, subject, train_or_eval, return_unflattened=False):
    """
    Extract Morlet wavelet log-amplitude features for BCI2a left/right MI.

    Assignment specification:
    - Complex Morlet convolution at 8, 10, 12, 20, 24 Hz
    - Use log-amplitude of analytic signal
    - Time window: 1.5 to 4.0 seconds post-cue
    - Channels: C3, C4
    - Flatten into one feature vector per trial

    Parameters
    ----------
    subject : str
        Example: "A01"
    train_or_eval : bool
        True for training session, False for evaluation session.

    Returns
    -------
    X_wavelet : np.ndarray
        Shape: n_trials x n_features

    y01 : np.ndarray
        Labels as 0/1.

    feature_info : dict
        Metadata about extracted features.
    """

    session = "T" if train_or_eval else "E"

    if subject not in dataset:
        raise ValueError(f"Subject {subject} not found in dataset.")

    X = dataset[subject][session]["X"]
    y = dataset[subject][session]["y"]
    info_dict = dataset[subject][session]["info"]

    sfreq = info_dict["fs"]
    tmin_epoch = info_dict["tmin"]

    # Ensure labels are 0/1
    y01 = y - y.min()

    # Assignment-specific channels
    wanted_channels = ["C3", "C4"]
    ch_indices = get_channel_indices(BCI2A_CHANNEL_NAMES, wanted_channels)

    X_sel = X[:, ch_indices, :]

    # Assignment-specific frequencies
    freqs = np.array([8, 10, 12, 20, 24], dtype=float)

    # Reasonable Morlet width.
    # You can tune this, but freqs / 2 is a common practical choice.
    n_cycles = freqs / 2.0

    # Complex Morlet transform
    # Output shape: n_trials, n_channels, n_freqs, n_times
    complex_tfr = tfr_array_morlet(
        X_sel,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="complex",
        use_fft=True,
        n_jobs=None,
    )

    # Analytic amplitude
    amplitude = np.abs(complex_tfr)

    # Log-amplitude
    log_amplitude = np.log(amplitude + 1e-12)

    # Build time vector for the epochs
    n_times = X.shape[-1]
    times = tmin_epoch + np.arange(n_times) / sfreq

    # Assignment-specific post-cue window
    time_window = (1.5, 4.0)
    t_mask = (times >= time_window[0]) & (times <= time_window[1])

    if not np.any(t_mask):
        raise ValueError(
            f"No samples found in time window {time_window}. "
            f"Epoch time range is {times[0]:.3f} to {times[-1]:.3f} s."
        )

    log_amp_window = log_amplitude[:, :, :, t_mask]
    times_window = times[t_mask]

    # Flatten channels x freqs x times into one vector per trial
    X_wavelet = log_amp_window.reshape(log_amp_window.shape[0], -1)

    feature_info = {
        "subject": subject,
        "session": session,
        "sfreq": sfreq,
        "channels": wanted_channels,
        "frequencies": freqs.tolist(),
        "n_cycles": n_cycles.tolist(),
        "time_window": time_window,
        "times_window": times_window,
        "feature_shape_before_flattening": log_amp_window.shape,
        "n_trials": X_wavelet.shape[0],
        "n_features": X_wavelet.shape[1],
    }

    print("Wavelet feature shape:", X_wavelet.shape)
    print("Before flattening:", log_amp_window.shape)

    if return_unflattened:
        return X_wavelet, y01, feature_info, log_amp_window

    return X_wavelet, y01, feature_info

def plot_morlet_feature_validation(log_amp_window, y01, feature_info):
    """
    Plot class-averaged Morlet log-amplitude features.

    log_amp_window shape:
    n_trials, n_channels, n_freqs, n_times
    """

    channels = feature_info["channels"]
    freqs = np.array(feature_info["frequencies"])
    times = feature_info["times_window"]

    class_names = {
        0: "left hand",
        1: "right hand",
    }

    for ch_idx, ch_name in enumerate(channels):
        for class_id in [0, 1]:
            class_data = log_amp_window[y01 == class_id, ch_idx, :, :]

            if class_data.shape[0] == 0:
                continue

            avg_log_amp = class_data.mean(axis=0)

            plt.figure(figsize=(9, 4))
            plt.imshow(
                avg_log_amp,
                aspect="auto",
                origin="lower",
                extent=[times[0], times[-1], freqs[0], freqs[-1]],
            )
            plt.colorbar(label="Log-amplitude")
            plt.xlabel("Time after cue [s]")
            plt.ylabel("Frequency [Hz]")
            plt.title(f"{ch_name} Morlet log-amplitude - {class_names[class_id]}")
            plt.yticks(freqs)
            plt.tight_layout()

    plt.show(block=True)

def plot_morlet_left_right_difference(log_amp_window, y01, feature_info):
    channels = feature_info["channels"]
    freqs = np.array(feature_info["frequencies"])
    times = feature_info["times_window"]

    for ch_idx, ch_name in enumerate(channels):
        left_avg = log_amp_window[y01 == 0, ch_idx, :, :].mean(axis=0)
        right_avg = log_amp_window[y01 == 1, ch_idx, :, :].mean(axis=0)

        diff = left_avg - right_avg

        plt.figure(figsize=(9, 4))
        plt.imshow(
            diff,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
        )
        plt.colorbar(label="Left - right log-amplitude")
        plt.xlabel("Time after cue [s]")
        plt.ylabel("Frequency [Hz]")
        plt.title(f"{ch_name}: Left minus right Morlet features")
        plt.yticks(freqs)
        plt.tight_layout()

    plt.show(block=True)

def extract_riemannian_mdm_features(dataset, subject, train_or_eval):
    """
    Extract covariance matrices for Riemannian MDM.

    Method 4:
    - Bandpass in mu+beta range
    - Use all 22 EEG channels
    - Compute covariance matrix per trial
    - Use with Riemannian MDM classifier
    """

    session = "T" if train_or_eval else "E"

    if subject not in dataset:
        raise ValueError(f"Subject {subject} not found in dataset.")

    X = dataset[subject][session]["X"]
    y = dataset[subject][session]["y"]
    info_dict = dataset[subject][session]["info"]

    sfreq = info_dict["fs"]

    # Ensure labels are 0/1
    y01 = y - y.min()

    # Use all 22 EEG channels
    X_eeg = X[:, :22, :]

    # Same broad mu+beta band as CSP
    mu_beta_band = (8.0, 30.0)
    X_filt = bandpass_epochs(X_eeg, sfreq, mu_beta_band)

    # Compute one covariance matrix per trial
    cov_estimator = Covariances(estimator="oas")
    cov_matrices = cov_estimator.fit_transform(X_filt)

    feature_info = {
        "subject": subject,
        "session": session,
        "sfreq": sfreq,
        "band": mu_beta_band,
        "channels": "all 22 EEG channels",
        "method": "Riemannian MDM on covariance matrices",
        "cov_shape": cov_matrices.shape,
    }

    print("Covariance matrix shape:", cov_matrices.shape)

    ts = TangentSpace(metric="riemann")
    features = ts.fit_transform(cov_matrices)

    print("Tangent-space feature shape:", features.shape)

    return features, y01, feature_info

   

def extract_all_features_depracated(subject, train_or_eval, plot: bool = False):
    data_path = get_bci2a_mat_path(subject, "T" if train_or_eval else "E")
    dataset = load_bci2a_dataset(data_path)
    dataset, preprocessing_infos = preprocess_bci2a_dataset(
        dataset,
        apply_filter=True,
        apply_ica=True,
        iir_order=4,
        iir_band=[8, 30],
        ica_components=20,
        ica_exclude=None,
    )

    # Train
    X_csp_train, y_csp_train = extract_CSP(dataset, subject, train_or_eval=True)
    X_bp_train, y_bp_train, bp_feat_info_t = extract_bandpower_features(dataset, subject, train_or_eval=True, data_path=data_path)
    X_wavelet, y, wavelet_info, log_amp_window = extract_morlet_wavelet_features(
        dataset,
        subject="A01",
        train_or_eval=True,
        return_unflattened=True,
    )
    plot_morlet_feature_validation(
        log_amp_window=log_amp_window,
        y01=y,
        feature_info=wavelet_info,
    )
    plot_morlet_left_right_difference(
        log_amp_window=log_amp_window,
        y01=y,
        feature_info=wavelet_info,
    )
    X_cov_train, y_train, info_train = extract_riemannian_mdm_features(dataset, "A01", True)


    train_or_eval = False  # Change to False to extract eval features
    data_path = get_bci2a_mat_path(subject, "T" if train_or_eval else "E")
    dataset = load_bci2a_dataset(data_path)

    # Eval
    X_csp_eval, y_csp_eval = extract_CSP(dataset, subject, train_or_eval=False)
    X_bp_eval, y_bp_eval, bp_feat_info_e = extract_bandpower_features(dataset, subject, train_or_eval=False, data_path=data_path)
    X_wavelet, y, wavelet_info, log_amp_window = extract_morlet_wavelet_features(
        dataset,
        subject="A01",
        train_or_eval=False,
        return_unflattened=True,
    )
    plot_morlet_feature_validation(
        log_amp_window=log_amp_window,
        y01=y,
        feature_info=wavelet_info,
    )
    plot_morlet_left_right_difference(
        log_amp_window=log_amp_window,
        y01=y,
        feature_info=wavelet_info,
    )
    X_cov_eval, y_eval, info_eval = extract_riemannian_mdm_features(dataset, "A01", False)

    clf = MDM(metric="riemann")

    clf.fit(X_cov_train, y_train)
    y_pred = clf.predict(X_cov_eval)

    acc = np.mean(y_pred == y_eval)
    print("MDM accuracy:", acc)

def extract_all_features(dataset, subject, plot: bool = False):
    """
    Extract all features for one subject and both sessions.

    Output structure:
        features[subject][method]["T"]["X"]
        features[subject][method]["T"]["y"]
        features[subject][method]["E"]["X"]
        features[subject][method]["E"]["y"]
    """

    subject_features = {
        "bandpower": {},
        "csp": {},
        "morlet": {},
        "riemannian_mdm": {},
    }

    for session, train_or_eval in [("T", True), ("E", False)]:

        # -------------------------
        # Method 1: Bandpower
        # -------------------------
        data_path = get_bci2a_mat_path(subject, session)  # Just to get the path format
        X_bp, y_bp, bp_info = extract_bandpower_features(
            dataset,
            subject,
            train_or_eval=train_or_eval,
            data_path=data_path,
        )

        subject_features["bandpower"][session] = {
            "X": X_bp,
            "y": y_bp,
            "info": bp_info,
        }

        # -------------------------
        # Method 2: CSP
        # -------------------------
        X_csp = extract_CSP(
            dataset,
            subject,
            train_or_eval=train_or_eval,
            plot=plot,
        )

        # Since your current CSP function returns only X_csp,
        # get labels directly from the dataset.
        y_csp = dataset[subject][session]["y"]
        y_csp = y_csp - y_csp.min()

        subject_features["csp"][session] = {
            "X": X_csp,
            "y": y_csp,
            "info": {
                "method": "CSP",
                "n_features": X_csp.shape[1],
                "session": session,
            },
        }

        # -------------------------
        # Method 3: Morlet wavelet
        # -------------------------
        X_wavelet, y_wavelet, wavelet_info, log_amp_window = extract_morlet_wavelet_features(
            dataset,
            subject=subject,
            train_or_eval=train_or_eval,
            return_unflattened=True,
        )

        subject_features["morlet"][session] = {
            "X": X_wavelet,
            "y": y_wavelet,
            "info": wavelet_info,
        }

        if plot:
            plot_morlet_feature_validation(
                log_amp_window=log_amp_window,
                y01=y_wavelet,
                feature_info=wavelet_info,
            )

            plot_morlet_left_right_difference(
                log_amp_window=log_amp_window,
                y01=y_wavelet,
                feature_info=wavelet_info,
            )

        # -------------------------
        # Method 4: Riemannian MDM
        # -------------------------
        X_cov, y_cov, cov_info = extract_riemannian_mdm_features(
            dataset,
            subject,
            train_or_eval=train_or_eval,
        )

        subject_features["riemannian_mdm"][session] = {
            "X": X_cov,
            "y": y_cov,
            "info": cov_info,
        }

    return subject_features

def output_features(plot: bool = False, output_dir: str = "features"):
    """
    Extract features for all BCI2a subjects and save them.

    Saves:
        features/all_features.pkl

    The saved object has structure:
        features[subject][method]["T"]["X"]
        features[subject][method]["T"]["y"]
        features[subject][method]["E"]["X"]
        features[subject][method]["E"]["y"]
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    subjects = [
        "A01", "A02", "A03", "A04", "A05",
        "A06", "A07", "A08", "A09",
    ]
    # subjects =  ["A01", "A05"]  # For quick testing, comment out to run all subjects

    features = {}

    data_path = Path("/home/dani/Documents/TUM/3.Semester/BCI/practical-ss26-team4/data") 

    dataset = load_bci2a_dataset(data_path)

    dataset, preprocessing_infos = preprocess_bci2a_dataset(
        dataset,
        apply_filter=True,
        apply_ica=True,
        iir_order=4,
        iir_band=[8, 30],
        ica_components=20,
        ica_exclude=None,
    )

    for subject in subjects:
        if subject not in dataset:
            raise ValueError(f"Subject {subject} not found in dataset.")
        
        print(f"\nExtracting features for {subject}")
        features[subject] = extract_all_features(
            dataset=dataset,
            subject=subject,
            plot=plot,
        )

    save_path = output_path / "all_features.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(features, f)

    print(f"\nSaved features to: {save_path}")

    return features

def load_features(features_path: str = "/home/dani/Documents/TUM/3.Semester/BCI/practical-ss26-team4/week4/features/all_features.pkl"):
    with open(features_path, "rb") as f:
        features = pickle.load(f)
    return features

def main():
    # Extract and save features for all subjects and both sessions
    # Plotting can be enabled for visual validation, but it will slow down the process significantly.
    # You need to manually close the plots for each subject/session to continue.
    # To change which subjects, or tweak preprocessing, you can modify the output_features() function.
    # features = output_features(plot=False)

    # For demonstration, load the saved features and print their shapes
    # features = load_features("features/all_features.pkl")

    # subjects = [
    #     "A01", "A02", "A03", "A04", "A05",
    #     "A06", "A07", "A08", "A09",
    # ]
    # methods = ["bandpower", "csp", "morlet", "riemannian_mdm"]
    # for subject in subjects:
    #     print(f"\nSubject: {subject}")
    #     for method in methods:
    #         print(f"  Method: {method}")
    #         X_csp_train = features[subject][method]["T"]["X"]
    #         print("train features shape:", X_csp_train.shape)
    #         y_csp_train = features[subject][method]["T"]["y"]
    #         print("train labels shape:", y_csp_train.shape)

    #         X_csp_eval = features[subject][method]["E"]["X"]
    #         y_csp_eval = features[subject][method]["E"]["y"]
    #         print("eval features shape:", X_csp_eval.shape)
    #         print("eval labels shape:", y_csp_eval.shape)

    output_features(False, "/home/dani/Documents/TUM/3.Semester/BCI/practical-ss26-team4/week4/features")


if __name__ == "__main__":
    main()