
import numpy as np
import scipy.signal as sig
import mne
from mne.preprocessing import ICA


def filter_causal_iir_epochs(
    X: np.ndarray,
    fs: float,
    order: int = 4,
    Wn: list = [8, 30],
) -> np.ndarray:
    """
    Apply causal Butterworth IIR band-pass filter to epoched EEG data.

    Parameters
    ----------
    X : np.ndarray
        Shape: (n_trials, n_channels, n_times)

    fs : float
        Sampling frequency.

    order : int
        Butterworth filter order.

    Wn : list
        Band-pass cutoff frequencies, e.g. [8, 30].

    Returns
    -------
    X_filt : np.ndarray
        Filtered data with same shape as X.
    """
    assert X.ndim == 3, f"Expected X shape (n_trials, n_channels, n_times), got {X.shape}"

    sos = sig.butter(
        N=order,
        Wn=Wn,
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    X_filt = sig.sosfilt(
        sos,
        X,
        axis=2,  # time axis
    )

    return X_filt


def apply_ica_epochs(
    X: np.ndarray,
    fs: float,
    ch_names=None,
    n_components: int = 20,
    exclude=None,
    random_state: int = 97,
    max_iter="auto",
    method="fastica",
):
    """
    Fit ICA on epoched EEG data and optionally remove selected components.

    Parameters
    ----------
    X : np.ndarray
        Shape: (n_trials, n_channels, n_times)

    fs : float
        Sampling frequency.

    ch_names : list or None
        EEG channel names. If None, generic names are used.

    n_components : int
        Number of ICA components.

    exclude : list or None
        ICA components to remove.
        Example: exclude=[0, 2]
        If None or empty, ICA is fitted but no components are removed.

    Returns
    -------
    X_ica : np.ndarray
        ICA-processed data, shape (n_trials, n_channels, n_times)

    ica : mne.preprocessing.ICA
        Fitted ICA object.

    epochs_ica : mne.EpochsArray
        MNE epochs object after ICA application.
    """

    assert X.ndim == 3, f"Expected X shape (n_trials, n_channels, n_times), got {X.shape}"
    n_trials, n_channels, n_times = X.shape

    if ch_names is None:
        ch_names = [f"EEG{i:02d}" for i in range(n_channels)]

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=fs,
        ch_types=["eeg"] * n_channels,
    )

    epochs = mne.EpochsArray(
        X,
        info,
        verbose=False,
    )

    n_components = min(n_components, n_channels)

    ica = ICA(
        n_components=n_components,
        random_state=random_state,
        max_iter=max_iter,
        method=method,
    )

    ica.fit(epochs)

    epochs_ica = epochs.copy()

    if exclude is not None:
        ica.exclude = exclude
        ica.apply(epochs_ica)

    X_ica = epochs_ica.get_data()

    return X_ica, ica, epochs_ica


def preprocess_epochs(
    X: np.ndarray,
    fs: float,
    ch_names=None,
    apply_filter: bool = True,
    apply_ica: bool = True,
    iir_order: int = 4,
    iir_band: list = [8, 30],
    ica_components: int = 20,
    ica_exclude=None,
):
    """
    Constant preprocessing pipeline for epoched BCI 2a data.

    Pipeline:
        X
        → causal IIR band-pass
        → ICA fitting/application
        → preprocessed X

    Parameters
    ----------
    X : np.ndarray
        Shape: (n_trials, n_channels, n_times)

    fs : float
        Sampling frequency.

    ch_names : list or None
        Channel names.

    apply_filter : bool
        Whether to apply causal IIR filtering.

    apply_ica : bool
        Whether to fit/apply ICA.

    iir_order : int
        Butterworth order.

    iir_band : list
        Band-pass range, e.g. [8, 30].

    ica_components : int
        Number of ICA components.

    ica_exclude : list or None
        ICA components to remove.

    Returns
    -------
    X_preprocessed : np.ndarray
        Preprocessed EEG, shape (n_trials, n_channels, n_times)

    preprocessing_info : dict
        Dictionary containing filter parameters and ICA object.
    """

    X_proc = X.copy()

    preprocessing_info = {
        "filter_applied": apply_filter,
        "ica_applied": apply_ica,
        "iir_order": iir_order,
        "iir_band": iir_band,
        "ica_exclude": ica_exclude,
        "ica": None,
    }

    if apply_filter:
        X_proc = filter_causal_iir_epochs(
            X_proc,
            fs=fs,
            order=iir_order,
            Wn=iir_band,
        )

    if apply_ica:
        X_proc, ica, epochs_ica = apply_ica_epochs(
            X_proc,
            fs=fs,
            ch_names=ch_names,
            n_components=ica_components,
            exclude=ica_exclude,
        )

        preprocessing_info["ica"] = ica
        preprocessing_info["epochs_ica"] = epochs_ica

    return X_proc, preprocessing_info