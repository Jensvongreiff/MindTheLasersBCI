
import numpy as np
import copy
import scipy.signal as sig
import mne
from mne.preprocessing import ICA

BCI2A_CHANNEL_NAMES = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2",
    "POz",
]

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

def preprocess_bci2a_dataset(
    dataset: dict,
    ch_names=None,
    apply_filter: bool = True,
    apply_ica: bool = True,
    iir_order: int = 4,
    iir_band: list = [8, 30],
    ica_components: int = 20,
    ica_exclude=None,
):
    """
    Apply preprocess_epochs() to every subject/session in a loaded BCI2a dataset.

    Parameters
    ----------
    dataset : dict
        Output of load_bci2a_dataset(...).

    ch_names : list or None
        Channel names passed to preprocess_epochs.
        If None, defaults to standard BCI2a 22-channel names.

    apply_filter : bool
        Passed to preprocess_epochs.

    apply_ica : bool
        Passed to preprocess_epochs.

    iir_order : int
        Passed to preprocess_epochs.

    iir_band : list
        Passed to preprocess_epochs.

    ica_components : int
        Passed to preprocess_epochs.

    ica_exclude : list or None
        Passed to preprocess_epochs.

    Returns
    -------
    preprocessed_dataset : dict
        Same structure as original dataset, but with preprocessed X.

    preprocessing_infos : dict
        Matching nested dictionary containing preprocessing info per subject/session.
        Example:
        preprocessing_infos["A01"]["T"]
    """

    if ch_names is None:
        ch_names = BCI2A_CHANNEL_NAMES

    preprocessed_dataset = {}
    preprocessing_infos = {}

    for subject_id, subject_data in dataset.items():
        preprocessed_dataset[subject_id] = {}
        preprocessing_infos[subject_id] = {}

        for session, session_data in subject_data.items():
            X = session_data["X"]
            y = session_data["y"]
            fs = session_data["fs"]
            info = session_data["info"]

            n_channels = X.shape[1]
            session_ch_names = ch_names[:n_channels]

            X_preprocessed, preprocessing_info = preprocess_epochs(
                X=X,
                fs=fs,
                ch_names=session_ch_names,
                apply_filter=apply_filter,
                apply_ica=apply_ica,
                iir_order=iir_order,
                iir_band=iir_band,
                ica_components=min(ica_components, n_channels),
                ica_exclude=ica_exclude,
            )

            # Keep the exact same dataset structure
            preprocessed_dataset[subject_id][session] = {
                "X": X_preprocessed,
                "y": y.copy() if isinstance(y, np.ndarray) else copy.deepcopy(y),
                "fs": fs,
                "info": copy.deepcopy(info),
            }

            # Store preprocessing info separately so the dataset structure stays clean
            preprocessing_infos[subject_id][session] = preprocessing_info

            print(
                f"{subject_id}{session}: "
                f"X {X.shape} -> {X_preprocessed.shape}"
            )

    return preprocessed_dataset, preprocessing_infos
