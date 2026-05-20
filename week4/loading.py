from pathlib import Path
import numpy as np
import scipy.io as sio


def get_bci2a_mat_path(subject, session):
    """
    Construct path to BCI Competition IV 2a .mat file.

    Implicitly uses data/bci2a_dataset/ directory structure.

    Examples
    --------
    get_bci2a_mat_path(1, "T") -> Path("data/bci2a_dataset/A01T.mat")
    get_bci2a_mat_path("A01", "T") -> Path("data/bci2a_dataset/A01T.mat")
    """
    session = session.upper()

    if session not in ["T", "E"]:
        raise ValueError("session must be 'T' or 'E'")

    # Handle both integer and string subject IDs
    if isinstance(subject, str):
        subject_id = f"{subject}{session}.mat"
    else:
        subject_id = f"A{int(subject):02d}{session}.mat"

    return Path("data/bci2a_dataset") / subject_id



def load_bci2a_left_right(
    mat_path,
    tmin=0.0,
    tmax=4.0,
    keep_eeg_channels_only=True,
    reject_artifacts=False,
):
    """
    Load BCI Competition IV Dataset 2a .mat file for left/right hand MI.

    Parameters
    ----------
    mat_path : str or Path
        Path to file, e.g. "data/A01T.mat".

    tmin : float
        Start time of epoch relative to cue/trial marker, in seconds.

    tmax : float
        End time of epoch relative to cue/trial marker, in seconds.

    keep_eeg_channels_only : bool
        If True, keep only first 22 EEG channels and drop last 3 EOG channels.

    reject_artifacts : bool
        If True, remove trials marked as artifacts.

    Returns
    -------
    X : np.ndarray
        EEG epochs, shape (n_trials, n_channels, n_times).

    y : np.ndarray
        Labels, shape (n_trials,).
        0 = left hand
        1 = right hand

    fs : int
        Sampling frequency.

    info : dict
        Small metadata dictionary.
    """

    mat_path = Path(mat_path)
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    runs = np.ravel(mat["data"])

    fs = int(runs[0].fs)

    start_offset = int(round(tmin * fs))
    stop_offset = int(round(tmax * fs))
    n_times = stop_offset - start_offset

    X_epochs = []
    y_epochs = []
    run_ids = []
    artifact_flags = []

    for run_idx, run in enumerate(runs):
        trial_positions = np.asarray(run.trial, dtype=int).ravel()
        labels = np.asarray(run.y, dtype=int).ravel()

        # Skip non-MI runs such as calibration/rest/EOG runs
        if len(trial_positions) == 0:
            continue

        X_run = np.asarray(run.X, dtype=float)

        # Your file has X shape: samples x channels
        # Convert to channels x samples
        X_run = X_run.T

        # Keep only 22 EEG channels, drop 3 EOG channels
        if keep_eeg_channels_only:
            X_run = X_run[:22, :]

        n_channels, n_samples = X_run.shape

        artifacts = np.asarray(run.artifacts).astype(bool).ravel()

        # MATLAB sample indices are often 1-based.
        # If so, convert them to Python 0-based indices.
        if trial_positions.min() == 1:
            trial_positions = trial_positions - 1

        for trial_start, label, is_artifact in zip(
            trial_positions,
            labels,
            artifacts,
        ):
            # In this dataset:
            # 1 = left hand
            # 2 = right hand
            # 3 = feet
            # 4 = tongue
            if label not in [1, 2]:
                continue

            if reject_artifacts and is_artifact:
                continue

            start = trial_start + start_offset
            stop = trial_start + stop_offset

            # Skip incomplete epochs near recording boundaries
            if start < 0 or stop > n_samples:
                continue

            epoch = X_run[:, start:stop]

            if epoch.shape != (n_channels, n_times):
                continue

            X_epochs.append(epoch)

            if label == 1:
                y_epochs.append(0)  # left hand
            elif label == 2:
                y_epochs.append(1)  # right hand

            run_ids.append(run_idx)
            artifact_flags.append(bool(is_artifact))

    X = np.stack(X_epochs, axis=0)
    y = np.asarray(y_epochs, dtype=int)

    info = {
        "file": str(mat_path),
        "fs": fs,
        "tmin": tmin,
        "tmax": tmax,
        "n_trials": len(y),
        "n_channels": X.shape[1],
        "n_times": X.shape[2],
        "class_mapping": {
            0: "left hand",
            1: "right hand",
        },
        "run_ids": np.asarray(run_ids),
        "artifact_flags": np.asarray(artifact_flags),
    }

    return X, y, fs, info



def load_bci2a_dataset(
    data_dir,
    subjects=range(1, 10),
    tmin=0.0,
    tmax=4.0,
    keep_eeg_channels_only=True,
    reject_artifacts=False,
):
    """
    Load BCI Competition IV Dataset 2a for all subjects.

    Expected files:
        A01T.mat, A01E.mat, ..., A09T.mat, A09E.mat

    Returns
    -------
    dataset : dict
        dataset["A01"]["T"]["X"]
        dataset["A01"]["T"]["y"]
        dataset["A01"]["T"]["fs"]
        dataset["A01"]["T"]["info"]

        dataset["A01"]["E"]["X"]
        dataset["A01"]["E"]["y"]
        ...
    """

    data_dir = Path(data_dir)
    dataset = {}

    for subject in subjects:
        subject_id = f"A{subject:02d}"
        dataset[subject_id] = {}

        for session in ["T", "E"]:
            mat_path = data_dir / f"{subject_id}{session}.mat"

            if not mat_path.exists():
                raise FileNotFoundError(f"Could not find file: {mat_path}")

            X, y, fs, info = load_bci2a_left_right(
                mat_path,
                tmin=tmin,
                tmax=tmax,
                keep_eeg_channels_only=keep_eeg_channels_only,
                reject_artifacts=reject_artifacts,
            )

            dataset[subject_id][session] = {
                "X": X,
                "y": y,
                "fs": fs,
                "info": info,
            }

            print(
                f"{subject_id}{session}: "
                f"X={X.shape}, y={y.shape}, "
                f"classes={np.bincount(y)}"
            )

    return dataset




def load_bci2a_rest_runs(
    mat_path,
    keep_eeg_channels_only=True,
):
    """
    Load the runs without trial labels from a BCI 2a .mat file.

    These are useful for estimating subject-specific IAF.

    Returns
    -------
    rest_runs : np.ndarray
        Shape (n_channels, n_samples).

    fs : int
        Sampling frequency.

    info : dict
        Metadata.
    """

    mat_path = Path(mat_path)
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    runs = np.ravel(mat["data"])
    fs = int(runs[0].fs)

    rest_runs = []
    run_ids = []

    for run_idx, run in enumerate(runs):
        trial_positions = np.asarray(run.trial).ravel()

        # Keep only runs without trials
        if len(trial_positions) != 0:
            continue

        X_run = np.asarray(run.X, dtype=float)

        # Your file has shape: samples × channels
        # Convert to: channels × samples
        X_run = X_run.T

        # Drop EOG channels, keep first 22 EEG channels
        if keep_eeg_channels_only:
            X_run = X_run[:22, :]

        rest_runs.append(X_run)
        run_ids.append(run_idx)

    info = {
        "file": str(mat_path),
        "fs": fs,
        "run_ids": run_ids,
        "n_rest_runs": len(rest_runs),
        "shapes": [x.shape for x in rest_runs],
    }

    X_rest = np.concatenate(rest_runs, axis=1)

    return X_rest, fs, info


if __name__ == "__main__":
    dataset = load_bci2a_dataset("/home/dani/Documents/TUM/3.Semester/BCI/practical-ss26-team4/data")