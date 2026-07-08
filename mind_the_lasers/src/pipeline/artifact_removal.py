from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union
import pickle
import os
from pathlib import Path
import numpy as np
from .signal import EEGWindow


class BaseArtifactRemoval(ABC):
    """Base class for artifact removal."""
    
    @abstractmethod
    def fit(self, offline_data: np.ndarray):
        """Called only during the calibration phase."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply fitted artifact removal to offline training data."""
        pass

    @abstractmethod
    def extract(self, window: EEGWindow) -> EEGWindow:
        """Called online during the game loop."""
        pass

class SpatialFilterICA(BaseArtifactRemoval):
    """
    MNE-based ICA artifact-removal block.

    Calibration/offline phase:
        fit(raw, bad_components=[...])

    Online phase:
        transform(window)

    The fitted ICA is applied spatially to each incoming EEGWindow. This keeps
    the online contract simple: EEGWindow -> EEGWindow.
    """

    def __init__(
        self,
        model_path: str = "",
        n_components: Optional[Union[int, float]] = None,
        random_state: int = 97,
        max_iter: Union[int, str] = "auto",
        method: str = "infomax",
        channel_names: Optional[Sequence[str]] = None,
        ch_types: Union[str, Sequence[str]] = "eeg",
    ):
        self.model_path = model_path
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.method = method
        self.channel_names = list(channel_names) if channel_names is not None else None
        self.ch_types = ch_types

        self.ica = None
        self.info = None
        self.bad_components: list[int] = []

        self.component_labels: list[str] = []
        self.component_probabilities = np.array([], dtype=float)

        if os.path.exists(self.model_path):
            self.load_pickle(self.model_path)

    @staticmethod
    def _import_mne():
        try:
            import mne
            from mne.preprocessing import ICA
        except ImportError as exc:
            raise ImportError(
                "SpatialFilterICA requires MNE. Install it with `pip install mne`."
            ) from exc
        return mne, ICA

    def _make_raw_from_array(
        self,
        data: np.ndarray,
        sampling_rate: float,
        channel_names: Optional[Sequence[str]] = None,
    ):
        """Create an MNE RawArray from channel x sample EEG data."""
        mne, _ = self._import_mne()

        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(
                f"Expected EEG data with shape (n_channels, n_samples), got {data.shape}."
            )

        n_channels = data.shape[0]
        if channel_names is None:
            channel_names = self.channel_names
        if channel_names is None:
            channel_names = [f"EEG{i + 1:03d}" for i in range(n_channels)]
        else:
            channel_names = list(channel_names)

        if len(channel_names) != n_channels:
            raise ValueError(
                f"Got {len(channel_names)} channel names for {n_channels} data channels."
            )

        info = mne.create_info(
            ch_names=channel_names,
            sfreq=float(sampling_rate),
            ch_types=self.ch_types,
        )
        return mne.io.RawArray(data, info, verbose=False)

    def _make_epochs_from_array(
        self,
        data: np.ndarray,
        sampling_rate: float,
        channel_names: Optional[Sequence[str]] = None,
    ):
        """
        Create an MNE EpochsArray from EEG data.

        Expected shape:
            (n_epochs, n_channels, n_samples)
        """
        mne, _ = self._import_mne()

        data = np.asarray(data, dtype=float)

        if data.ndim != 3:
            raise ValueError(
                "Expected epoched EEG data with shape "
                f"(n_epochs, n_channels, n_samples), got {data.shape}."
            )

        n_channels = data.shape[1]

        if channel_names is None:
            channel_names = self.channel_names

        if channel_names is None:
            channel_names = [
                f"EEG{i + 1:03d}"
                for i in range(n_channels)
            ]
        else:
            channel_names = list(channel_names)

        if len(channel_names) != n_channels:
            raise ValueError(
                f"Got {len(channel_names)} channel names for "
                f"{n_channels} data channels."
            )

        info = mne.create_info(
            ch_names=channel_names,
            sfreq=float(sampling_rate),
            ch_types=self.ch_types,
        )

        return mne.EpochsArray(
            data=data,
            info=info,
            tmin=0.0,
            baseline=None,
            verbose=False,
        )

    def _coerce_to_raw(
        self,
        offline_data,
        sampling_rate: Optional[float] = None,
        channel_names: Optional[Sequence[str]] = None,
    ):
        """
        Convert offline EEG data into an MNE Raw or Epochs object.

        Accepted inputs
        ---------------
        MNE Raw or Epochs:
            Returned directly.

        NumPy array:
            (n_channels, n_samples)
                Converted to MNE RawArray.

            (n_epochs, n_channels, n_samples)
                Converted to MNE EpochsArray.
        """
        if (
            hasattr(offline_data, "get_data")
            and hasattr(offline_data, "info")
        ):
            return offline_data

        if sampling_rate is None:
            raise ValueError(
                "When fitting ICA from a NumPy array, sampling_rate "
                "must be provided."
            )

        data = np.asarray(offline_data, dtype=float)

        if data.ndim == 2:
            return self._make_raw_from_array(
                data=data,
                sampling_rate=sampling_rate,
                channel_names=channel_names,
            )

        if data.ndim == 3:
            return self._make_epochs_from_array(
                data=data,
                sampling_rate=sampling_rate,
                channel_names=channel_names,
            )

        raise ValueError(
            "Expected offline EEG data with shape "
            "(n_channels, n_samples) or "
            "(n_epochs, n_channels, n_samples), "
            f"got {data.shape}."
        )

    def detect_bad_components(
        self,
        raw_for_fit,
        probability_threshold: float = 0.85,
        reject_labels: Optional[Sequence[str]] = None,
    ) -> list[int]:
        """
        Detect artifact ICA components automatically using ICLabel.

        The supplied raw_for_fit should be the same temporary data copy that
        was used to fit self.ica. This function does not modify the downstream
        EEG data and does not fit ICA again.
        """
        if self.ica is None:
            raise ValueError(
                "ICA must be fitted before bad components can be detected."
            )

        try:
            from mne_icalabel import label_components
        except ImportError as exc:
            raise ImportError(
                "Automatic ICA component detection requires mne-icalabel. "
                "Install it with `pip install mne-icalabel`."
            ) from exc

        if not 0.0 <= probability_threshold <= 1.0:
            raise ValueError(
                "probability_threshold must be between 0 and 1."
            )

        if reject_labels is None:
            reject_labels = (
                "eye blink",
                "heart beat",
                "line noise",
                "channel noise",
            )

        result = label_components(
            raw_for_fit,
            self.ica,
            method="iclabel",
        )

        labels = list(result["labels"])
        probabilities = np.asarray(
            result["y_pred_proba"],
            dtype=float,
        )

        rejected = set(reject_labels)

        detected_components = [
            index
            for index, (label, probability) in enumerate(
                zip(labels, probabilities)
            )
            if label in rejected
            and probability >= probability_threshold
        ]

        # Save results so they can be inspected after calibration.
        self.component_labels = labels
        self.component_probabilities = probabilities

        self.set_bad_components(detected_components)

        return detected_components

    def fit(
        self,
        offline_data,
        bad_components: Optional[Sequence[int]] = None,
        sampling_rate: Optional[float] = None,
        channel_names: Optional[Sequence[str]] = None,
    ) -> "SpatialFilterICA":
        """
        Fit ICA during calibration.

        Parameters
        ----------
        offline_data:
            Either an MNE Raw object or a NumPy array with shape
            (n_channels, n_samples). Must be the unfiltered EEG data.
        bad_components:
            Optional ICA component indices to remove during online `transform`.
            If omitted, artifact components are selected automatically using
            ICLabel.
        sampling_rate:
            Required only when `offline_data` is a NumPy array.
        channel_names:
            Optional channel names for NumPy input. For online transform, the
            incoming EEGWindow must use the same channel order.

        Returns
        -------
        self
            The fitted artifact-removal block.
        """

        raw_data = self._coerce_to_raw(offline_data, sampling_rate, channel_names)
        _, ICA = self._import_mne()

        n_channels = len(raw_data.ch_names)
        if isinstance(self.n_components, int) and self.n_components > n_channels:
            raise ValueError(
                f"n_components={self.n_components} cannot exceed the number of "
                f"channels ({n_channels})."
            )

        raw_for_fit = raw_data.copy()

        raw_for_fit.set_montage(
            "standard_1020",
            on_missing="raise",
        )

        # Prepare Raw signal for ICA Bads detection
        raw_for_fit.filter(
            l_freq=1.0,
            h_freq=100.0,
            method="iir",
            iir_params={
                "order": 4,
                "ftype": "butter",
            },
            phase="zero",
            verbose=False,
        )
        raw_for_fit.set_eeg_reference(
            ref_channels="average",
            projection=False,
            verbose=False,
        )


        self.ica = ICA(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
            method=self.method,
            fit_params={"extended": True}
            if self.method == "infomax"
            else None,
        )
        self.ica.fit(raw_for_fit, verbose=False)

        self.info = raw_data.info.copy()
        self.channel_names = list(raw_data.ch_names)
        if bad_components is None:
            self.detect_bad_components(raw_for_fit)
        else:
            self.set_bad_components(bad_components)


        # print("\nICLabel component results:")

        # for index, (label, probability) in enumerate(
        #     zip(
        #         self.component_labels,
        #         self.component_probabilities,
        #     )
        # ):
        #     print(
        #         f"ICA{index:03d}: "
        #         f"{label:<18} "
        #         f"confidence={probability:.3f} "
        #         f"excluded={index in self.bad_components}"
        #     )

        # self.ica.plot_properties(
        #     raw_for_fit,
        #     picks=self.bad_components,
        # )

        print("Excluded ICA components:", self.bad_components)

        self.save_pickle()

        return self

    def set_bad_components(self, bad_components: Sequence[int]) -> None:
        """Update the ICA components that should be removed online."""
        if self.ica is None:
            raise ValueError("ICA must be fitted before setting bad components.")
        self.bad_components = list(bad_components)
        self.ica.exclude = self.bad_components.copy()

    def transform(
        self,
        X: np.ndarray,
        sampling_rate: float,
    ) -> np.ndarray:
        """
        Apply the fitted ICA to offline EEG data used for model training.

        Parameters
        ----------
        X:
            EEG data with either shape:

            - (n_channels, n_samples) for continuous data
            - (n_epochs, n_channels, n_samples) for epoched training data

        sampling_rate:
            Sampling rate of X in Hz.

        Returns
        -------
        np.ndarray
            Artifact-removed EEG data with the same shape as X.

        Notes
        -----
        This method does not fit ICA again. It applies the same fitted ICA
        decomposition and bad-component exclusion list used by the online
        transform method.
        """
        if self.ica is None:
            try:
                self.load_pickle(self.model_path)
            except FileNotFoundError:
                raise ValueError(
                    "ICA has not been fitted with offline calibration data yet."
                )

        X = np.asarray(X, dtype=float)

        if X.ndim not in (2, 3):
            raise ValueError(
                "Expected X with shape (n_channels, n_samples) or "
                f"(n_epochs, n_channels, n_samples), got {X.shape}."
            )

        if sampling_rate <= 0:
            raise ValueError(
                f"sampling_rate must be positive, got {sampling_rate}."
            )

        expected_channels = len(self.channel_names or [])

        if expected_channels == 0:
            raise ValueError(
                "No channel information is available from ICA fitting."
            )

        channel_axis = 0 if X.ndim == 2 else 1
        actual_channels = X.shape[channel_axis]

        if actual_channels != expected_channels:
            raise ValueError(
                f"ICA was fitted with {expected_channels} channels, but X "
                f"contains {actual_channels} channels."
            )

        # Continuous data: channels × samples
        if X.ndim == 2:
            raw = self._make_raw_from_array(
                data=X,
                sampling_rate=sampling_rate,
                channel_names=self.channel_names,
            )

            cleaned_raw = self.ica.apply(
                raw.copy(),
                exclude=self.bad_components,
                verbose=False,
            )

            return cleaned_raw.get_data()

        # Epoched data: epochs × channels × samples
        mne, _ = self._import_mne()

        info = mne.create_info(
            ch_names=self.channel_names,
            sfreq=float(sampling_rate),
            ch_types=self.ch_types,
        )

        epochs = mne.EpochsArray(
            X,
            info,
            baseline=None,
            verbose=False,
        )

        cleaned_epochs = self.ica.apply(
            epochs.copy(),
            exclude=self.bad_components,
            verbose=False,
        )

        return cleaned_epochs.get_data()

    def extract(self, window: EEGWindow) -> EEGWindow:
        """Apply the fitted ICA to one online EEG window.
        
            Note: This can be optimized by using a matrix calculation instead of an MNE object
        """
        if self.ica is None:
            raise ValueError("ICA has not been fitted with offline calibration data yet.")

        data = np.asarray(window.data, dtype=float)
        expected_channels = len(self.channel_names or [])
        if expected_channels and data.shape[0] != expected_channels:
            raise ValueError(
                f"ICA was fitted with {expected_channels} channels, but the incoming "
                f"window has {data.shape[0]} channels."
            )

        raw_window = self._make_raw_from_array(
            data=data,
            sampling_rate=window.sampling_rate,
            channel_names=self.channel_names,
        )
        cleaned_raw = self.ica.apply(
            raw_window.copy(),
            exclude=self.bad_components,
            verbose=False,
        )

        return EEGWindow(
            data=cleaned_raw.get_data(),
            sampling_rate=window.sampling_rate,
            timestamp=window.timestamp,
            ground_truth=window.ground_truth,
            is_artifact_free=True,
        )

    def save_pickle(
        self
    ) -> None:
        """
        Save the complete fitted SpatialFilterICA object.
        """

        file_path = self.model_path

        if self.ica is None:
            raise ValueError(
                "ICA must be fitted before it can be saved."
            )

        file_path = Path(file_path)
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with file_path.open("wb") as file:
            pickle.dump(
                self,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load_pickle(
        self,
        file_path: Union[str, Path],
    ) -> "SpatialFilterICA":
        """
        Load a fitted SpatialFilterICA object and copy its state into self.

        After this method returns, the current instance can immediately be
        used for online transform calls. No additional fit call is required.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"ICA pickle file does not exist: {file_path}"
            )

        with file_path.open("rb") as file:
            loaded = pickle.load(file)

        if not isinstance(loaded, SpatialFilterICA):
            raise TypeError(
                "The pickle file does not contain a "
                "SpatialFilterICA instance."
            )

        if loaded.ica is None:
            raise ValueError(
                "The loaded SpatialFilterICA instance does not "
                "contain a fitted ICA decomposition."
            )

        # Copy constructor/configuration fields.
        self.n_components = loaded.n_components
        self.random_state = loaded.random_state
        self.max_iter = loaded.max_iter
        self.method = loaded.method
        self.channel_names = (
            list(loaded.channel_names)
            if loaded.channel_names is not None
            else None
        )
        self.ch_types = loaded.ch_types

        # Copy fitted ICA state.
        self.ica = loaded.ica
        self.info = loaded.info

        self.bad_components = list(
            loaded.bad_components
        )

        self.component_labels = list(
            loaded.component_labels
        )

        self.component_probabilities = np.asarray(
            loaded.component_probabilities,
            dtype=float,
        )

        # Ensure the MNE ICA object and wrapper exclusion lists agree.
        self.ica.exclude = self.bad_components.copy()

        return self