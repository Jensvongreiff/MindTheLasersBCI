from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union
import numpy as np
from .signal import EEGWindow


class BaseArtifactRemoval(ABC):
    """Base class for artifact removal."""
    
    @abstractmethod
    def fit(self, offline_data: np.ndarray):
        """Called only during the calibration phase."""
        pass

    @abstractmethod
    def transform(self, window: EEGWindow) -> EEGWindow:
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
        n_components: Optional[Union[int, float]] = 20,
        random_state: int = 97,
        max_iter: Union[int, str] = "auto",
        method: str = "fastica",
        fit_l_freq: Optional[float] = 1.0,
        fit_h_freq: Optional[float] = 40.0,
        channel_names: Optional[Sequence[str]] = None,
        ch_types: Union[str, Sequence[str]] = "eeg",
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.method = method
        self.fit_l_freq = fit_l_freq
        self.fit_h_freq = fit_h_freq
        self.channel_names = list(channel_names) if channel_names is not None else None
        self.ch_types = ch_types

        self.ica = None
        self.info = None
        self.bad_components: list[int] = []

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

    def _coerce_to_raw(
        self,
        offline_data,
        sampling_rate: Optional[float] = None,
        channel_names: Optional[Sequence[str]] = None,
    ):
        """
        Accept either an existing MNE Raw object or a NumPy array.

        NumPy input must have shape (n_channels, n_samples) and requires
        `sampling_rate`.
        """
        if hasattr(offline_data, "get_data") and hasattr(offline_data, "filter"):
            return offline_data

        if sampling_rate is None:
            raise ValueError(
                "When fitting ICA from a NumPy array, `sampling_rate` must be provided. "
                "Alternatively, pass an MNE Raw object."
            )
        return self._make_raw_from_array(offline_data, sampling_rate, channel_names)

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
            (n_channels, n_samples).
        bad_components:
            ICA component indices to remove during online `transform`.
            If omitted, no components are removed until this list is updated.
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

        # This mirrors the user's previous use_ICA helper: fit ICA on a copy that
        # is filtered 1-40 Hz by default, while preserving the original raw object.
        raw_for_fit = raw_data.copy()
        if self.fit_l_freq is not None or self.fit_h_freq is not None:
            raw_for_fit.filter(
                self.fit_l_freq,
                self.fit_h_freq,
                phase="zero",
                verbose=False,
            )

        self.ica = ICA(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
            method=self.method,
        )
        self.ica.fit(raw_for_fit, verbose=False)

        self.info = raw_data.info.copy()
        self.channel_names = list(raw_data.ch_names)
        self.bad_components = list(bad_components or [])
        self.ica.exclude = self.bad_components.copy()
        return self

    def set_bad_components(self, bad_components: Sequence[int]) -> None:
        """Update the ICA components that should be removed online."""
        if self.ica is None:
            raise ValueError("ICA must be fitted before setting bad components.")
        self.bad_components = list(bad_components)
        self.ica.exclude = self.bad_components.copy()

    def transform(self, window: EEGWindow) -> EEGWindow:
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
            is_artifact_free=True,
        )