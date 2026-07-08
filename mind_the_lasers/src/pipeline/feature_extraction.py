from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple
import os
import pickle
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from .signal import EEGWindow

# Common BCI Competition IV 2a EEG channel order used by the offline feature_extraction.py.
# These defaults are only a convenience. For another headset/order, pass
# channel_indices or all_channel_names explicitly to the extractor.
BCI2A_EEG_CHANNEL_NAMES = (
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
)

DEFAULT_BANDPOWER_CHANNELS = ("C3", "Cz", "C4")
DEFAULT_MORLET_CHANNELS = ("C3", "C4")

# --- Base Classes ---
class BaseFeatureExtractor(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, sampling_rate: float): pass

    @abstractmethod
    def transform(self, X: np.ndarray, sampling_rate: float) -> np.ndarray: pass

    @abstractmethod
    def extract(self, window: EEGWindow) -> np.ndarray: pass


# --- Shared feature helpers ---
def _as_window_array(window: EEGWindow) -> np.ndarray:
    """Return window data as a float 2D array: (n_channels, n_samples)."""
    data = np.asarray(window.data, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"EEGWindow.data must have shape (n_channels, n_samples), got {data.shape}."
        )
    if data.shape[1] < 2:
        raise ValueError("EEGWindow.data must contain at least two time samples.")
    return data


def _validate_band(band: Tuple[float, float], sfreq: float) -> Tuple[float, float]:
    low, high = float(band[0]), float(band[1])
    nyq = sfreq / 2.0
    if low <= 0:
        raise ValueError("Low cutoff must be > 0 Hz.")
    if high >= nyq:
        raise ValueError(f"High cutoff must be below Nyquist frequency: {nyq} Hz.")
    if low >= high:
        raise ValueError(f"Invalid band {band}: low cutoff must be below high cutoff.")
    return low, high


def _bandpass_window(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Band-pass a single window along the time axis.

    Parameters
    ----------
    data:
        Shape (n_channels, n_samples) or (n_epochs, n_channels, n_samples).
    zero_phase:
        If True, use sosfiltfilt like the offline feature_extraction.py code.
        If False, use causal sosfilt.

    Notes
    -----
    For very short windows, sosfiltfilt can fail because there are not enough
    samples for padding. In that case, this helper falls back to causal sosfilt.
    """
    low, high = _validate_band(band, sfreq)
    nyq = sfreq / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")

    if zero_phase:
        try:
            return sosfiltfilt(sos, data, axis=-1)
        except ValueError:
            # Short online windows may be shorter than sosfiltfilt's pad length.
            return sosfilt(sos, data, axis=-1)
    return sosfilt(sos, data, axis=-1)


def _logvar(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Log-variance over the time axis."""
    return np.log(np.var(data, axis=-1) + eps)


def _resolve_channel_indices(
    channel_indices: Optional[Sequence[int]] = None,
    selected_channels: Optional[Sequence[str]] = None,
    all_channel_names: Optional[Sequence[str]] = None,
) -> Optional[list[int]]:
    """Resolve channel indices from explicit indices or selected channel names.

    Parameters
    ----------
    channel_indices:
        Direct integer channel positions. This always takes precedence.
    selected_channels:
        User-facing channel names to select, e.g. ("C3", "Cz", "C4").
    all_channel_names:
        Full channel order for the incoming EEGWindow.data. If omitted while
        selected_channels are provided, the BCI2A_EEG_CHANNEL_NAMES order is
        used as a convenience default.
    """
    if channel_indices is not None:
        return [int(idx) for idx in channel_indices]

    if selected_channels is None:
        return None

    reference_names = list(all_channel_names) if all_channel_names is not None else list(BCI2A_EEG_CHANNEL_NAMES)
    missing = [ch for ch in selected_channels if ch not in reference_names]
    if missing:
        raise ValueError(
            f"Selected channel(s) {missing} not found in all_channel_names. "
            "Pass channel_indices directly if your online channel order has no names."
        )
    return [reference_names.index(ch) for ch in selected_channels]


def _select_channels(data: np.ndarray, channel_indices: Optional[Sequence[int]]) -> np.ndarray:
    """Select channels from a 2D or 3D EEG array."""
    if channel_indices is None:
        return data
    channel_indices = list(channel_indices)
    if data.ndim == 2:
        return data[channel_indices, :]
    if data.ndim == 3:
        return data[:, channel_indices, :]
    raise ValueError(f"Expected 2D or 3D EEG data, got shape {data.shape}.")


# --- Feature Extractor Implementations ---
class CSPWrapper(BaseFeatureExtractor):
    def __init__(self, model_path: str, n_components: int = 4):
        self.model_path = model_path
        self.n_components = n_components
        self.csp = None
        
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.csp = pickle.load(f)

    def fit(self, X: np.ndarray, y: np.ndarray, sampling_rate: float | None = None):
        """Fits the Common Spatial Pattern filters using MNE decoding logic."""
        from mne.decoding import CSP
        self.csp = CSP(n_components=self.n_components, reg="ledoit_wolf", log=True, norm_trace=False, cov_est='epoch')
        self.csp.fit(X, y)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.csp, f)

    def transform(self, X: np.ndarray, sampling_rate: float | None = None) -> np.ndarray:
        """Transforms a batch dataset (Offline)."""
        if self.csp is None:
            raise RuntimeError(
                "CSP has not been fitted or loaded. Call fit() first."
            )

        return self.csp.transform(X)

    def extract(self, window: EEGWindow) -> np.ndarray:
        """Transforms a single live window (Online)."""
        return self.transform(
            window.data[np.newaxis, :, :],
            sampling_rate=window.sampling_rate,
        )[0]


class BandpowerFeatureExtractor(BaseFeatureExtractor):
    """
    Mu/beta log-bandpower feature extractor.

    This is the online-window equivalent of extract_bandpower_features(). It
    selects configurable channels, band-passes them into mu and beta bands,
    computes log-variance per selected channel, and concatenates the two bands.

    Defaults match the offline code: C3, Cz, C4 using the BCI2A channel order.
    For another headset/order, pass either channel_indices directly or provide
    selected_channels plus all_channel_names.

    Default output order is:
        C3_mu, Cz_mu, C4_mu, C3_beta, Cz_beta, C4_beta
    """

    def __init__(
        self,
        channel_indices: Optional[Sequence[int]] = None,
        selected_channels: Optional[Sequence[str]] = None,
        all_channel_names: Optional[Sequence[str]] = BCI2A_EEG_CHANNEL_NAMES,
        iaf: Optional[float] = None,
        mu_band: Optional[Tuple[float, float]] = None,
        beta_band: Tuple[float, float] = (13.0, 30.0),
        filter_order: int = 4,
        zero_phase: bool = True,
        eps: float = 1e-12,
    ):
        if channel_indices is None and selected_channels is None:
            selected_channels = DEFAULT_BANDPOWER_CHANNELS
        self.selected_channels = tuple(selected_channels) if selected_channels is not None else None
        self.channel_indices = _resolve_channel_indices(
            channel_indices=channel_indices,
            selected_channels=self.selected_channels,
            all_channel_names=all_channel_names,
        )
        self.iaf = iaf
        self.mu_band = mu_band
        self.beta_band = beta_band
        self.filter_order = filter_order
        self.zero_phase = zero_phase
        self.eps = eps

    @property
    def feature_names(self) -> list[str]:
        if self.selected_channels is not None:
            ch_labels = list(self.selected_channels)
        else:
            ch_labels = [f"ch{idx}" for idx in self.channel_indices]
        return [f"{ch}_mu" for ch in ch_labels] + [f"{ch}_beta" for ch in ch_labels]

    @property
    def effective_mu_band(self) -> Tuple[float, float]:
        if self.mu_band is not None:
            return self.mu_band
        if self.iaf is not None:
            return (float(self.iaf) - 2.0, float(self.iaf) + 2.0)
        # Sensible non-subject-specific default when no resting-state IAF is supplied.
        return (8.0, 12.0)
    
    def fit(self, X: np.ndarray, y: np.ndarray, sampling_rate: float | None = None):
        # No fitting needed for bandpower.
        return self

    def transform(
        self,
        X: np.ndarray,
        sampling_rate: float,
    ) -> np.ndarray:
        """
        Transform a batch of EEG epochs into mu/beta log-bandpower features.

        Parameters
        ----------
        X:
            EEG data with shape:
                (n_epochs, n_channels, n_samples)

            A single epoch with shape:
                (n_channels, n_samples)

            is also accepted and converted to a batch of size one.

        sampling_rate:
            Sampling rate in Hz.

        Returns
        -------
        np.ndarray
            Shape:
                (n_epochs, 2 * n_selected_channels)

            Feature order:
                channel_1_mu, channel_2_mu, ...,
                channel_1_beta, channel_2_beta, ...
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        if X.ndim != 3:
            raise ValueError(
                "Expected X with shape "
                f"(n_epochs, n_channels, n_samples), got {X.shape}."
            )

        if X.shape[-1] < 2:
            raise ValueError("Each EEG epoch must contain at least two samples.")

        X_selected = _select_channels(X, self.channel_indices)
        sfreq = float(sampling_rate)

        X_mu = _bandpass_window(
            X_selected,
            sfreq=sfreq,
            band=self.effective_mu_band,
            order=self.filter_order,
            zero_phase=self.zero_phase,
        )

        X_beta = _bandpass_window(
            X_selected,
            sfreq=sfreq,
            band=self.beta_band,
            order=self.filter_order,
            zero_phase=self.zero_phase,
        )

        # Both have shape: (n_epochs, n_selected_channels)
        features_mu = _logvar(X_mu, eps=self.eps)
        features_beta = _logvar(X_beta, eps=self.eps)

        return np.concatenate(
            [features_mu, features_beta],
            axis=1,
        )

    def extract(self, window: EEGWindow) -> np.ndarray:
        return self.transform(
            window.data[np.newaxis, :, :],
            sampling_rate=window.sampling_rate,
        )[0]


class MorletWaveletFeatureExtractor(BaseFeatureExtractor):
    """
    Morlet wavelet log-amplitude feature extractor.

    This is the online-window equivalent of extract_morlet_wavelet_features().
    It applies complex Morlet convolution, takes log-amplitude, optionally crops
    a time window, and flattens channels x frequencies x times.

    Defaults match the offline code: C3/C4, frequencies [8, 10, 12, 20, 24].
    Online, you can either pass time_window=None to use the whole incoming
    EEGWindow, or pass time_window=(1.5, 4.0) if your EEGWindow contains that
    post-cue time range. For another headset/order, pass either channel_indices
    directly or provide selected_channels plus all_channel_names.
    """

    def __init__(
        self,
        channel_indices: Optional[Sequence[int]] = None,
        selected_channels: Optional[Sequence[str]] = None,
        all_channel_names: Optional[Sequence[str]] = BCI2A_EEG_CHANNEL_NAMES,
        freqs: Sequence[float] = (8.0, 10.0, 12.0, 20.0, 24.0),
        n_cycles: Optional[Sequence[float]] = None,
        time_window: Optional[Tuple[float, float]] = None,
        tmin: float = 0.0,
        eps: float = 1e-12,
        use_fft: bool = True,
    ):
        if channel_indices is None and selected_channels is None:
            selected_channels = DEFAULT_MORLET_CHANNELS
        self.selected_channels = tuple(selected_channels) if selected_channels is not None else None
        self.channel_indices = _resolve_channel_indices(
            channel_indices=channel_indices,
            selected_channels=self.selected_channels,
            all_channel_names=all_channel_names,
        )
        self.freqs = np.asarray(freqs, dtype=float)
        self.n_cycles = np.asarray(n_cycles, dtype=float) if n_cycles is not None else self.freqs / 2.0
        self.time_window = time_window
        self.tmin = float(tmin)
        self.eps = eps
        self.use_fft = use_fft

        if self.freqs.ndim != 1 or self.freqs.size == 0:
            raise ValueError("freqs must be a non-empty 1D sequence.")
        if self.n_cycles.shape != self.freqs.shape:
            raise ValueError("n_cycles must have the same length as freqs.")
    
    def fit(self, X: np.ndarray, y: np.ndarray, sampling_rate: float | None = None):
        # No fitting needed for Morlet.
        return self

    def transform(
        self,
        X: np.ndarray,
        sampling_rate: float,
    ) -> np.ndarray:
        """
        Transform a batch of EEG epochs into flattened Morlet
        log-amplitude feature vectors.

        Parameters
        ----------
        X:
            EEG data with shape:
                (n_epochs, n_channels, n_samples)

            A single epoch with shape:
                (n_channels, n_samples)

            is also accepted and converted to a batch of size one.

        sampling_rate:
            Sampling rate in Hz.

        Returns
        -------
        np.ndarray
            Shape:
                (n_epochs, n_selected_channels * n_freqs * n_selected_times)
        """
        from mne.time_frequency import tfr_array_morlet

        X = np.asarray(X, dtype=float)

        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        if X.ndim != 3:
            raise ValueError(
                "Expected X with shape "
                f"(n_epochs, n_channels, n_samples), got {X.shape}."
            )

        if X.shape[-1] < 2:
            raise ValueError("Each EEG epoch must contain at least two samples.")

        X_selected = _select_channels(X, self.channel_indices)
        sfreq = float(sampling_rate)

        complex_tfr = tfr_array_morlet(
            X_selected,
            sfreq=sfreq,
            freqs=self.freqs,
            n_cycles=self.n_cycles,
            output="complex",
            use_fft=self.use_fft,
            n_jobs=None,
        )

        # Shape:
        # (n_epochs, n_selected_channels, n_freqs, n_times)
        log_amplitude = np.log(
            np.abs(complex_tfr) + self.eps
        )

        if self.time_window is not None:
            n_times = X_selected.shape[-1]
            times = self.tmin + np.arange(n_times) / sfreq

            t_start, t_end = self.time_window
            time_mask = (times >= t_start) & (times <= t_end)

            if not np.any(time_mask):
                raise ValueError(
                    f"No samples found in time_window={self.time_window}. "
                    f"Epoch time range is "
                    f"{times[0]:.3f} to {times[-1]:.3f} seconds."
                )

            log_amplitude = log_amplitude[..., time_mask]

        n_epochs = log_amplitude.shape[0]

        return log_amplitude.reshape(n_epochs, -1)

    def extract(self, window: EEGWindow) -> np.ndarray:
        return self.transform(
            window.data[np.newaxis, :, :],
            sampling_rate=window.sampling_rate,
        )[0]


class RiemannianTangentSpaceFeatureExtractor(BaseFeatureExtractor):
    """
    Riemannian covariance + tangent-space feature extractor.

    This is the online-window equivalent of extract_riemannian_mdm_features(),
    with one important online requirement: TangentSpace must be fitted on
    calibration data before it can transform a single incoming window.

    Typical usage:
        extractor = RiemannianTangentSpaceFeatureExtractor(channel_indices=None)
        extractor.fit(X_calibration)  # shape: (epochs, channels, samples)
        features = extractor.extract(window)

    You may also pass a pre-trained pyriemann TangentSpace object.
    """

    def __init__(
        self,
        channel_indices: Optional[Sequence[int]] = None,
        band: Tuple[float, float] = (8.0, 30.0),
        filter_order: int = 4,
        covariance_estimator: str = "oas",
        metric: str = "riemann",
        tangent_space=None,
        zero_phase: bool = True,
        apply_bandpass: bool = False,
    ):
        self.channel_indices = list(channel_indices) if channel_indices is not None else None
        self.band = band
        self.filter_order = filter_order
        self.covariance_estimator = covariance_estimator
        self.metric = metric
        self.tangent_space = tangent_space
        self.zero_phase = zero_phase
        self.apply_bandpass = apply_bandpass

    def _covariances_from_epochs(self, X: np.ndarray, sfreq: float) -> np.ndarray:
        from pyriemann.estimation import Covariances

        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (epochs, channels, samples), got {X.shape}.")

        X_sel = _select_channels(X, self.channel_indices)
        if self.apply_bandpass:
            X_filt = _bandpass_window(
                X_sel,
                sfreq=sfreq,
                band=self.band,
                order=self.filter_order,
                zero_phase=self.zero_phase,
            )
        else:
            X_filt = X_sel

        cov_estimator = Covariances(estimator=self.covariance_estimator)
        return cov_estimator.fit_transform(X_filt)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, sampling_rate: float | None = None) -> "RiemannianTangentSpaceFeatureExtractor":
        """Fit TangentSpace from calibration epochs."""
        from pyriemann.tangentspace import TangentSpace

        if sampling_rate is None:
            raise ValueError("sampling_rate is required.")

        cov_matrices = self._covariances_from_epochs(X, sampling_rate)
        self.tangent_space = TangentSpace(metric=self.metric)
        self.tangent_space.fit(cov_matrices)
        return self
    
    def transform(
        self,
        X: np.ndarray,
        sampling_rate: float,
    ) -> np.ndarray:
        """
        Transform a batch of EEG epochs into Riemannian
        tangent-space feature vectors.

        The tangent-space reference must already have been fitted.

        Parameters
        ----------
        X:
            EEG data with shape:
                (n_epochs, n_channels, n_samples)

            A single epoch with shape:
                (n_channels, n_samples)

            is also accepted and converted to a batch of size one.

        sampling_rate:
            Sampling rate in Hz. It is required when apply_bandpass=True.
            It is still passed consistently when filtering is disabled.

        Returns
        -------
        np.ndarray
            Shape:
                (n_epochs, n_tangent_features)
        """
        if self.tangent_space is None:
            raise RuntimeError(
                "RiemannianTangentSpaceFeatureExtractor requires a fitted "
                "tangent_space. Call fit(...) or pass tangent_space=... first."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        if X.ndim != 3:
            raise ValueError(
                "Expected X with shape "
                f"(n_epochs, n_channels, n_samples), got {X.shape}."
            )

        cov_matrices = self._covariances_from_epochs(
            X,
            sfreq=float(sampling_rate),
        )

        features = self.tangent_space.transform(cov_matrices)

        return np.asarray(features, dtype=float)

    def extract(self, window: EEGWindow) -> np.ndarray:
        return self.transform(
            window.data[np.newaxis, :, :],
            sampling_rate=window.sampling_rate,
        )[0]
