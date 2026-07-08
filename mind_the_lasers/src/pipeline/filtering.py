from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt, sosfilt_zi, sosfilt
from .signal import EEGWindow
from typing import Literal

class BaseFilter(ABC):
    """Base class for all filtering steps."""
    @abstractmethod
    def process(self, window: EEGWindow) -> EEGWindow:
        pass

class IIRBandpassFilter(BaseFilter):
    """
    Example: 8-30Hz Bandpass filter for Motor Imagery.
    Uses lfilter (causal) instead of filtfilt (non-causal) for online safe processing.

    note: Look into maintaining filter states between windows to reduce edge artifacts.
    """
    def __init__(self, lowcut: float, highcut: float, fs: int, order: int = 4):
        self.fs = fs
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # Calculate filter coefficients once during initialization
        self.b, self.a = butter(order, [low, high], btype='band')
        print(f"Initialized IIRBandpassFilter with lowcut={lowcut}Hz, highcut={highcut}Hz, fs={fs}Hz, order={order}")

    def process(self, window: EEGWindow) -> EEGWindow:
        # Apply the filter along the time axis (axis=1)
        filtered_data = lfilter(self.b, self.a, window.data, axis=1)
        
        # Common Average Reference (CAR) is applied after filtering
        filtered_data = filtered_data - filtered_data.mean(axis=0, keepdims=True)
        
        # Return a new window object to prevent mutability bugs
        return EEGWindow(
        data=filtered_data,
        sampling_rate=window.sampling_rate,
        timestamp=window.timestamp,
        ground_truth=window.ground_truth,          # CRITICAL ADDITION
        is_artifact_free=window.is_artifact_free)


class OfflineZeroPhaseBandpassCARFilter(BaseFilter):
    """
    Offline diagnostic filter for motor-imagery EEG.

    Processing:
        1. Fourth-order Butterworth band-pass
        2. Zero-phase forward-backward filtering
        3. Common-average reference

    This reproduces the preprocessing used in the successful ablation:

        8-30 Hz band-pass
        -> CAR

    Important
    ---------
    This filter is not suitable for real-time inference because
    sosfiltfilt uses future samples.
    """

    def __init__(
        self,
        lowcut: float = 8.0,
        highcut: float = 30.0,
        fs: float = 250.0,
        order: int = 4,
    ):
        if fs <= 0:
            raise ValueError(f"Sampling rate must be positive, received {fs}.")

        if lowcut <= 0:
            raise ValueError(
                f"Low cutoff must be positive, received {lowcut}."
            )

        if highcut >= fs / 2:
            raise ValueError(
                f"High cutoff must be below Nyquist frequency "
                f"({fs / 2:.2f} Hz), received {highcut} Hz."
            )

        if lowcut >= highcut:
            raise ValueError(
                f"lowcut must be below highcut, received "
                f"{lowcut} >= {highcut}."
            )

        self.lowcut = float(lowcut)
        self.highcut = float(highcut)
        self.fs = float(fs)
        self.order = int(order)

        # Second-order sections are numerically more stable than
        # direct-form b/a coefficients.
        self.sos = butter(
            N=self.order,
            Wn=[self.lowcut, self.highcut],
            btype="bandpass",
            fs=self.fs,
            output="sos",
        )

        print(
            "Initialized OfflineZeroPhaseBandpassCARFilter("
            f"lowcut={self.lowcut} Hz, "
            f"highcut={self.highcut} Hz, "
            f"fs={self.fs} Hz, "
            f"order={self.order})"
        )

    def _validate_sampling_rate(self, sampling_rate: float) -> None:
        if not np.isclose(sampling_rate, self.fs):
            raise ValueError(
                "Filter sampling-rate mismatch: "
                f"filter was designed for {self.fs} Hz, "
                f"but data have {sampling_rate} Hz."
            )

    def process(self, window: EEGWindow) -> EEGWindow:
        """
        Filter one EEGWindow.

        Expected input shape:
            (n_channels, n_samples)
        """
        self._validate_sampling_rate(window.sampling_rate)

        data = np.asarray(window.data, dtype=np.float64)

        if data.ndim != 2:
            raise ValueError(
                "Expected window.data with shape "
                "(n_channels, n_samples), "
                f"received {data.shape}."
            )

        if not np.isfinite(data).all():
            raise ValueError(
                "EEG window contains NaN or infinite values."
            )

        # Filter along the final dimension: the time/sample axis.
        filtered_data = sosfiltfilt(
            self.sos,
            data,
            axis=-1,
        )

        # Common-average reference:
        # for each time sample, subtract the mean across channels.
        filtered_data = (
            filtered_data
            - filtered_data.mean(axis=0, keepdims=True)
        )

        return EEGWindow(
            data=filtered_data,
            sampling_rate=window.sampling_rate,
            timestamp=window.timestamp,
            ground_truth=window.ground_truth,
            is_artifact_free=window.is_artifact_free,
        )

    def process_batch(
        self,
        X: np.ndarray,
        sampling_rate: float,
    ) -> np.ndarray:
        """
        Filter a batch of training windows.

        Expected input shape:
            (n_windows, n_channels, n_samples)

        This method is needed during BCIPipeline.calibrate().
        """
        self._validate_sampling_rate(sampling_rate)

        data = np.asarray(X, dtype=np.float64)

        if data.ndim != 3:
            raise ValueError(
                "Expected training data with shape "
                "(n_windows, n_channels, n_samples), "
                f"received {data.shape}."
            )

        if not np.isfinite(data).all():
            raise ValueError(
                "Training data contain NaN or infinite values."
            )

        # Time/sample axis is the final dimension.
        filtered_data = sosfiltfilt(
            self.sos,
            data,
            axis=-1,
        )

        # For a batch shaped:
        # (windows, channels, samples),
        # channels are axis 1.
        filtered_data = (
            filtered_data
            - filtered_data.mean(axis=1, keepdims=True)
        )

        return filtered_data


class OfflineCausalSOSBandpassCARFilter(BaseFilter):
    """
    Causal Butterworth band-pass filter implemented with second-order sections,
    followed by common-average referencing (CAR).

    Intended offline use
    --------------------
    Apply ``filter_continuous()`` to an entire continuous recording before
    epoching or creating overlapping windows. This gives the same causal filter
    behavior that a future online stateful implementation should reproduce.

    Important
    ---------
    ``process()`` and ``process_batch()`` reset the filter state for every
    supplied window/epoch. They exist for compatibility with the current
    pipeline interface, but they do not reproduce continuous online filtering
    when called on overlapping sliding windows.
    """

    def __init__(
        self,
        lowcut: float = 8.0,
        highcut: float = 30.0,
        fs: float = 250.0,
        order: int = 4,
        apply_car: bool = True,
        initial_state: Literal["zeros", "first_sample"] = "zeros",
    ) -> None:
        self.lowcut = float(lowcut)
        self.highcut = float(highcut)
        self.fs = float(fs)
        self.order = int(order)
        self.apply_car = bool(apply_car)
        self.initial_state = initial_state

        self._validate_configuration()

        self.sos = butter(
            N=self.order,
            Wn=(self.lowcut, self.highcut),
            btype="bandpass",
            fs=self.fs,
            output="sos",
        )

        # Template used only when initial_state="first_sample".
        self._zi_template = sosfilt_zi(self.sos)

    def _validate_configuration(self) -> None:
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, received {self.fs}.")

        if self.order < 1:
            raise ValueError(
                f"order must be a positive integer, received {self.order}."
            )

        nyquist = self.fs / 2.0

        if not 0.0 < self.lowcut < self.highcut < nyquist:
            raise ValueError(
                "Cutoffs must satisfy "
                f"0 < lowcut < highcut < Nyquist ({nyquist:g} Hz), "
                f"received lowcut={self.lowcut:g}, "
                f"highcut={self.highcut:g}."
            )

        if self.initial_state not in {"zeros", "first_sample"}:
            raise ValueError(
                "initial_state must be either 'zeros' or 'first_sample', "
                f"received {self.initial_state!r}."
            )

    def _validate_sampling_rate(self, sampling_rate: float) -> None:
        if not np.isclose(float(sampling_rate), self.fs):
            raise ValueError(
                "Sampling-rate mismatch: "
                f"filter was designed for {self.fs:g} Hz, "
                f"but the data report {float(sampling_rate):g} Hz."
            )

    def _make_initial_state(self, data: np.ndarray) -> np.ndarray:
        """
        Return zi with the shape required by sosfilt for axis=-1.

        For data shaped:
            (channels, samples)
        zi is shaped:
            (sections, channels, 2)

        For data shaped:
            (epochs, channels, samples)
        zi is shaped:
            (sections, epochs, channels, 2)
        """
        leading_shape = data.shape[:-1]
        zi_shape = (self.sos.shape[0], *leading_shape, 2)

        if self.initial_state == "zeros":
            return np.zeros(zi_shape, dtype=np.float64)

        # Scale the steady-state step-response state independently for every
        # signal by that signal's first sample.
        template_shape = (
            self.sos.shape[0],
            *((1,) * len(leading_shape)),
            2,
        )
        template = self._zi_template.reshape(template_shape)
        first_sample = data[..., 0]

        return template * first_sample[np.newaxis, ..., np.newaxis]

    def filter_array(
        self,
        data: np.ndarray,
        *,
        sampling_rate: float | None = None,
    ) -> np.ndarray:
        """
        Causally filter an array along its final axis.

        Supported shapes
        ----------------
        Continuous recording:
            (n_channels, n_samples)

        Batch of independent epochs:
            (n_epochs, n_channels, n_samples)

        CAR is applied across the second-to-last axis, which is the channel
        axis for both supported shapes.
        """
        if sampling_rate is not None:
            self._validate_sampling_rate(sampling_rate)

        array = np.asarray(data, dtype=np.float64)

        if array.ndim not in (2, 3):
            raise ValueError(
                "Expected data shaped (channels, samples) or "
                "(epochs, channels, samples), "
                f"received {array.shape}."
            )

        if array.shape[-1] == 0:
            raise ValueError("The data contain no time samples.")

        if not np.isfinite(array).all():
            raise ValueError("The data contain NaN or infinite values.")

        if self.apply_car and array.shape[-2] < 2:
            raise ValueError(
                "CAR requires at least two channels. "
                f"Received {array.shape[-2]} channel."
            )

        zi = self._make_initial_state(array)

        filtered, _ = sosfilt(
            self.sos,
            array,
            axis=-1,
            zi=zi,
        )

        if self.apply_car:
            filtered = (
                filtered
                - filtered.mean(axis=-2, keepdims=True)
            )

        return filtered

    def filter_continuous(
        self,
        data: np.ndarray,
        *,
        sampling_rate: float | None = None,
    ) -> np.ndarray:
        """
        Filter one full continuous recording.

        This is the preferred method for offline emulation of the eventual
        online causal filter. Epoching and sliding-window creation should happen
        after this method has been called.
        """
        array = np.asarray(data)

        if array.ndim != 2:
            raise ValueError(
                "filter_continuous expects "
                "(n_channels, n_samples), "
                f"received {array.shape}."
            )

        return self.filter_array(
            array,
            sampling_rate=sampling_rate,
        )

    def filter_epochs(
        self,
        epochs: np.ndarray,
        *,
        sampling_rate: float | None = None,
    ) -> np.ndarray:
        """
        Filter independent full epochs.

        The state is reset at the beginning of every epoch. This is preferable
        to resetting it for every overlapping one-second window, but filtering
        the original continuous recording before epoching is more faithful to
        online operation.
        """
        array = np.asarray(epochs)

        if array.ndim != 3:
            raise ValueError(
                "filter_epochs expects "
                "(n_epochs, n_channels, n_samples), "
                f"received {array.shape}."
            )

        return self.filter_array(
            array,
            sampling_rate=sampling_rate,
        )

    def process(self, window: EEGWindow) -> EEGWindow:
        """
        Filter one EEGWindow with a newly initialized filter state.

        Do not use this method to emulate continuous filtering over overlapping
        sliding windows. The future stateful implementation should consume each
        new sample exactly once before windows are assembled.
        """
        filtered_data = self.filter_array(
            window.data,
            sampling_rate=window.sampling_rate,
        )

        return EEGWindow(
            data=filtered_data,
            sampling_rate=window.sampling_rate,
            timestamp=window.timestamp,
            ground_truth=window.ground_truth,
            is_artifact_free=window.is_artifact_free,
        )

    def process_batch(
        self,
        X: np.ndarray,
        *,
        sampling_rate: float,
    ) -> np.ndarray:
        """
        Pipeline-compatible batch method.

        Each item is treated as an independent epoch and receives a fresh
        initial state. Do not pass already-overlapping sliding windows here when
        the goal is continuous causal emulation.
        """
        return self.filter_epochs(
            X,
            sampling_rate=sampling_rate,
        )


