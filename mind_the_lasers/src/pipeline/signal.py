import time
import threading
import queue
import numpy as np
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import mne
import pyxdf

@dataclass
class EEGWindow:
    """Standardized container for continuous EEG data streams."""
    data: np.ndarray  
    sampling_rate: int
    timestamp: float  
    ground_truth: Optional[str] = None  # Injected during offline simulation for automated evaluation
    is_artifact_free: Optional[bool] = None

class BaseStreamer(threading.Thread):
    """Abstract base class for synchronous data ingestion."""
    def __init__(self, input_queue: multiprocessing.Queue, window_samples: int, fs: int = 250):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.window_samples = window_samples
        self.fs = fs
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

class LSLStreamer(BaseStreamer):
    """Ingests live data from the hardware Lab Streaming Layer using a sliding window."""
    def __init__(self, input_queue: multiprocessing.Queue, window_samples: int, stream_type: str = 'EEG', fs: int = 250):
        super().__init__(input_queue, window_samples, fs)
        self.stream_type = stream_type
        self.stride_samples = int(0.1 * self.fs) # 100ms stride

    def run(self):
        from pylsl import StreamInlet, resolve_byprop
        streams = resolve_byprop('type', self.stream_type, timeout=5.0)
        if not streams:
            raise RuntimeError(f"No active LSL stream found for type '{self.stream_type}'.")
        inlet = StreamInlet(streams[0])

        buffer = []
        while not self.stop_event.is_set():
            chunk, timestamps = inlet.pull_chunk(timeout=1.0)
            if chunk:
                buffer.extend(chunk)
                while len(buffer) >= self.window_samples:
                    window_data = np.array(buffer[:self.window_samples]).T # (Channels, Samples)
                    
                    try:
                        self.input_queue.put_nowait(EEGWindow(data=window_data, sampling_rate=self.fs, timestamp=time.time()))
                    except queue.Full:
                        try:
                            self.input_queue.get_nowait()
                            self.input_queue.put_nowait(EEGWindow(data=window_data, sampling_rate=self.fs, timestamp=time.time()))
                        except (queue.Empty, queue.Full):
                            pass
                    
                    # Apply sliding window stride
                    buffer = buffer[self.stride_samples:]

class OfflineStreamer(BaseStreamer):
    """Simulates a live LSL stream by sequentially feeding pre-recorded epochs into a sliding window."""
    def __init__(self, X_test: np.ndarray, y_test: np.ndarray, input_queue: multiprocessing.Queue, window_samples: int, fs: int = 250):
        super().__init__(input_queue, window_samples, fs)
        self.X_test = X_test
        self.y_test = y_test
        self.stride_samples = int(0.1 * self.fs) # 100ms stride
        self.label_map = {0: "left", 1: "right", 2: "rest"}

    def run(self):
        for trial_idx, (trial_data, label) in enumerate(zip(self.X_test, self.y_test)):
            if self.stop_event.is_set():
                break

            ground_truth = self.label_map[label]
            n_samples = trial_data.shape[1]
            
            # Slide a window across the 3-second trial epoch
            for i in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                window_data = trial_data[:, i:i+self.window_samples]
                
                try:
                    self.input_queue.put_nowait(EEGWindow(
                        data=window_data, 
                        sampling_rate=self.fs, 
                        timestamp=time.time(),
                        ground_truth=ground_truth
                    ))
                except queue.Full:
                    try:
                        self.input_queue.get_nowait()
                        self.input_queue.put_nowait(EEGWindow(
                            data=window_data, 
                            sampling_rate=self.fs, 
                            timestamp=time.time(),
                            ground_truth=ground_truth
                        ))
                    except (queue.Empty, queue.Full):
                        pass
                
                # Sleep to accurately mimic hardware latency (100ms real-time delay)
                time.sleep(0.1)

        # Transmit EOF sentinel to gracefully shut down the pipeline and trigger the evaluation report
        self.input_queue.put(None)

def load_and_split_offline_data(mat_path: str, fs: int = 250) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses the BCI Competition IV 2a dataset.
    Extracts Left (1) -> 0, Right (2) -> 1, and synthesizes Rest (2) from the pre-cue fixation period.
    Generates 3-second epochs to support downstream sliding-window simulation.
    """
    import scipy.io as sio
    from sklearn.model_selection import train_test_split

    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    runs = np.ravel(mat["data"])
    
    X_all, y_all = [], []
    epoch_samples = int(3.0 * fs) # Extract 3 full seconds per trial for sliding window

    for run in runs:
        trial_positions = np.asarray(run.trial, dtype=int).ravel()
        labels = np.asarray(run.y, dtype=int).ravel()
        if len(trial_positions) == 0: continue
        
        X_run = np.asarray(run.X, dtype=float).T
        if X_run.shape[0] >= 22:
            X_run = X_run[:22, :] # Isolate the 22 standard EEG channels

        if trial_positions.min() == 1:
            trial_positions -= 1

        for pos, label in zip(trial_positions, labels):
            if label in [1, 2]: # Left or Right MI
                # Active MI extraction: [0.5s to 3.5s] relative to cue
                start_act = int(pos + 0.5 * fs)
                end_act = start_act + epoch_samples
                if end_act < X_run.shape[1]:
                    X_all.append(X_run[:, start_act:end_act])
                    y_all.append(0 if label == 1 else 1)

                # Rest extraction: [-2.5s to 0.5s] relative to cue (Cross fixation period)
                start_rest = int(pos - 2.5 * fs)
                end_rest = start_rest + epoch_samples
                if start_rest >= 0:
                    X_all.append(X_run[:, start_rest:end_rest])
                    y_all.append(2)

    X = np.stack(X_all, axis=0)
    y = np.asarray(y_all, dtype=int)
    
    # 50/50 Stratified Split for calibration vs. simulation
    return train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

class EEGDataLoaderOffline:
    def __init__(self, data_path: Path | str, events: dict = None):
        self.data_path = data_path
        self.raw_data = None
        self.markers = None
        self.channel_labels = None
        self.epochs = None
        self.filtered_events_id = None
        self.sampling_rate = None

        if events is not None:
            self.events = events
        else:
            self.events = {"LEFT": "ARROW LEFT ONSET",
                        "RIGHT": "ARROW RIGHT ONSET",
                        "REST": "CIRCLE ONSET"}

    def get_raw_offline(self, trial: Path, marker_durations: list[float] | None = None,) -> tuple[mne.io.RawArray, list, list[str]]:
        """
        Load offline calibration data from an XDF file.

        Parameters
        ----------
        trial : Path
            Path to the XDF file.
        marker_durations : list[float] | None
            Durations assigned cyclically to the markers. Defaults to [3, 1, 3].

        Returns
        -------
        raw_data : mne.io.RawArray
            Loaded EEG data.
        markers : list
            Unique marker descriptions.
        channel_labels : list[str]
            EEG channel labels.
        """
        if marker_durations is None:
            marker_durations = [3, 1, 3]

        streams, _ = pyxdf.load_xdf(str(trial), verbose=False)

        event_channel = None
        eeg_channel = None

        pupil_channel = None
        pupil_capture_channel = None
        pupil_capture_fixations_channel = None

        markers = []

        # ------------------------------------------------------------------
        # Find streams explicitly by name and/or stream type.
        # Do not treat every unknown stream as a marker stream.
        # ------------------------------------------------------------------
        for i, stream in enumerate(streams):
            stream_info = stream["info"]

            name = str((stream_info.get("name") or [""])[0])
            stream_type = str((stream_info.get("type") or [""])[0])

            print(name)

            name_lower = name.casefold()
            type_lower = stream_type.casefold()

            if name == "EEG - Impedances":
                continue

            if name == "pupil_capture_pupillometry_only":
                pupil_channel = i

            elif name == "pupil_capture":
                pupil_capture_channel = i

            elif name == "pupil_capture_fixations":
                pupil_capture_fixations_channel = i

            elif name_lower == "eeg" or type_lower == "eeg":
                eeg_channel = i

            elif (
                name_lower in {"markers", "marker", "events", "event"}
                or type_lower in {"markers", "marker", "events", "event"}
            ):
                event_channel = i

        if eeg_channel is None:
            available_streams = [
                (
                    str((stream["info"].get("name") or [""])[0]),
                    str((stream["info"].get("type") or [""])[0]),
                )
                for stream in streams
            ]

            raise RuntimeError(
                "No EEG stream was found in the XDF file. "
                f"Available streams: {available_streams}"
            )

        eeg_stream = streams[eeg_channel]

        # ------------------------------------------------------------------
        # Get EEG channel metadata.
        # ------------------------------------------------------------------
        try:
            channels = (
                eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
            )
        except (KeyError, IndexError, TypeError):
            channels = []

        n_data_channels = np.asarray(eeg_stream["time_series"]).shape[1]

        if len(channels) == n_data_channels:
            channel_labels = []

            for i, channel in enumerate(channels):
                labels = channel.get("label") or []
                label = str(labels[0]) if labels else f"CH{i + 1}"
                channel_labels.append(label)
        else:
            channel_labels = [
                f"CH{i + 1}"
                for i in range(n_data_channels)
            ]

        # ------------------------------------------------------------------
        # Print pupil-channel information when available.
        # ------------------------------------------------------------------
        def get_stream_labels(stream_index: int) -> list[str]:
            try:
                stream_channels = (
                    streams[stream_index]["info"]["desc"][0]
                    ["channels"][0]["channel"]
                )
            except (KeyError, IndexError, TypeError):
                return []

            labels = []

            for i, channel in enumerate(stream_channels):
                channel_label = channel.get("label") or []
                labels.append(
                    str(channel_label[0])
                    if channel_label
                    else f"CH{i + 1}"
                )

            return labels

        if pupil_channel is not None:
            print(
                "Pupil channels:",
                get_stream_labels(pupil_channel),
            )

        if pupil_capture_channel is not None:
            print(
                "Pupil capture channels:",
                get_stream_labels(pupil_capture_channel),
            )

        if pupil_capture_fixations_channel is not None:
            print(
                "Pupil capture fixations channels:",
                get_stream_labels(
                    pupil_capture_fixations_channel
                ),
            )

        # ------------------------------------------------------------------
        # Create the MNE RawArray.
        #
        # The existing scaling is retained: the XDF EEG values are assumed
        # to be stored in microvolts and are converted to volts for MNE.
        # ------------------------------------------------------------------
        data = np.asarray(
            eeg_stream["time_series"],
            dtype=float,
        ).T * 1e-6

        sfreq = float(
            eeg_stream["info"]["nominal_srate"][0]
        )

        info = mne.create_info(
            ch_names=channel_labels,
            sfreq=sfreq,
            ch_types="eeg",
        )

        raw_data = mne.io.RawArray(
            data,
            info,
            verbose=False,
        )

        # ------------------------------------------------------------------
        # Create a montage only from channels that contain valid XYZ
        # coordinates. Missing coordinates no longer cause the loader to fail.
        # ------------------------------------------------------------------
        montage_dict = {}

        for channel, label in zip(channels, channel_labels):
            locations = channel.get("location") or []

            if not locations or not isinstance(locations[0], dict):
                continue

            location = locations[0]
            position = []

            try:
                for axis in ("X", "Y", "Z"):
                    axis_values = location.get(axis) or []

                    if not axis_values:
                        raise ValueError(
                            f"Missing {axis} coordinate"
                        )

                    position.append(float(axis_values[0]))

            except (TypeError, ValueError, IndexError):
                continue

            if np.all(np.isfinite(position)):
                montage_dict[label] = position

        if montage_dict:
            montage = mne.channels.make_dig_montage(
                ch_pos=montage_dict,
                coord_frame="head",
            )

            raw_data.set_montage(
                montage,
                on_missing="warn",
            )
        else:
            print(
                "No valid electrode coordinates were found in the XDF "
                "metadata. Continuing without a montage."
            )

        # ------------------------------------------------------------------
        # Calibration recordings may contain no marker stream.
        # ------------------------------------------------------------------
        if event_channel is None:
            return raw_data, markers, channel_labels

        event_stream = streams[event_channel]

        marker = np.asarray(
            event_stream["time_series"],
        ).reshape(-1)

        marker = np.asarray(
            [str(value) for value in marker],
            dtype=str,
        )

        time_marker = np.asarray(
            event_stream["time_stamps"],
            dtype=float,
        ).reshape(-1)

        time_data = np.asarray(
            eeg_stream["time_stamps"],
            dtype=float,
        ).reshape(-1)

        if time_data.size == 0:
            raise RuntimeError(
                "The EEG stream contains no timestamps."
            )

        real_time_marker = time_marker - time_data[0]

        duration_list = np.asarray(
            [
                marker_durations[i % len(marker_durations)]
                for i in range(len(real_time_marker))
            ],
            dtype=float,
        )

        annotations = mne.Annotations(
            onset=real_time_marker,
            duration=duration_list,
            description=marker,
        )

        raw_data.set_annotations(annotations)

        markers = sorted(set(marker.tolist()))

        self.raw_data = raw_data
        self.markers = markers
        self.channel_labels = channel_labels
        self.sampling_rate = sfreq

    def get_epochs(
        self,
        raw_data: mne.io.Raw,
        markers: list,
        tmin: float = 0.3,
        tmax: float = 1.3,
        baseline: tuple[float, float] | None = None,
        event_dict: dict | None = None,
    ) -> tuple[mne.Epochs, dict]:
        """
        Get the epochs from the raw data based on the markers.

        Parameters
        ----------
        raw_data : mne.io.RawArray
            Raw data object.
        markers : list
            List of markers.
        tmin : float, optional
            Start time of the time window (epoch). The default is 0.3.
        tmax : float, optional
            End time of the time window (epoch). The default is 1.3.
        baseline : tuple, optional
            Window used for baseline correction. The default is None.
        event_dict : dict, optional
            Dictionary of event IDs. The default is None.
        Returns
        -------
        epochs : mne.Epochs
            Epochs object.
        filtered_events_id : dict
            Dictionary of filtered events.

        """
        # Get the events based on the annotations (in our case the ARROW markers)
        markers_dict = {marker: i for i, marker in enumerate(markers)}
        if event_dict is not None:
            events, events_id = mne.events_from_annotations(raw_data, event_id=event_dict, verbose=False)
        else:
            events, events_id = mne.events_from_annotations(raw_data, event_id=markers_dict, verbose=False)

        # Get the events that are in the marker_IDs
        filtered_events_id = {key: value for key, value in events_id.items() if "" in key}
        tmin_ = None


        # Adapt time window in case we need to do baseline correction
        if baseline:
            tmin_ = tmin
            tmin = baseline[0]
        else:
            baseline = None

        # Create epochs
        epochs = mne.Epochs(
            raw_data,
            events=events,
            tmin=tmin,
            tmax=tmax,
            event_id=filtered_events_id,
            baseline=baseline,
            preload=True,
            verbose=False,
            event_repeated='drop'
        )

        # Crop the epochs to the desired time window
        if tmin_ is not None:
            epochs.crop(tmin=tmin_, tmax=tmax)

        self.epochs = epochs
        self.filtered_events_id = filtered_events_id

    def load_data(
        self,
        test_size: float = 0.5,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the XDF recording, create task epochs, and return a stratified
        train/test split.

        The class labels are assigned according to the order in self.events.

        With the default configuration:
            0 -> LEFT
            1 -> RIGHT
            2 -> REST

        Parameters
        ----------
        test_size : float, optional
            Fraction of epochs assigned to the test set. Default is 0.5.
        random_state : int, optional
            Random seed used for the train/test split. Default is 42.

        Returns
        -------
        X_train : np.ndarray
            Training EEG epochs with shape
            (n_train_epochs, n_channels, n_timepoints).
        X_test : np.ndarray
            Test EEG epochs with shape
            (n_test_epochs, n_channels, n_timepoints).
        y_train : np.ndarray
            Integer training labels.
        y_test : np.ndarray
            Integer test labels.
        """
        from sklearn.model_selection import train_test_split

        if not 0.0 < test_size < 1.0:
            raise ValueError(
                f"test_size must be between 0 and 1, got {test_size}."
            )

        if not self.events:
            raise ValueError("No task events have been configured.")

        class_names = list(self.events.keys())
        marker_names = list(self.events.values())

        if len(set(marker_names)) != len(marker_names):
            raise ValueError(
                "Each class in self.events must reference a unique marker. "
                f"Received: {self.events}"
            )

        # Load the continuous EEG data and annotations.
        self.get_raw_offline(Path(self.data_path))

        if self.raw_data is None:
            raise RuntimeError("The XDF file did not produce an MNE Raw object.")

        if self.markers is None:
            raise RuntimeError("The XDF file did not produce a marker list.")

        # Confirm that all requested task markers exist in the recording.
        available_markers = {str(marker) for marker in self.markers}

        missing_markers = [
            marker
            for marker in marker_names
            if marker not in available_markers
        ]

        if missing_markers:
            raise ValueError(
                "The following configured task markers were not found in "
                f"the recording: {missing_markers}"
            )

        # get_epochs() expects:
        #     annotation description -> positive integer event ID
        #
        # self.events instead stores:
        #     semantic class name -> annotation description
        event_dict = {
            marker_name: event_id
            for event_id, marker_name in enumerate(
                marker_names,
                start=1,
            )
        }

        self.get_epochs(
            raw_data=self.raw_data,
            markers=self.markers,
            event_dict=event_dict,
            tmin=0.3,
            tmax=3.3,
        )

        if self.epochs is None or len(self.epochs) == 0:
            raise RuntimeError(
                "No epochs were created for the configured task events."
            )

        # EEG data shape:
        # (n_epochs, n_channels, n_timepoints)
        X = self.epochs.get_data().copy()

        # Translate MNE's positive event IDs into zero-based classifier labels.
        #
        # Default:
        #     event ID 1 -> LEFT  -> label 0
        #     event ID 2 -> RIGHT -> label 1
        #     event ID 3 -> REST  -> label 2
        event_id_to_label = {
            event_dict[marker_name]: class_index
            for class_index, marker_name in enumerate(marker_names)
        }

        y = np.asarray(
            [
                event_id_to_label[event_code]
                for event_code in self.epochs.events[:, 2]
            ],
            dtype=np.int64,
        )

        # Store the mappings for interpreting predictions downstream.
        self.class_names = class_names

        self.label_mapping = {
            class_name: class_index
            for class_index, class_name in enumerate(class_names)
        }

        self.inverse_label_mapping = {
            class_index: class_name
            for class_name, class_index in self.label_mapping.items()
        }

        self.event_dict = event_dict

        # A stratified split requires at least two samples of every class.
        class_counts = np.bincount(
            y,
            minlength=len(class_names),
        )

        classes_with_too_few_samples = [
            class_names[class_index]
            for class_index, count in enumerate(class_counts)
            if count < 2
        ]

        if classes_with_too_few_samples:
            raise ValueError(
                "A stratified train/test split requires at least two epochs "
                "for every class. Too few epochs were found for: "
                f"{classes_with_too_few_samples}"
            )

        return train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
