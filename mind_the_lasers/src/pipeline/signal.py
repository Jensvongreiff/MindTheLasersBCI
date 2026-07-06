import time
import threading
import queue
import numpy as np
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

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