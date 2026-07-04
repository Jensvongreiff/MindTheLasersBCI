import time
import threading
import numpy as np
import multiprocessing
from dataclasses import dataclass
from typing import Optional

@dataclass
class EEGWindow:
    data: np.ndarray  
    sampling_rate: int
    timestamp: float  
    is_artifact_free: Optional[bool] = None

class LSLStreamer(threading.Thread):
    """Pulls LSL data, creates EEGWindows, and pushes them to the pipeline queue."""
    def __init__(self, input_queue: multiprocessing.Queue, window_samples: int, stream_type: str = 'EEG'):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.window_samples = window_samples
        self.stream_type = stream_type
        self.stop_event = threading.Event()
        self.inlet = None
        self.fs = 0

    def connect(self):
        try:
            from pylsl import StreamInlet, resolve_byprop
        except ImportError:
            raise ImportError("pylsl is not installed. Run `pip install pylsl`.")

        print(f"Resolving {self.stream_type} stream...")
        streams = resolve_byprop('type', self.stream_type, timeout=5.0)
        if not streams:
            raise RuntimeError(f"No active LSL stream found for type '{self.stream_type}'.")
        
        self.inlet = StreamInlet(streams[0])
        self.fs = int(self.inlet.info().nominal_srate())
        print(f"Connected to stream at {self.fs} Hz.")

    def run(self):
        if self.inlet is None:
            self.connect()

        buffer = []
        while not self.stop_event.is_set():
            chunk, timestamps = self.inlet.pull_chunk(timeout=1.0)
            if chunk:
                buffer.extend(chunk)

                if len(buffer) >= self.window_samples:
                    window_data = np.array(buffer[:self.window_samples]).T
                    ts = timestamps[-1] if timestamps else time.time()

                    eeg_window = EEGWindow(data=window_data, sampling_rate=self.fs, timestamp=ts)

                    try:
                        if self.input_queue.full():
                            self.input_queue.get_nowait()
                        self.input_queue.put_nowait(eeg_window)
                    except Exception:
                        pass
                    
                    buffer = []

    def stop(self):
        self.stop_event.set()