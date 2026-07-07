import multiprocessing
import time
import scipy.signal as sig
from typing import Optional, Dict
from .signal import EEGWindow
from .filtering import BaseFilter
from .artifact_removal import BaseArtifactRemoval
from .feature_extraction import BaseFeatureExtractor
from .model import BaseClassifier, BaseEndToEndModel

class BCIPipeline:
    def __init__(
        self,
        filter_step: Optional[BaseFilter] = None,
        artifact_step: Optional[BaseArtifactRemoval] = None,
        feature_step: Optional[BaseFeatureExtractor] = None,
        classifier_step: Optional[BaseClassifier] = None,
        end_to_end_model: Optional[BaseEndToEndModel] = None
    ):
        self.filter_step = filter_step
        self.artifact_step = artifact_step
        
        if end_to_end_model is not None:
            self.model_path = 'end_to_end'
            self.end_to_end_model = end_to_end_model
        else:
            self.model_path = 'standard'
            self.feature_step = feature_step
            self.classifier_step = classifier_step

    def calibrate(self, X_train, y_train, sampling_rate: float | None = None):
        """Fits the pipeline models given an analytical offline training set."""
        print(f"\nCalibrating Pipeline on Dataset: {X_train.shape}...")
        
        # 1. Apply matching causal filter to the offline batch array
        if self.filter_step:
            print("Applying causal filter to training data to match online phase...")
            X_train = sig.lfilter(self.filter_step.b, self.filter_step.a, X_train, axis=2)

        # 2. Proceed to fit the models
        if self.model_path == 'end_to_end':
            self.end_to_end_model.fit(X_train, y_train)
        else:
            self.feature_step.fit(X_train, y_train, sampling_rate=sampling_rate)
            features = self.feature_step.transform(X_train, sampling_rate=sampling_rate)
            self.classifier_step.fit(features, y_train)
        print("Calibration successful. Weights serialized.")

    def process_window(self, window: EEGWindow) -> Dict[str, float]:
        current_data = window
        
        # Execute preprocessing steps if they exist
        if self.filter_step:
            current_data = self.filter_step.process(current_data)
        if self.artifact_step:
            current_data = self.artifact_step.transform(current_data)
            
        if self.model_path == 'end_to_end':
            return self.end_to_end_model.predict_proba(current_data)
        else:
            features = self.feature_step.extract(current_data)
            return self.classifier_step.predict_proba(features)

def bci_worker_process(pipeline: BCIPipeline, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue):
    print("BCI Pipeline Computational Process Initialized.")
    while True:
        try:
            window = input_queue.get() 
            if window is None: # Forward the EOF sentinel
                output_queue.put(None)
                break
            t_start = time.perf_counter()
            probabilities = pipeline.process_window(window)
            t_end = time.perf_counter()

            output_queue.put({
                "probabilities": probabilities, 
                "ground_truth": window.ground_truth,
                "latency": t_end - t_start
            })
        except Exception as e:
            print(f"Non-Fatal Exception in Processing Thread: {e}")