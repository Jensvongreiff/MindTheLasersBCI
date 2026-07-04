import time
import multiprocessing
import numpy as np
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

    def calibrate(self, trials_per_class: int = 15, trial_duration: float = 1.0, fs: int = 250):
        """Optional synchronous data collection and model fitting phase."""
        try:
            from pylsl import StreamInlet, resolve_byprop
        except ImportError:
            raise ImportError("pylsl required for calibration.")

        print("Resolving EEG stream for calibration...")
        streams = resolve_byprop('type', 'EEG', timeout=5.0)
        inlet = StreamInlet(streams[0])
        
        classes = {"left": 0, "right": 1, "rest": 2}
        X, y = [], []

        print("\nCalibration starting. Follow the console prompts.\n")
        time.sleep(3)

        for class_name, label in classes.items():
            for trial in range(trials_per_class):
                print(f"[{class_name.upper()}] - Trial {trial+1}/{trials_per_class}")
                time.sleep(1.5)
                print(">>> GO! <<<")

                inlet.pull_chunk() 
                start_time = time.time()
                trial_data = []

                while time.time() - start_time < trial_duration:
                    chunk, _ = inlet.pull_chunk()
                    if chunk:
                        trial_data.extend(chunk)

                arr = np.array(trial_data).T
                required_samples = int(trial_duration * fs)
                
                if arr.shape[1] >= required_samples:
                    X.append(arr[:, :required_samples])
                    y.append(label)

        X_train, y_train = np.stack(X), np.array(y)
        print(f"\nCollection complete. Training on shape: {X_train.shape}")

        if self.model_path == 'end_to_end':
            self.end_to_end_model.fit(X_train, y_train)
        else:
            # Requires implementing fit() in Feature extractor and classifier
            features = self.feature_step.fit_transform(X_train, y_train)
            self.classifier_step.fit(features, y_train)
            
        print("Calibration successful. Weights saved.")

    def process_window(self, window: EEGWindow) -> Dict[str, float]:
        current_data = window
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
    print("BCI Worker Process Started.")
    while True:
        try:
            window = input_queue.get() 
            if window is None: 
                break
            probabilities = pipeline.process_window(window)
            output_queue.put({"timestamp": window.timestamp, "probabilities": probabilities})
        except Exception as e:
            print(f"Error in BCI pipeline: {e}")