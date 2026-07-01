import multiprocessing
from typing import Optional, Union, Dict
from signal import EEGWindow
from filtering import BaseFilter
from artifact_removal import BaseArtifactRemoval
from model import BaseFeatureExtractor, BaseClassifier, BaseEndToEndModel

class BCIPipeline:
    """
    Assembles and executes the chosen processing steps.
    Handles varying lengths based on whether an End-to-End model or a 
    Feature + Classifier combination is used.
    """
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
        
        # Validation to ensure input/output logic holds
        if end_to_end_model is not None:
            if feature_step is not None or classifier_step is not None:
                raise ValueError("Provide EITHER an end_to_end_model OR (feature_step + classifier_step).")
            self.model_path = 'end_to_end'
            self.end_to_end_model = end_to_end_model
        else:
            if feature_step is None or classifier_step is None:
                raise ValueError("Standard model requires both feature_step and classifier_step.")
            self.model_path = 'standard'
            self.feature_step = feature_step
            self.classifier_step = classifier_step

    def process_window(self, window: EEGWindow) -> Dict[str, float]:
        """Runs a single window through the configured pipeline."""
        current_data = window
        
        # 1. Filtering
        if self.filter_step:
            current_data = self.filter_step.process(current_data)
            
        # 2. Artifact Removal
        if self.artifact_step:
            current_data = self.artifact_step.transform(current_data)
            
        # 3. Model Execution
        if self.model_path == 'end_to_end':
            return self.end_to_end_model.predict_proba(current_data)
        else:
            features = self.feature_step.extract(current_data)
            return self.classifier_step.predict_proba(features)


# --- Multiprocessing Strategy for parallel execution ---
def bci_worker_process(pipeline: BCIPipeline, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue):
    """
    This function should be spawned as a separate Process.
    It waits for data from the LSL stream (or buffer), processes it, and sends 
    probabilities to the game.
    """
    print("BCI Worker Process Started.")
    while True:
        try:
            # Block until a new EEG window is available from the hardware/buffer
            window = input_queue.get() 
            
            if window is None: # Sentinel value to shut down process
                break
                
            # Process the data
            probabilities = pipeline.process_window(window)
            
            # Send the result to the game loop non-blocking
            output_queue.put({
                "timestamp": window.timestamp,
                "probabilities": probabilities
            })
            
        except Exception as e:
            print(f"Error in BCI pipeline: {e}")