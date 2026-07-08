import os

from .pipeline_constructor import BCIPipeline
from .filtering import IIRBandpassFilter, OfflineZeroPhaseBandpassCARFilter, OfflineCausalSOSBandpassCARFilter
from .artifact_removal import SpatialFilterICA
from .feature_extraction import CSPWrapper, BandpowerFeatureExtractor
from .model import EEGNetBCIWrapper, LDAWrapper, LDAWrapperTest


def build_pipeline(baseline: str, window_samples: int, suffix: str = "") -> BCIPipeline:
    """
    Builds a calibrated BCIPipeline.

    Parameters
    ----------
    baseline : "eegnet" or "csp-lda"

    window_samples : number of samples per EEG window
    """

    base_dir = os.path.dirname(__file__)
    data_path=r"C:\Users\marti\Documents\Programmieren\RCI\4Semester\BCI\practical-ss26-team4\data\sub-P999\ses-S002\eeg\sub-P666_ses-S002_task-arrow_run-001_eeg.xdf"

    # Format the suffix for the file names (e.g., "_0001")
    ext = f"_{suffix}" if suffix else ""    

    if baseline == "eegnet":

        weights_path = os.path.join(
            base_dir,
            "weights",
            f"eegnet{ext}.pt",
        )

        ica_path = os.path.join(
            base_dir, 
            "weights", 
            f"ica{ext}.pkl",
        )

        model = EEGNetBCIWrapper(
            weights_path=weights_path,
        )

        return BCIPipeline(
            filter_step=IIRBandpassFilter(lowcut=8.0, highcut=30.0, fs=250),
            artifact_step=SpatialFilterICA(model_path=ica_path),
            end_to_end_model=model,
        )

    elif baseline == "sos-csp-lda":
        csp_path = os.path.join(
                base_dir,
                "weights",
                f"csp{ext}.pkl",
            )

        ica_path = os.path.join(
            base_dir,
            "weights",
            f"ica{ext}.pkl",
            )


        lda_path = os.path.join(
            base_dir,
            "weights",
            f"lda{ext}.pkl",
        )

        return BCIPipeline(
            filter_step=OfflineCausalSOSBandpassCARFilter(),
            feature_step=CSPWrapper(
                model_path=csp_path,
            ),
            artifact_step= SpatialFilterICA(model_path=ica_path),
            classifier_step=LDAWrapper(
                model_path=lda_path,
            ),
        )

    elif baseline == "zp-csp-lda":
        csp_path = os.path.join(
                base_dir,
                "weights",
                f"csp{ext}.pkl",
            )

        ica_path = os.path.join(
            base_dir,
            "weights",
            f"ica{ext}.pkl",
            )


        lda_path = os.path.join(
            base_dir,
            "weights",
            f"lda{ext}.pkl",
        )

        return BCIPipeline(
            filter_step=OfflineZeroPhaseBandpassCARFilter(),
            feature_step=CSPWrapper(
                model_path=csp_path,
            ),
            artifact_step= SpatialFilterICA(model_path=ica_path),
            classifier_step=LDAWrapper(
                model_path=lda_path,
            ),
        )
    
    elif baseline == "bp-lda":
        ica_path = os.path.join(
            base_dir,
            "weights",
            f"ica{ext}.pkl",
            )


        lda_path = os.path.join(
            base_dir,
            "weights",
            f"lda{ext}.pkl",
        )

        return BCIPipeline(
            filter_step=IIRBandpassFilter(lowcut=8.0, highcut=30.0, fs=250),
            feature_step=BandpowerFeatureExtractor(),
            # artifact_step= SpatialFilterICA(model_path=ica_path),
            classifier_step=LDAWrapper(
                model_path=lda_path,
            ),
        )

