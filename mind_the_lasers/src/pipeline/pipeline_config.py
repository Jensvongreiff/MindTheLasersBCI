import os

from .artifact_removal import SpatialFilterICA
from .pipeline_constructor import BCIPipeline
from .feature_extraction import CSPWrapper
from .model import EEGNetBCIWrapper, LDAWrapper


def build_pipeline(baseline: str, window_samples: int, suffix: str = "") -> BCIPipeline:
    """
    Builds a calibrated BCIPipeline.

    Parameters
    ----------
    baseline : "eegnet" or "csp-lda"

    window_samples : number of samples per EEG window
    """

    base_dir = os.path.dirname(__file__)

    # Format the suffix for the file names (e.g., "_0001")
    ext = f"_{suffix}" if suffix else ""    

    if baseline == "eegnet":

        weights_path = os.path.join(
            base_dir,
            "weights",
            f"eegnet{ext}.pt",
        )

        model = EEGNetBCIWrapper(
            weights_path=weights_path,
        )

        return BCIPipeline(
            end_to_end_model=model,
        )
    
    ica_path = os.path.join(
        base_dir,
        "weights",
        "ica.pkl",
    )

    csp_path = os.path.join(
        base_dir,
        "weights",
        f"csp{ext}.pkl",
    )

    lda_path = os.path.join(
        base_dir,
        "weights",
        f"lda{ext}.pkl",
    )

    return BCIPipeline(
        feature_step=CSPWrapper(
            model_path=csp_path,
        ),
        artifact_step= SpatialFilterICA(model_path=ica_path),
        classifier_step=LDAWrapper(
            model_path=lda_path,
        ),
    )