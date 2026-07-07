import os

from .pipeline_constructor import BCIPipeline
from .feature_extraction import CSPWrapper
from .model import EEGNetBCIWrapper, LDAWrapper


def build_pipeline(baseline: str, window_samples: int):
    """
    Builds a calibrated BCIPipeline.

    Parameters
    ----------
    baseline : "eegnet" or "csp-lda"

    window_samples : number of samples per EEG window
    """

    base_dir = os.path.dirname(__file__)

    if baseline == "eegnet":

        weights_path = os.path.join(
            base_dir,
            "weights",
            "eegnet.pt",
        )

        model = EEGNetBCIWrapper(
            weights_path=weights_path,
            n_channels=22,
            n_samples=window_samples,
        )

        return BCIPipeline(
            end_to_end_model=model,
        )

    csp_path = os.path.join(
        base_dir,
        "weights",
        "csp.pkl",
    )

    lda_path = os.path.join(
        base_dir,
        "weights",
        "lda.pkl",
    )

    return BCIPipeline(
        feature_step=CSPWrapper(
            model_path=csp_path,
        ),
        classifier_step=LDAWrapper(
            model_path=lda_path,
        ),
    )