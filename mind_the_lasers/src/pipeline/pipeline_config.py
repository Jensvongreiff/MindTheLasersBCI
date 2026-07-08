import os
import re

from .pipeline_constructor import BCIPipeline
from .filtering import IIRBandpassFilter, OfflineZeroPhaseBandpassCARFilter, OfflineCausalSOSBandpassCARFilter
from .artifact_removal import SpatialFilterICA
from .feature_extraction import CSPWrapper, BandpowerFeatureExtractor
from .model import EEGNetBCIWrapper, LDAWrapper, LDAWrapperTest


BASELINE_ALIASES = {
    "csp-lda": "sos-csp-lda",
}


def _latest_numeric_suffix(
    weights_dir: str,
    required_files: list[tuple[str, str]],
) -> str:
    suffixes: set[str] = set()
    first_prefix, first_extension = required_files[0]
    pattern = re.compile(
        rf"^{re.escape(first_prefix)}_(\d+){re.escape(first_extension)}$"
    )

    for filename in os.listdir(weights_dir):
        match = pattern.match(filename)
        if match:
            suffixes.add(match.group(1))

    complete_suffixes = [
        suffix
        for suffix in suffixes
        if all(
            os.path.exists(
                os.path.join(
                    weights_dir,
                    f"{prefix}_{suffix}{extension}",
                )
            )
            for prefix, extension in required_files
        )
    ]

    if not complete_suffixes:
        raise FileNotFoundError(
            "No complete generated weight set was found in "
            f"{weights_dir}. Run mind_the_lasers.src.tests.evaluate_pipelines "
            "first or pass an explicit suffix."
        )

    return sorted(complete_suffixes, key=int)[-1]


def _resolve_suffix(base_dir: str, baseline: str, suffix: str) -> str:
    if suffix != "latest":
        return suffix

    weights_dir = os.path.join(base_dir, "weights")

    if baseline == "eegnet":
        required_files = [("eegnet", ".pt"), ("ica", ".pkl")]
    elif baseline in {"sos-csp-lda", "zp-csp-lda"}:
        required_files = [("csp", ".pkl"), ("lda", ".pkl"), ("ica", ".pkl")]
    elif baseline == "bp-lda":
        required_files = [("lda", ".pkl")]
    else:
        raise ValueError(f"Cannot resolve latest weights for {baseline!r}.")

    return _latest_numeric_suffix(weights_dir, required_files)


def build_pipeline(baseline: str, window_samples: int, suffix: str = "") -> BCIPipeline:
    """
    Builds a calibrated BCIPipeline.

    Parameters
    ----------
    baseline : "eegnet", "csp-lda", "sos-csp-lda", "zp-csp-lda", or "bp-lda"

    window_samples : number of samples per EEG window
    """

    base_dir = os.path.dirname(__file__)
    baseline = BASELINE_ALIASES.get(baseline, baseline)
    suffix = _resolve_suffix(base_dir, baseline, suffix)

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
            filter_step=OfflineZeroPhaseBandpassCARFilter(),
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
            classifier_step=LDAWrapperTest(
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
            classifier_step=LDAWrapperTest(
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
            classifier_step=LDAWrapperTest(
                model_path=lda_path,
            ),
        )

    raise ValueError(
        "Unknown baseline "
        f"{baseline!r}. Expected one of: csp-lda, sos-csp-lda, "
        "zp-csp-lda, bp-lda, eegnet."
    )
