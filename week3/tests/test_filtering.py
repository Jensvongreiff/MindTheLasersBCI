from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from filtering import apply_causal_filter_iir
from week3.loading_helpers import get_raw_offline


fp = Path("/home/dani/Documents/TUM/3.Semester/BCI/baseline-bci-26/data/sub-P999/eeg/sub-P999_ses-S002_task-arrow_run-001_eeg.xdf")

raw,_,_ = get_raw_offline(fp)

y = apply_causal_filter_iir(raw)

print(y)
