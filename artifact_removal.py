import matplotlib
matplotlib.use('Qt5Agg') 

import matplotlib.pyplot as plt
from pathlib import Path
import mne
from mne.preprocessing import ICA

from loading_helpers import get_raw_offline

def use_ASR():
    pass

def use_ICA(raw: mne.io.RawArray, n_components=20, random_state=97, max_iter="auto", method="fastica"):
    """
    Fit ICA on raw EEG Data
    Parameters
    ----------
    raw : mne.io.RawArray
        raw data RawArray
    Rest: ICA params

    Returns
    -------
    raw_ica : mne.io.RawArray
        ICA processed RawArray data object.
    ica : ICA object
        For data analysis
    """
    raw_data = raw
    ica = ICA(n_components=n_components, random_state=random_state, max_iter=max_iter, method=method)
    ica.fit(raw_data.copy().filter(1., 40., phase="zero"))
    raw_ica = raw_data.copy()
    return raw_ica, ica

def compare_results(raw_path: str, method: str = "ICA"):
    # 1. Setup local repository directory for plots
    plots_dir = Path("plots/temp")
    plots_dir.mkdir(exist_ok=True) # Creates the folder if it doesn't exist
    
    raw, _, _ = get_raw_offline(Path(raw_path))
    
    # 2. Plot and Save Raw Data
    fig_raw = raw.copy().plot(start=10, duration=6, scalings="auto", title="Raw data", show=False)
    fig_raw.savefig(plots_dir / "01_raw_data_before.pdf")
    plt.show() # Blocks until you close the window


    if method == "ICA":
        result, ica = use_ICA(raw, n_components=20, random_state=97, max_iter="auto", method="fastica")
        
        # 3. Plot and Save Topomaps (plot_components can return a list of figures)
        figs_topo = ica.plot_components(show=False)
        if isinstance(figs_topo, list):
            for i, fig in enumerate(figs_topo):
                fig.savefig(plots_dir / f"02_ica_topomaps_page_{i+1}.pdf")
        else:
            figs_topo.savefig(plots_dir / "02_ica_topomaps.pdf")
        plt.show()

        # 4. Plot, Save, and Interact with ICA Sources
        fig_sources = ica.plot_sources(raw, show=False)
            # Save the initial un-clicked state
        fig_sources.savefig(plots_dir / "03_ica_sources_interactive.pdf")
            # Open the window so you can click and exclude components
        plt.show() 

            # Apply the exclusions you clicked on
        print(f"Applying ICA. Excluding components: {ica.exclude}")
        ica.apply(result)

        # 5. Plot and Save the Cleaned Data
        fig_clean = result.plot(start=10, duration=6, scalings="auto", title="After ICA", show=False)
        fig_clean.savefig(plots_dir / "04_raw_data_after.pdf")
        plt.show()
    elif method == "ASR":
        use_ASR()
    else:
        print("Not a valid method")
        return 1

def main():
    data_path = "data/sub-P999/eeg/sub-P999_ses-S002_task-arrow_run-001_eeg.xdf"
    compare_results(data_path, "ICA")

if __name__ == "__main__":
    main()