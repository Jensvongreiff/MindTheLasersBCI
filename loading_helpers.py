import numpy as np
import mne
from pathlib import Path
import pyxdf

def load_one_channel_data(path: str, channel_index: int = 4): # Default C3 channel (motor cortex)
    fp = Path(path)
    raw, _, _ = get_raw_offline(fp)
    X = raw.get_data()
    x = X[channel_index, :]  

    return x

def load_data(path: str):
    fp = Path(path)
    raw, _, _ = get_raw_offline(fp)
    X = raw.get_data()

    return X


def get_raw_offline(
    trial: Path, marker_durations: list[float] | None = None
) -> tuple[mne.io.RawArray, list, list[str]]:
    """
    Function to load the raw data from the trial and return the raw data object, the markers and the channel labels.

    Parameters
    ----------
    trial : Path
        Path to the trial file.
    marker_durations : list, optional
        List of marker durations. The default is [3, 1, 3].

    Returns
    -------
    raw_data : mne.io.RawArray
        Raw data object.
    markers : list
        List of markers.
    channel_labels : list[str]
        List of channel labels.
    """
    # Load the data
    if marker_durations is None:
        marker_durations = [3, 1, 3]

    streams, header = pyxdf.load_xdf(trial, verbose=False)

    event_channel = None
    markers = []
    
    pupil_capture_channel, pupil_capture_fixations_channel, pupil_channel = None, None, None
    eeg_channel = None

    # Iterate through streams to find desired indices
    for i, stream in enumerate(streams):
        name = stream["info"]["name"][0]  # name of the stream
        print(name)
        if name == "EEG - Impedances":
            _index_impedance = i  # impedance stream
        elif name == "EEG":
            eeg_channel = i  # eeg stream
        elif name == "pupil_capture_pupillometry_only":
            pupil_channel = i  # pupil stream
        elif name == "pupil_capture":
            pupil_capture_channel = i  # pupil stream
        elif name == "pupil_capture_fixations":
            pupil_capture_fixations_channel = i
        else:
            event_channel = i  # markers stream

    # Get the channel labels
    channels = streams[eeg_channel]["info"]["desc"][0]["channels"][0]["channel"]
    channel_labels = [channel["label"][0] for channel in channels]
    
    # same for pupil channel
    if pupil_channel is not None:
        pupil_channels = streams[pupil_channel]["info"]["desc"][0]["channels"][0]["channel"]
        pupil_labels = [channel["label"][0] for channel in pupil_channels]
        print("Pupil channels:", pupil_labels)
    if pupil_capture_channel is not None:
        pupil_capture_channels = streams[pupil_capture_channel]["info"]["desc"][0]["channels"][0]["channel"]
        pupil_capture_labels = [channel["label"][0] for channel in pupil_capture_channels]
        print("Pupil capture channels:", pupil_capture_labels)
    if pupil_capture_fixations_channel is not None:
        pupil_capture_fixations_channels = streams[pupil_capture_fixations_channel]["info"]["desc"][0]["channels"][0]["channel"]
        pupil_capture_fixations_labels = [channel["label"][0] for channel in pupil_capture_fixations_channels]
        print("Pupil capture fixations channels:", pupil_capture_fixations_labels)

    # Create montage object - this is needed for the raw data object (Layout of the electrodes)
    montage_dict = {
        channel["label"][0]: [float(channel["location"][0][dim][0]) for dim in "XYZ"] for channel in channels
    }
    montage = mne.channels.make_dig_montage(ch_pos=montage_dict, coord_frame="head")

    # Get EEG data - https://mne.tools/dev/auto_examples/io/read_xdf.html#ex-read-xdf
    data = streams[eeg_channel]["time_series"].T * 1e-6  # scaling the data to volts

    # Get sampling frequency and create info object
    sfreq = float(streams[eeg_channel]["info"]["nominal_srate"][0])
    info = mne.create_info(channel_labels, sfreq, ch_types="eeg")

    # Create raw object and set the montage
    raw_data = mne.io.RawArray(data, info, verbose=False)
    raw_data.set_montage(montage)

    # In the case where the offline collected data (calibration from online) does not have any markers
    if event_channel is None:
        return raw_data, markers, channel_labels

    # get naming of markers and convert to numpy array
    marker = np.array(streams[event_channel]["time_series"]).squeeze()

    # get time stamps of markers
    time_marker = np.array(streams[event_channel]["time_stamps"]).squeeze()

    # get time stamps of data
    time_data = np.array(streams[eeg_channel]["time_stamps"])

    # get relative time of markers
    real_time_marker = (time_marker - time_data[0]).astype(float)

    # Create array of durations for each individual marker
    duration_list = np.zeros(len(real_time_marker))
    for i, _duration in enumerate(duration_list):
        duration_list[i] = marker_durations[i % len(marker_durations)]

    # Annotate the raw data with the markers (So that we know what events are happening at what time in the data)
    annotations = mne.Annotations(onset=real_time_marker, duration=duration_list, description=marker)
    raw_data.set_annotations(annotations)

    # Save the unique markers for later use
    markers = list(set(marker))
    markers.sort()

    return raw_data, markers, channel_labels


if __name__ == "__main__":
    print(load_data("/home/dani/Documents/TUM/3.Semester/BCI/baseline-bci-26/data/sub-P999/eeg/sub-P999_ses-S002_task-arrow_run-001_eeg.xdf"))