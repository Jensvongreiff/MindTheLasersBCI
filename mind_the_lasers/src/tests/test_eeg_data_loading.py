from mind_the_lasers.src.pipeline.signal import EEGDataLoaderOffline


def main():
    dino_data_path = "D:/Programming/BCI_Practical/practical-ss26-team4/data/sub-P999/eeg/sub-P999_ses-S04_task-dino_run-001_eeg.xdf"
    arrow_long_data_path = "D:/Programming/BCI_Practical/practical-ss26-team4/data/sub-P999/eeg/sub-P999_ses-S003_task-arrow_long_run-001_eeg.xdf"
    arrow_data_path= "D:/Programming/BCI_Practical/practical-ss26-team4/data/sub-P999/eeg/sub-P999_ses-S002_task-arrow_run-001_eeg.xdf"

    # Sample data
    default_run_data_path = "D:/Programming/BCI_Practical/practical-ss26-team4/data/sub-P999/sub-P999_ses-S009_task-Default_run-001_eeg.xdf"
    ses_02 = "D:/Programming/BCI_Practical/practical-ss26-team4/data/sub-P999/ses-S002/sub-P666_ses-S002_task-arrow_run-001_eeg.xdf"
    ses_02_dino_run = "D:/Programming/BCI_Practical/practical-ss26-team4/data/sub-P999/ses-S002/sub-P666_ses-S002_task-dino_run-003_eeg.xdf"

    data_loader = EEGDataLoaderOffline(default_run_data_path, window_length=1.5, stride=0.1)

    X_train, X_test, y_train, y_test = data_loader.load_data()

    print(f"Raw data info:{data_loader.raw_data.info}")
    print(f"Markers info:{data_loader.markers}")
    print(f"Channel labels:{data_loader.channel_labels}")

    epochs = data_loader.epochs
    events = data_loader.events

    epochs_left = epochs[events["LEFT"]]
    epochs_rest = epochs[events["REST"]]
    epochs_right = epochs[events["RIGHT"]]
    print(f"Epochs left:{epochs_left}")
    print(f"Epochs rest:{epochs_rest}")
    print(f"Epochs right:{epochs_right}")

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    import numpy as np
    from collections import Counter


    print("\n--- Class mapping ---")

    print("Configured semantic class -> XDF marker:")
    for class_name, marker_name in data_loader.events.items():
        print(f"  {class_name!r} -> {marker_name!r}")

    print("\nMNE marker -> event ID:")
    for marker_name, event_id in epochs.event_id.items():
        print(f"  {marker_name!r} -> {event_id}")

    print("\nClassifier class -> label:")
    for class_name, label in data_loader.label_mapping.items():
        print(f"  {class_name!r} -> {label}")

    print("\nClassifier label -> class:")
    for label, class_name in data_loader.inverse_label_mapping.items():
        print(f"  {label} -> {class_name!r}")

    mne_id_to_label = {
        epochs.event_id[marker_name]:
            data_loader.label_mapping[class_name]
        for class_name, marker_name in data_loader.events.items()
    }

    print("\nComplete MNE event ID -> classifier label mapping:")
    for mne_id, label in mne_id_to_label.items():
        class_name = data_loader.inverse_label_mapping[label]
        print(f"  MNE ID {mne_id} -> label {label} -> {class_name}")
    
    id_to_marker = {
        event_id: marker_name
        for marker_name, event_id in epochs.event_id.items()
    }

    mne_id_to_label = {
        epochs.event_id[marker_name]:
            data_loader.label_mapping[class_name]
        for class_name, marker_name in data_loader.events.items()
    }

    print("\n--- First 20 original epochs ---")

    for epoch_index, event_code in enumerate(epochs.events[:20, 2]):
        marker_name = id_to_marker[event_code]
        classifier_label = mne_id_to_label[event_code]
        class_name = data_loader.inverse_label_mapping[classifier_label]

        print(
            f"Epoch {epoch_index:3d}: "
            f"MNE ID={event_code}, "
            f"marker={marker_name!r}, "
            f"label={classifier_label}, "
            f"class={class_name}"
        )




if __name__ == "__main__":
    main()