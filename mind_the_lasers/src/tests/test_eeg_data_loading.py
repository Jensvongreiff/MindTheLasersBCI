from mind_the_lasers.src.pipeline.signal import EEGDataLoaderOffline


def main():
    dino_data_path = "D:/Programming/BCI_Practical/practical-ss26-team4/data/sub-P999/eeg/sub-P999_ses-S04_task-dino_run-001_eeg.xdf"
    arrow_long_data_path = "D:/Programming/BCI_Practical/practical-ss26-team4/data/sub-P999/eeg/sub-P999_ses-S003_task-arrow_long_run-001_eeg.xdf"
    arrow_data_path= "D:/Programming/BCI_Practical/practical-ss26-team4/data/sub-P999/eeg/sub-P999_ses-S002_task-arrow_run-001_eeg.xdf"

    # Sample data
    default_run_data_path = "D:/Programming/BCI_Practical/practical-ss26-team4/data/sub-P999/sub-P999_ses-S009_task-Default_run-001_eeg.xdf"


    data_loader = EEGDataLoaderOffline(default_run_data_path)

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




if __name__ == "__main__":
    main()