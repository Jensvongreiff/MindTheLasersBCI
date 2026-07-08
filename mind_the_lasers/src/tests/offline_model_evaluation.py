import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

from mind_the_lasers.src.pipeline.signal import EEGDataLoaderOffline, EEGWindow
from mind_the_lasers.src.pipeline.pipeline_config import build_pipeline

def main():
    # 1. Configuration
    dataset_path = r"C:\Users\marti\Documents\Programmieren\RCI\4Semester\BCI\practical-ss26-team4\data\sub-P999\our_structure\sub-P666_ses-S002_task-arrow_run-001_eeg.xdf" 
    
    fs = 250
    window_samples = int(fs * 1.0)  # 1-second window matches the 1-second epoch from DataLoader
    n_splits = 5                    
    baselines = ["csp-lda", "eegnet"]
    label_map = {0: "left", 1: "right", 2: "rest"}

    # 2. Data Loading
    print(f"Loading dataset: {dataset_path}")
    data_loader = EEGDataLoaderOffline(data_path=dataset_path)
    X_train, X_test, y_train, y_test = data_loader.load_data()
    
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    
    print(f"Total epochs available: {X.shape[0]}")
    print(f"Data shape (Epochs, Channels, Samples): {X.shape}\n")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 3. Static Evaluation Loop (No Smoothing, No Sliding Window)
    for baseline in baselines:
        print(f"{'='*40}")
        print(f"Evaluating Baseline: {baseline.upper()}")
        print(f"{'='*40}")
        
        fold_accuracies = []
        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            # Initialize pipeline
            pipeline = build_pipeline(baseline, window_samples, suffix="temp")
            pipeline.calibrate(X_train_fold, y_train_fold, sampling_rate=fs)

            y_true_fold = []
            y_pred_fold = []

            for trial_data, label in zip(X_test_fold, y_test_fold):
                # Take exactly one window per trial (static classification)
                window_data = trial_data[:, :window_samples]
                
                window = EEGWindow(
                    data=window_data, 
                    sampling_rate=fs, 
                    timestamp=time.time(),
                    ground_truth=label_map[label]
                )

                # Raw Inference (Returns Dict[str, float])
                probs = pipeline.process_window(window)
                
                # Extract the class with the highest probability
                pred_label = max(probs, key=probs.get)

                y_true_fold.append(label_map[label])
                y_pred_fold.append(pred_label)

            fold_acc = accuracy_score(y_true_fold, y_pred_fold)
            fold_accuracies.append(fold_acc)
            
            all_y_true.extend(y_true_fold)
            all_y_pred.extend(y_pred_fold)
            
            print(f"Fold {fold}/{n_splits} Accuracy: {fold_acc:.2f}")

        print(f"\n{baseline.upper()} OVERALL MEAN ACCURACY: {np.mean(fold_accuracies):.2f} ± {np.std(fold_accuracies):.2f}")
        print("\nClassification Report:")
        print(classification_report(all_y_true, all_y_pred))
        print("\n")

if __name__ == "__main__":
    main()