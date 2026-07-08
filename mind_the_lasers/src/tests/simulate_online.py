import os
import csv
import json
import argparse
import pyxdf
import numpy as np

from mind_the_lasers.src.pipeline.signal import EEGWindow
from mind_the_lasers.src.pipeline.pipeline_config import build_pipeline
from mind_the_lasers.src.pipeline.smoothing import SmoothingController

def evaluate_simulation(csv_path: str, output_json_path: str):
    """Parses the continuous simulation CSV to generate scientific conclusions."""
    print("Evaluating simulation data for scientific metrics...")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("CSV is empty. Cannot evaluate.")
        return

    # Metrics accumulators
    rest_probs = {'left': [], 'rest': [], 'right': []}
    active_matches = 0
    active_total = 0
    rest_matches = 0
    rest_total = 0
    
    activation_lags = []
    deactivation_lags = []
    
    # State tracking for temporal dynamics
    current_true_state = rows[0]['True_Marker']
    state_start_time = float(rows[0]['Timestamp'])
    lag_resolved = True

    for i, row in enumerate(rows):
        true_marker = row['True_Marker']
        pred = row['Smoothed_Prediction']
        ts = float(row['Timestamp'])
        
        # State Transition Detection
        if true_marker != current_true_state:
            current_true_state = true_marker
            state_start_time = ts
            lag_resolved = False

        # Lag Resolution (Wait until prediction matches the new true state)
        if not lag_resolved and pred == true_marker:
            lag = (ts - state_start_time) * 1000  # Convert to ms
            if true_marker == 'rest':
                deactivation_lags.append(lag)
            else:
                activation_lags.append(lag)
            lag_resolved = True

        # Accuracy and Bias Accumulation
        if true_marker == 'rest':
            rest_total += 1
            if pred == 'rest': rest_matches += 1
            rest_probs['left'].append(float(row['Prob_Left']))
            rest_probs['rest'].append(float(row['Prob_Rest']))
            rest_probs['right'].append(float(row['Prob_Right']))
        else:
            active_total += 1
            if pred == true_marker: active_matches += 1

    # Computations
    mean_rest_probs = {k: float(np.mean(v)) if v else 0.0 for k, v in rest_probs.items()}
    avg_act_lag = float(np.mean(activation_lags)) if activation_lags else 0.0
    avg_deact_lag = float(np.mean(deactivation_lags)) if deactivation_lags else 0.0
    
    false_activation_rate = 1.0 - (rest_matches / rest_total) if rest_total > 0 else 0.0
    active_accuracy = (active_matches / active_total) if active_total > 0 else 0.0

    # Dynamic Scientific Conclusions
    conclusions = []
    
    # Evaluate Bias
    dominant_rest_class = max(mean_rest_probs, key=mean_rest_probs.get)
    if dominant_rest_class != 'rest' and mean_rest_probs[dominant_rest_class] > 0.40:
        conclusions.append(
            f"Severe Class Bias: During inter-trial intervals, the model defaults to '{dominant_rest_class}' "
            f"with an average probability of {mean_rest_probs[dominant_rest_class]:.2f}. Spatial filters (CSP/ICA) "
            "are failing to center the baseline variance, interpreting raw noise as motor imagery."
        )
    elif false_activation_rate > 0.30:
        conclusions.append(
            f"High False Activation Rate ({false_activation_rate:.2%}): The smoothing confidence threshold "
            "is set too low to suppress the inherent noise of the resting state."
        )

    # Evaluate Temporal Dynamics
    if avg_deact_lag > 500:
        conclusions.append(
            f"Sticky Lock-In Effect: Majority voting introduces a severe delay ({avg_deact_lag:.0f}ms) when "
            "transitioning from a command back to rest, severely hindering game responsiveness."
        )

    # Evaluate Overall Viability
    if active_accuracy < 0.40:
        conclusions.append(
            f"Low Feature Separability: The model struggles to identify genuine commands ({active_accuracy:.2%} active accuracy), "
            "suggesting the frequency band configuration does not capture this subject's ERD/ERS phenomena."
        )
    elif active_accuracy > 0.70 and false_activation_rate < 0.15:
        conclusions.append(
            "Optimal State: The pipeline successfully distinguishes active ERD/ERS from baseline noise "
            "with minimal lag, making it viable for online gameplay."
        )

    # Assemble JSON payload
    report = {
        "continuous_accuracy": {
            "overall": ((active_matches + rest_matches) / (active_total + rest_total)) if (active_total + rest_total) > 0 else 0.0,
            "during_active_markers": active_accuracy,
            "during_rest": (rest_matches / rest_total) if rest_total > 0 else 0.0,
            "false_activation_rate": false_activation_rate
        },
        "bias_analysis_during_rest": mean_rest_probs,
        "temporal_dynamics": {
            "average_activation_lag_ms": avg_act_lag,
            "average_deactivation_lag_ms": avg_deact_lag
        },
        "scientific_conclusions": conclusions
    }

    with open(output_json_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Scientific evaluation complete. JSON saved to {output_json_path}")
    print("\nIdentified Conclusions:")
    for c in conclusions:
        print(f"- {c}")


def main():
    parser = argparse.ArgumentParser(description="Run continuous online BCI pipeline simulation and scientific evaluation.")
    
    parser.add_argument(
        "--path", 
        type=str, 
        default=r"C:\Users\marti\Documents\Programmieren\RCI\4Semester\BCI\practical-ss26-team4\data\sub-P999\ses-S002\eeg\sub-P666_ses-S002_task-arrow_run-001_eeg.xdf",
        help="Path to the input .xdf recording file"
    )
    parser.add_argument(
        "--baseline", 
        type=str, 
        default="csp-lda", 
        choices=["csp-lda", "eegnet"],
        help="BCI processing baseline track to evaluate"
    )
    parser.add_argument(
        "--suffix", 
        type=str, 
        default="0001",
        help="4-digit run suffix matching calibrated production weights"
    )
    # ==========================================
    # 1. Configuration
    # ==========================================
    args = parser.parse_args()
    dataset_path = args.path
    baseline = args.baseline
    run_suffix = args.suffix
    
    fs = 250
    window_samples = int(fs * 1.0) 
    stride_samples = int(fs * 0.1)  
    trial_duration_sec = 3.0        
    
    os.makedirs("reports", exist_ok=True)
    output_csv = f"mind_the_lasers/reports/online_simulation_{baseline}_{run_suffix}.csv"
    output_json = f"mind_the_lasers/reports/online_evaluation_{baseline}_{run_suffix}.json"

    # ==========================================
    # 2. XDF Parsing & Stream Alignment
    # ==========================================
    print(f"Loading continuous stream from {dataset_path}...")
    streams, header = pyxdf.load_xdf(dataset_path)
    
    eeg_stream = None
    marker_stream = None

    for stream in streams:
        t = stream["info"]["type"][0].lower()
        if t == 'eeg':
            eeg_stream = stream
        elif t == 'markers':
            marker_stream = stream

    if not eeg_stream or not marker_stream:
        raise ValueError("Could not find both EEG and Marker streams in the .xdf file.")

    eeg_times = eeg_stream["time_stamps"]
    eeg_data = eeg_stream["time_series"].T
    
    marker_times = marker_stream["time_stamps"]
    marker_labels = [m[0] for m in marker_stream["time_series"]]

    # ==========================================
    # 3. Pipeline Initialization
    # ==========================================
    print(f"Initializing {baseline.upper()} pipeline with weights _{run_suffix}...")
    pipeline = build_pipeline(baseline, window_samples, suffix=run_suffix)
    smoothing = SmoothingController(window_size=5, confidence_threshold=0.60)

    # ==========================================
    # 4. Continuous Simulation & Logging
    # ==========================================
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp", "True_Marker", 
            "Prob_Left", "Prob_Rest", "Prob_Right", 
            "Smoothed_Prediction", "Rejected_by_Threshold"
        ])

        marker_idx = 0
        total_markers = len(marker_times)
        
        print("Running continuous online simulation...")
        for i in range(0, eeg_data.shape[1] - window_samples + 1, stride_samples):
            window_end_time = eeg_times[i + window_samples - 1]
            
            active_marker = "rest"
            
            while marker_idx < total_markers - 1 and window_end_time >= marker_times[marker_idx + 1]:
                marker_idx += 1
                
            if marker_idx < total_markers:
                if marker_times[marker_idx] <= window_end_time <= marker_times[marker_idx] + trial_duration_sec:
                    raw_label = str(marker_labels[marker_idx]).lower()
                    if "left" in raw_label or "1" in raw_label: active_marker = "left"
                    elif "right" in raw_label or "2" in raw_label: active_marker = "right"

            window = EEGWindow(
                data=eeg_data[:, i:i+window_samples], 
                sampling_rate=fs, 
                timestamp=window_end_time,
                ground_truth=active_marker
            )

            probs = pipeline.process_window(window)
            pred_label, conf, rejected = smoothing.process(probs)

            writer.writerow([
                f"{window_end_time:.3f}", active_marker,
                f"{probs.get('left', 0):.3f}", f"{probs.get('rest', 0):.3f}", f"{probs.get('right', 0):.3f}",
                pred_label, rejected
            ])

    print(f"Simulation complete. Log saved to {output_csv}")

    # ==========================================
    # 5. Scientific Evaluation
    # ==========================================
    evaluate_simulation(output_csv, output_json)

if __name__ == "__main__":
    main()