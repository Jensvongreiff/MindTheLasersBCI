import os
import json
import collections
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from typing import List, Dict, Any

class DecoderEvaluator:
    def __init__(self):
        self.y_true: List[str] = []
        self.y_pred: List[str] = []
        self.confidences: List[float] = []
        self.latencies: List[float] = []
        self.rejections: List[bool] = []
        self.classes = ["left", "rest", "right"]

    def log_step(self, truth: str, pred: str, confidence: float, rejected: bool, latency: float):
        self.y_true.append(truth)
        self.y_pred.append(pred)
        self.confidences.append(confidence)
        self.rejections.append(rejected)
        self.latencies.append(latency)

    def generate_report(self, baseline_name: str) -> Dict[str, Any]:
        if not self.y_true:
            return {}

        acc = accuracy_score(self.y_true, self.y_pred)
        bal_acc = balanced_accuracy_score(self.y_true, self.y_pred)
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.classes)
        report = classification_report(self.y_true, self.y_pred, labels=self.classes, target_names=self.classes, output_dict=True, zero_division=0)
        
        # Calculate true class-wise accuracy (TN + TP / Total)
        class_wise_acc = {}
        for i, cls in enumerate(self.classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fp - fn
            class_wise_acc[cls] = float((tp + tn) / np.sum(cm))

        preds_count = collections.Counter(self.y_pred)
        predictions_per_class = {cls: preds_count.get(cls, 0) for cls in self.classes}

        rest_indices = [i for i, y in enumerate(self.y_true) if y == "rest"]
        if rest_indices:
            false_activations = sum(1 for i in rest_indices if self.y_pred[i] != "rest")
            false_activation_rate = false_activations / len(rest_indices)
        else:
            false_activations = 0
            false_activation_rate = 0.0

        rejection_rate = sum(self.rejections) / len(self.rejections) if self.rejections else 0.0

        conf_dist = {
            "mean": float(np.mean(self.confidences)),
            "median": float(np.median(self.confidences)),
            "std": float(np.std(self.confidences)),
            "min": float(np.min(self.confidences)),
            "max": float(np.max(self.confidences))
        } if self.confidences else {}

        latency_stats = {
            "mean_ms": float(np.mean(self.latencies) * 1000),
            "95th_percentile_ms": float(np.percentile(self.latencies, 95) * 1000)
        } if self.latencies else {}

        metrics = {
            "overall_accuracy": acc,
            "balanced_accuracy": bal_acc,
            "confusion_matrix": cm.tolist(),
            "class_wise_accuracy": class_wise_acc,
            "precision_per_class": {cls: report[cls]["precision"] for cls in self.classes},
            "recall_per_class": {cls: report[cls]["recall"] for cls in self.classes},
            "f1_score_per_class": {cls: report[cls]["f1-score"] for cls in self.classes},
            "macro_f1_score": report["macro avg"]["f1-score"],
            "predictions_per_class": predictions_per_class,
            "false_activations_during_rest": {
                "count": false_activations,
                "rate": false_activation_rate
            },
            "rejection_rate": rejection_rate,
            "confidence_distribution": conf_dist,
            "decision_latency": latency_stats
        }

        os.makedirs("reports", exist_ok=True)
        filepath = os.path.join("reports", f"decoder_metrics_{baseline_name}.json")
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\n[Metrics] Offline simulation report exported to {filepath}")
        
        return metrics