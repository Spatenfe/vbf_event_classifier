import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
import os

def calculate_metrics(y_true, y_pred, output_dir=None):
    """
    Calculate classification metrics and optionally save to file.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Save classification report
        report = classification_report(y_true, y_pred, zero_division=0)
        with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
            f.write(report)
            
    return metrics
