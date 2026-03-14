import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import numpy as np

def plot_confusion_matrix(y_true, y_pred, labels=None, output_dir=None):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    if labels is None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
    else:
        plt.show()

def plot_comparison(results_df, metric="accuracy", output_dir=None):
    """
    Plot comparison of methods.
    results_df: DataFrame containing 'method' and metric columns.
    """
    if metric not in results_df.columns:
        print(f"Warning: Metric '{metric}' not found in results. Skipping plot.")
        return

    n_methods = int(results_df["method"].nunique()) if "method" in results_df.columns else len(results_df)
    width = max(10, min(30, 0.6 * max(1, n_methods)))
    plt.figure(figsize=(width, 6))
    
    # Check for dummy_classifier baseline
    dummy_row = results_df[results_df["method"] == "dummy_classifier"]
    baseline = None
    if not dummy_row.empty:
        baseline = dummy_row.iloc[0][metric]
        
    sns.barplot(x="method", y=metric, data=results_df)
    
    if baseline is not None:
        plt.axhline(y=baseline, color='r', linestyle='--', label=f'Random Guessing ({baseline:.3f})')
        plt.legend()
        
    plt.title(f"Comparison of {metric}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"comparison_{metric}.png"))
        plt.close()
    else:
        plt.show()

def plot_misclassification_overlap(methods_predictions: dict, targets: np.ndarray, output_dir: str, dataset_name='val'):
    """
    Plots the distribution of the number of methods that misclassify each datapoint.
    methods_predictions: dict mapping method_name -> predictions array
    targets: array of true labels
    """
    n_samples = len(targets)
    # List of boolean arrays: True if the method misclassified the sample
    misclassifications = []
    
    filtered_methods = {k: v for k, v in methods_predictions.items() if k != "dummy_classifier"}
    
    for method_name, preds in filtered_methods.items():
        if len(preds) != n_samples:
            print(f"Warning: predictions for {method_name} do not match target length. Skipping for overlap plot.")
            continue
        misclassifications.append(preds != targets)
        
    if not misclassifications:
        return
        
    # Stack and sum: for each sample, how many methods misclassified it
    misclassifications = np.column_stack(misclassifications)
    n_errors_per_sample = misclassifications.sum(axis=1)
    
    # Count occurrences for 0, 1, 2, ... n_methods
    n_methods = len(misclassifications[0])
    counts = [(n_errors_per_sample == i).sum() for i in range(n_methods + 1)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(n_methods + 1), counts, color='lightcoral', edgecolor='black')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom')
             
    plt.xlabel('Number of Methods Misclassifying a Datapoint')
    plt.ylabel('Number of Datapoints')
    
    title = f'Misclassification Overlap ({dataset_name})'
    if "dummy_classifier" in methods_predictions:
        plt.figtext(0.99, 0.01, '* Note: dummy_classifier excluded', horizontalalignment='right', 
                    verticalalignment='bottom', fontsize=9, style='italic', color='gray')
    
    plt.title(title)
    plt.xticks(range(n_methods + 1))
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_misclassification_overlap.png"))
    plt.close()


def plot_method_agreement_matrix(methods_predictions: dict, output_dir: str, dataset_name='val'):
    """
    Plots a heatmap showing the fraction of times methods agree on predictions.
    methods_predictions: dict mapping method_name -> predictions array
    """
    method_names = list(methods_predictions.keys())
    n_methods = len(method_names)
    if n_methods < 2:
        return
        
    # Check lengths
    lengths = {len(p) for p in methods_predictions.values()}
    if len(lengths) > 1:
        print("Warning: Prediction arrays have different lengths. Cannot plot agreement matrix.")
        return
        
    n_samples = list(lengths)[0]
    agreement_matrix = np.zeros((n_methods, n_methods))
    
    for i, name1 in enumerate(method_names):
        preds1 = methods_predictions[name1]
        for j, name2 in enumerate(method_names):
            if i == j:
                agreement_matrix[i, j] = 1.0
            elif i < j:
                preds2 = methods_predictions[name2]
                agreement = np.mean(preds1 == preds2)
                agreement_matrix[i, j] = agreement
                agreement_matrix[j, i] = agreement
                
    plt.figure(figsize=(max(8, n_methods * 0.8), max(6, n_methods * 0.6)))
    sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='viridis', 
                xticklabels=method_names, yticklabels=method_names, 
                vmin=0.0, vmax=1.0)
    
    plt.title(f'Method Agreement Fraction ({dataset_name})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_method_agreement.png"))
    plt.close()
