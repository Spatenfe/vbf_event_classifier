import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import pandas as pd

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
    plt.figure(figsize=(10, 6))
    
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
