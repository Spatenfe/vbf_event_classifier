"""
Feature prediction visualization module.

Generates scatter plots showing correct and incorrect predictions
for each input feature.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_feature_predictions(X_val, y_val, predictions, feature_names, output_dir):
    """
    Create scatter plots for each feature showing correct/incorrect predictions.
    
    For each feature, creates a plot with:
    - Green points: correctly classified samples
    - Red points: incorrectly classified samples
    - X-axis: sample index
    - Y-axis: feature value
    
    Args:
        X_val: Validation features (numpy array or DataFrame)
        y_val: True labels (array-like)
        predictions: Predicted labels (array-like)
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    if feature_names is None or len(feature_names) == 0:
        return
    
    # Create output directory for feature plots
    feature_plot_dir = os.path.join(output_dir, "feature_predictions")
    os.makedirs(feature_plot_dir, exist_ok=True)
    
    # Convert predictions and labels to numpy arrays for comparison
    y_val_arr = np.array(y_val)
    pred_arr = np.array(predictions)
    
    # Determine correct and incorrect predictions
    correct_mask = y_val_arr == pred_arr
    incorrect_mask = ~correct_mask
    
    # Get number of features
    n_features = X_val.shape[1] if hasattr(X_val, 'shape') else len(X_val[0])
    n_samples = len(y_val)
    sample_indices = np.arange(n_samples)
    
    # Plot each feature
    for i in range(min(n_features, len(feature_names))):
        feature_name = feature_names[i]
        
        # Extract feature values
        if hasattr(X_val, 'iloc'):
            # DataFrame
            feature_values = X_val.iloc[:, i].values
        else:
            # Numpy array
            feature_values = X_val[:, i]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot incorrect predictions (red) first so they're on top
        if np.any(incorrect_mask):
            ax.scatter(sample_indices[incorrect_mask], 
                      feature_values[incorrect_mask],
                      c='red', alpha=0.6, s=20, label='Incorrect', edgecolors='darkred', linewidth=0.5)
        
        # Plot correct predictions (green)
        if np.any(correct_mask):
            ax.scatter(sample_indices[correct_mask], 
                      feature_values[correct_mask],
                      c='green', alpha=0.6, s=20, label='Correct', edgecolors='darkgreen', linewidth=0.5)
        
        # Formatting
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Feature Value', fontsize=11)
        ax.set_title(f'Feature: {feature_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add accuracy info
        accuracy = np.sum(correct_mask) / len(correct_mask) * 100
        ax.text(0.02, 0.98, f'Accuracy: {accuracy:.1f}%', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        safe_feature_name = feature_name.replace('/', '_').replace('\\', '_')
        plot_path = os.path.join(feature_plot_dir, f'feature_{i:03d}_{safe_feature_name}.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    tqdm.write(f"  Generated {n_features} feature prediction plots in {feature_plot_dir}")
