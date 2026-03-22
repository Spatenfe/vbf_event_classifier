import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def format_source_details(source_str):
    if not source_str:
        return {}
    if source_str == "single sample":
        return {"Sampling": "Single Event", "Features": "All", "Scaling": "STD", "Layer": "SA + AP"}
        
    source_lower = source_str.lower()
    
    if "10x_" in source_lower:
        sampling = "10x Oversamp"
    elif "3x_" in source_lower:
        sampling = "3x Oversamp"
    elif "undersampled" in source_lower:
        sampling = "Undersampled"
    else:
        sampling = "Baseline"
        
    if "new_features" in source_lower:
        features = "New"
    elif "old_features" in source_lower:
        features = "Old"
    else:
        features = "All"
        
    if "no_scaling" in source_lower or source_lower.endswith("_none"):
        scaling = "None"
    elif "quantile" in source_lower:
        scaling = "Quantile"
    elif "robust" in source_lower:
        scaling = "Robust"
    elif "maxabs" in source_lower:
        scaling = "MaxAbs"
    elif "yeo_johnson" in source_lower:
        scaling = "Yeo-Johns"
    elif "l2" in source_lower:
        scaling = "STD+L2"
    elif "max" in source_lower and "std" in source_lower:
        scaling = "STD+Max"
    elif "std" in source_lower:
        scaling = "STD"
    else:
        scaling = "Other"
        
    return {"Sampling": sampling, "Features": features, "Scaling": scaling, "Layer": ""}

def main():
    parser = argparse.ArgumentParser(description="Plot best configuration per method across ablation runs.")
    parser.add_argument("--results-dir", default="results/ablation", help="Root ablation results directory (default: results/ablation)")
    parser.add_argument("--deepset-score", type=float, default=None, metavar="SCORE", help="Optional baseline score for a DeepSet reference line")
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    csv_files = glob.glob(str(base_dir / "**" / "aggregated_results.csv"), recursive=True)
    
    all_data = []
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        # file_path is like results/ablation/3x_large_all_features/large_data/aggregated_results.csv
        # We want the folder directly under results/ablation
        parts = Path(file_path).parts
        try:
            ablation_idx = parts.index('ablation')
            ablation_folder = parts[ablation_idx + 1]
        except ValueError:
            ablation_folder = "unknown"
            
        df['ablation_folder'] = ablation_folder
        all_data.append(df)
        
    if not all_data:
        print("No aggregated_results.csv found.")
        return
        
    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df[full_df['method'] != 'dummy_classifier']
    
    metric = "val_accuracy"
    print(f"Using metric: {metric}")
    
    # Find best result per method
    best_idx = full_df.groupby('method')[metric].idxmax()
    best_df = full_df.loc[best_idx].copy()
    
    # Create the clear name
    # e.g. 3x_large_all_features_std
    best_df['source_name'] = best_df['ablation_folder'] + "_" + best_df['normalization']
    
    if args.deepset_score is not None:
        deepset_df = pd.DataFrame([{'method': 'DeepSet', metric: args.deepset_score, 'source_name': 'single sample'}])
        best_df = pd.concat([best_df, deepset_df], ignore_index=True)
    
    # Sort for better plotting
    best_df = best_df.sort_values(by=metric, ascending=True)
    
    plt.figure(figsize=(14, 8))
    
    # Create bar plot
    bars = plt.barh(best_df['method'], best_df[metric], color='skyblue')
    
    # Add text labels on the bars with the source name
    for bar, source, score in zip(bars, best_df['source_name'], best_df[metric]):
        # Plot accuracy to the right of the bar
        plt.text(score + 0.005, bar.get_y() + bar.get_height()/2, 
                 f"{score:.4f}", 
                 va='center', ha='left', fontsize=9, color='black')
        
        # Plot source text inside the left of the bar
        if source:
            details = format_source_details(source)
            if details:
                y = bar.get_y() + bar.get_height() / 2
                plt.text(0.01, y, f"Sampling: {details['Sampling']}", va='center', ha='left', fontsize=9, color='black')
                plt.text(0.24, y, f"Features: {details['Features']}", va='center', ha='left', fontsize=9, color='black')
                plt.text(0.42, y, f"Scaling: {details['Scaling']}", va='center', ha='left', fontsize=9, color='black')
                if details.get("Layer"):
                    plt.text(0.55, y, f"Layer: {details['Layer']}", va='center', ha='left', fontsize=9, color='black')
                 
    plt.xlabel(metric)
    plt.title(f"Best Configuration per Method ({metric})")
    
    # Adjust layout to fit labels
    plt.xlim(0, 1.0)  # fixed x-axis to 1.0
    plt.tight_layout()
    
    out_path = base_dir / "best_ablation_methods.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
