import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def plot_comparison(results_df, metric="val_accuracy", output_dir="."):
    """
    Plot comparison of methods.
    results_df: DataFrame containing 'method' and metric columns.
    """
    print(f"Plotting {metric}...")
    n_methods = int(results_df["method"].nunique()) if "method" in results_df.columns else len(results_df)
    # Adjust width based on number of methods
    width = max(10, min(30, 0.8 * max(1, n_methods)))
    
    plt.figure(figsize=(width, 8))
    
    # Check for dummy_classifier baseline
    dummy_row = results_df[results_df["method"] == "dummy_classifier"]
    baseline = None
    if not dummy_row.empty and metric in dummy_row.columns:
        baseline = dummy_row.iloc[0][metric]
        
    # Sort values for better visualization
    results_df_sorted = results_df.sort_values(by=metric, ascending=False)
    
    # Check if metric exists
    if metric not in results_df_sorted.columns:
        print(f"Metric {metric} not found in dataframe. Available columns: {results_df_sorted.columns}")
        return

    sns.barplot(x="method", y=metric, data=results_df_sorted, palette="viridis")
    
    if baseline is not None:
        plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline (Dummy Classifier): {baseline:.3f}')
        plt.legend()
        
    plt.title(f'Comparison of Methods by {metric}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = f"comparison_{metric}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

def main():
    file_path = "results/lower_greater_1/large_data/std/summary_results.csv"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        # Try relative to the user's provided path in prompt if default fails
        file_path = "/home/stud/foef/other/vbf_event_classifier/results/lower_greater_1/large_data/std/summary_results.csv"
        if not os.path.exists(file_path):
            print(f"File also not found at absolute path: {file_path}")
            return

    print(f"Loading results from {file_path}")
    df = pd.read_csv(file_path)
    
    output_dir = os.path.dirname(file_path)
    
    # metrics to plot
    metrics = [col for col in df.columns if col != 'method']
    
    for metric in metrics:
        plot_comparison(df, metric=metric, output_dir=output_dir)

if __name__ == "__main__":
    main()
