import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Plot comparison of multiple experiment results from summary CSVs.")
    parser.add_argument("--files", nargs="+", required=True, help="List of paths to summary_results.csv files.")
    parser.add_argument("--labels", nargs="+", required=True, help="List of labels corresponding to each CSV file.")
    parser.add_argument("--baseline", type=int, default=0, help="0-based index of the file to use as the baseline for difference plots.")
    parser.add_argument("--metric", type=str, default="val_accuracy", help="Metric to plot (default: val_accuracy).")
    parser.add_argument("--output", type=str, default="comparison_plot.png", help="Output filename.")
    args = parser.parse_args()

    if len(args.files) != len(args.labels):
        print(f"Error: Number of files ({len(args.files)}) output match number of labels ({len(args.labels)}).")
        return

    if not 0 <= args.baseline < len(args.files):
        print(f"Error: Baseline index {args.baseline} is out of bounds for the {len(args.files)} files provided.")
        return

    # Load DataFrames
    dfs = []
    for fpath in args.files:
        if not os.path.exists(fpath):
            print(f"Warning: File {fpath} not found. Skipping.")
            continue
        try:
            df = pd.read_csv(fpath)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    if len(dfs) < 2:
        print("Error: Need at least 2 valid CSVs to perform relative comparison.")
        return

    metric = args.metric

    # Determine common methods across all loaded datasets that actually have the target metric
    methods_sets = [set(df['method']) for df in dfs if metric in df.columns]
    if not methods_sets:
        print(f"Error: Metric '{metric}' not found in any of the provided CSVs.")
        return
        
    common_methods = set.intersection(*methods_sets)
    if not common_methods:
        print("Error: No common methods found across the provided datasets.")
        return

    # Filter DataFrames to only include common methods
    filtered_dfs = []
    for df in dfs:
        filtered = df[df['method'].isin(common_methods)].drop_duplicates(subset=['method']).set_index('method')
        filtered_dfs.append(filtered)

    # Sort based on the baseline dataframe's order of the chosen metric for better visualization
    baseline_df = filtered_dfs[args.baseline]
    sorted_methods = baseline_df.sort_values(metric, ascending=True).index.tolist()
    
    # Re-order all dataframes to match this sorted index
    aligned_dfs = [df.loc[sorted_methods].reset_index() for df in filtered_dfs]
    baseline_aligned = aligned_dfs[args.baseline]

    # Setup figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 16))

    num_groups = len(aligned_dfs)
    barWidth = 0.8 / num_groups  # Spread within 0.8 space

    # --- Plot 1: Absolute Values ---
    for i, df in enumerate(aligned_dfs):
        # Calculate x positions for this group
        r = [x + i * barWidth for x in range(len(sorted_methods))]
        
        # We can try to match user's custom colors if length matches, otherwise use default colormap
        colors = plt.cm.tab10(i)
        
        ax1.bar(r, df[metric], color=colors, width=barWidth, edgecolor='grey', label=args.labels[i])

    # Styling abstract plot
    ax1.set_ylabel(f'Absolute {metric.replace("_", " ").title()}', fontweight='bold')
    ax1.set_xlabel('Method', fontweight='bold')
    
    # Center ticks
    tick_positions = [x + (num_groups - 1) * barWidth / 2 for x in range(len(sorted_methods))]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(sorted_methods, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_title(f'{metric.replace("_", " ").title()} Comparison: Absolute Values')


    # --- Plot 2: Relative Differences ---
    # We compare every dataset against the baseline, excluding the baseline itself from the bars
    
    comparison_indices = [i for i in range(num_groups) if i != args.baseline]
    num_diff_groups = len(comparison_indices)
    
    if num_diff_groups > 0:
        diff_barWidth = 0.8 / num_diff_groups
        baseline_name = args.labels[args.baseline]
        
        # Define distinct base colors and hatch patterns for multiple comparisons
        # Positive and negative values for the same comparison group get the same color
        base_colors = ['green', 'royalblue', 'purple', 'teal', 'olive']
        hatches = ['', '//', '..', 'xx', '\\\\']
        
        for idx_counter, i in enumerate(comparison_indices):
            # Calculate difference (in percentage points if accuracy)
            diff = (aligned_dfs[i][metric] - baseline_aligned[metric]) * 100
            
            # Position
            r_diff = [x + idx_counter * diff_barWidth for x in range(len(sorted_methods))]
            
            label_diff = f'{baseline_name} vs {args.labels[i]}'
            
            # Cycle through distinct base colors and hatches
            group_color = base_colors[idx_counter % len(base_colors)]
            hatch_pattern = hatches[idx_counter % len(hatches)]
            
            ax2.bar(r_diff, diff, color=group_color, width=diff_barWidth, edgecolor='white', label=label_diff, alpha=0.8, hatch=hatch_pattern)

        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylabel(f'{metric.replace("_", " ").title()} Difference\n(Percentage Points)', fontweight='bold')
        ax2.set_xlabel('Method', fontweight='bold')
        
        tick_positions_diff = [x + (num_diff_groups - 1) * diff_barWidth / 2 for x in range(len(sorted_methods))]
        ax2.set_xticks(tick_positions_diff)
        ax2.set_xticklabels(sorted_methods, rotation=45, ha='right')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.legend()
        ax2.set_title(f'Comparative {metric.replace("_", " ").title()} Improvement over {baseline_name}')

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Plot saved successfully to {args.output}")

    # --- Print Textual Stats ---
    # Exclude dummy_classifier if present
    print("\n" + "="*80)
    print(f"{'STATISTICS SUMMARY':^80}")
    print("="*80)
    
    valid_methods_mask = [m != 'dummy_classifier' for m in sorted_methods]
    if sum(valid_methods_mask) > 0:
        print(f"{'COMPARISON':<35} | {'AVG IMPACT':>12} | {'MIN IMPACT':>12} | {'MAX IMPACT':>12}")
        print("-" * 80)
        
        for i in comparison_indices:
            diff_series = (aligned_dfs[i][metric] - baseline_aligned[metric]) * 100
            diff_valid = diff_series[valid_methods_mask]
            
            avg_diff = diff_valid.mean()
            min_diff = diff_valid.min()
            max_diff = diff_valid.max()
            
            comp_name = f"{baseline_name} vs {args.labels[i]}"
            # Truncate comparison name if too long
            comp_name = comp_name[:32] + "..." if len(comp_name) > 35 else comp_name
            
            print(f"{comp_name:<35} | {avg_diff:>11.2f}% | {min_diff:>11.2f}% | {max_diff:>11.2f}%")
        print("="*80)

if __name__ == "__main__":
    main()
