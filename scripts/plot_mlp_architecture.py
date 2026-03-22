import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def main():
    parser = argparse.ArgumentParser(description="Visualise an MLP classifier architecture from its config.json.")
    parser.add_argument("--config", default="ml_framework/methods/mlp_classifier/config.json",
                        help="Path to the method config.json (default: ml_framework/methods/mlp_classifier/config.json)")
    parser.add_argument("--output", default="results/mlp_architecture_plot.png",
                        help="Output image path (default: results/mlp_architecture_plot.png)")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = json.load(f)

    params = config.get("params", {})
    layers = params.get("hidden_layer_sizes", [])
    activation = params.get("activation", "")

    fig, ax = plt.subplots(figsize=(12, 4))

    layer_names = ["Input Layer\n(Features)"]
    layer_names += [f"Hidden {i+1}\n({n} nodes)\n{activation.upper()}" for i, n in enumerate(layers)]
    layer_names.append("Output Layer\n(Classes)")

    # x positions for layers
    x_positions = [0.1 + i*0.8/(len(layer_names)-1) for i in range(len(layer_names))]
    y_center = 0.65

    for i, (name, x) in enumerate(zip(layer_names, x_positions)):
        # Draw block
        box_width = 0.6 / len(layer_names)
        box_height = 0.3
        rect = patches.Rectangle((x - box_width/2, y_center - box_height/2), box_width, box_height, 
                                 linewidth=2, edgecolor='darkblue', facecolor='azure', zorder=2)
        ax.add_patch(rect)
        plt.text(x, y_center, name, ha='center', va='center', fontsize=10, zorder=3, weight='bold')
        
        # Draw arrow to next layer
        if i < len(layer_names) - 1:
            next_x = x_positions[i+1]
            ax.annotate('', xy=(next_x - box_width/2, y_center), xytext=(x + box_width/2, y_center),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2), zorder=1)
                        
    # Add training info
    info_text = (f"Optimizer: {str(params.get('solver', 'adam')).upper()} | "
                 f"Learning Rate: {params.get('learning_rate_init')} | "
                 f"Batch Size: {params.get('batch_size')}\n"
                 f"Early Stopping: {params.get('early_stopping')} (Patience {params.get('n_iter_no_change')}) | "
                 f"Max Iterations: {params.get('max_iter')} | "
                 f"L2 Alpha: {params.get('alpha')}")
                 
    plt.text(0.5, 0.20, info_text, ha='center', va='center', fontsize=11, 
             bbox=dict(facecolor='lightgreen', alpha=0.3, edgecolor='green', boxstyle='round,pad=1'))

    plt.axis('off')

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
