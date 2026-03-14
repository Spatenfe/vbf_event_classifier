import sys
import os
import argparse
import json
import logging
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import ParameterGrid

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_framework.core.runner import ExperimentRunner

# Import commonly used methods and dataloaders
import ml_framework.methods.xgb_classifier
import ml_framework.methods.random_forest
import ml_framework.methods.mlp_classifier
import ml_framework.methods.gradient_boosting
import ml_framework.methods.hist_gradient_boosting
import ml_framework.dataloaders.standard_loader


def main():
    parser = argparse.ArgumentParser(description="Universal Hyperparameter Grid Search")
    parser.add_argument("--config", type=str, required=True, help="Path to grid search config JSON")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs for methods")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    experiment_name = config.get('experiment_name', 'grid_search_experiment')
    base_path = config.get("output", {}).get("dir", "results")
    output_base_dir = os.path.join(base_path, experiment_name)

    # Grid Search specifics
    grid_search_config = config.get("grid_search", {})
    method_name = grid_search_config.get("method")
    param_grid = grid_search_config.get("param_grid", {})
    metric_to_optimize = grid_search_config.get("metric_to_optimize", "val_accuracy")

    if not method_name or not param_grid:
        print("Error: grid_search block requires 'method' and 'param_grid'.")
        sys.exit(1)

    # Generate all combinations
    combinations = list(ParameterGrid(param_grid))
    
    tqdm.write(f"Starting Grid Search for {method_name}")
    tqdm.write(f"Total combinations to evaluate: {len(combinations)}")
    tqdm.write(f"Output Directory: {output_base_dir}")

    if os.path.exists(output_base_dir):
        tqdm.write(f"Error: Output directory '{output_base_dir}' already exists. Exiting.")
        sys.exit(1)
    os.makedirs(output_base_dir, exist_ok=False)

    import shutil
    shutil.copy(args.config, os.path.join(output_base_dir, "grid_search_config.json"))

    # Construct virtual method configs for the runner
    method_configs = []
    for i, params in enumerate(combinations):
        unique_name = f"{method_name}_config_{i}"
        method_configs.append({
            "name": method_name,
            "display_name": unique_name, # Usually just uses 'name', but we'll override later if needed, otherwise directories will collide
            "params": params,
            "grid_idx": i
        })

    # The ExperimentRunner uses the "name" field to create output directories.
    # To prevent overwriting, we temporarily inject the unique identifier into the class registry,
    # or better, we patch the ExperimentRunner logic. The easiest universal way is to use a 
    # modified method config and let runner handle it, but runner expects standard names.
    # We'll use a wrapper loop to run them sequentially or construct a patched registry.
    
    # Actually, ExperimentRunner currently takes `method_config` and runs `method_name = method_config["name"]`.
    # Let's dynamically duplicate the method registration under unique names for the run.
    from ml_framework.core.registry import Registry
    try:
        base_method_cls = Registry.get_method(method_name)
    except Exception as e:
        print(f"Error getting base method: {e}")
        sys.exit(1)

    runnable_configs = []
    for cfg in method_configs:
        u_name = f"{method_name}_{cfg['grid_idx']}"
        # Register alias on the fly
        Registry._methods[u_name] = base_method_cls
        runnable_configs.append({
            "name": u_name,
            "params": cfg["params"]
        })

    datasources = config.get("datasource", [{}])
    normalizations = config.get("normalization", [{}])

    best_overall_score = -1.0
    best_overall_params = None
    best_overall_config_str = ""

    for i, ds in enumerate(tqdm(datasources, desc="Datasources")):
        ds_name = ds.get("name", f"datasource_{i}")
        
        for j, norm in enumerate(tqdm(normalizations, desc="Normalizations", leave=False)):
            norm_name = norm.get("name", f"norm_{j}")
            
            loader_name = ds.get("loader", "standard_loader")
            loader_params = ds.copy()
            if "loader" in loader_params: del loader_params["loader"]
            if "name" in loader_params: del loader_params["name"]
                
            norm_params = norm.copy()
            if "name" in norm_params: del norm_params["name"]
            loader_params["preprocessing"] = norm_params
            
            dataloader_config = {"name": loader_name, "params": loader_params}
            current_output_dir = os.path.join(output_base_dir, ds_name, norm_name)
            
            runner = ExperimentRunner(
                dataloader_config=dataloader_config,
                method_configs=runnable_configs,
                output_dir=current_output_dir,
                n_jobs=args.n_jobs,
                save_model=False,
            )
            runner.run()
            
            # Read aggregated results
            summary_path = os.path.join(current_output_dir, "summary_results.csv")
            if os.path.exists(summary_path):
                df = pd.read_csv(summary_path)
                
                # Find best performing method
                if metric_to_optimize in df.columns:
                    best_idx = df[metric_to_optimize].idxmax()
                    best_row = df.loc[best_idx]
                    best_method_alias = best_row['method']
                    best_score = best_row[metric_to_optimize]
                    
                    # Map alias back to params
                    best_grid_idx = int(best_method_alias.split('_')[-1])
                    best_params = combinations[best_grid_idx]
                    
                    tqdm.write(f"\n[{ds_name} + {norm_name}] Best {metric_to_optimize}: {best_score:.4f} with params: {best_params}")
                    
                    if best_score > best_overall_score:
                        best_overall_score = best_score
                        best_overall_params = best_params
                        best_overall_config_str = f"{ds_name} + {norm_name}"

    if best_overall_params is not None:
        tqdm.write("\n" + "="*60)
        tqdm.write(f"GRID SEARCH COMPLETE")
        tqdm.write(f"Best overall config: {best_overall_config_str}")
        tqdm.write(f"Best {metric_to_optimize}: {best_overall_score:.4f}")
        tqdm.write(f"Best parameters:")
        for k, v in best_overall_params.items():
            tqdm.write(f"  {k}: {v}")
        tqdm.write("="*60)
        
        # Save best params
        with open(os.path.join(output_base_dir, "best_parameters.json"), "w") as f:
            json.dump({
                "metric": metric_to_optimize,
                "score": best_overall_score,
                "best_config": best_overall_config_str,
                "parameters": best_overall_params
            }, f, indent=4)


if __name__ == "__main__":
    main()
