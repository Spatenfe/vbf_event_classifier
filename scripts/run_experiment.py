import sys
import os
import argparse
import json
import logging
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_framework.core.runner import ExperimentRunner
# Import all methods and dataloaders to ensure registration
import ml_framework.methods.dummy_classifier
import ml_framework.methods.logistic_regression
import ml_framework.methods.random_forest
import ml_framework.methods.svc
import ml_framework.methods.gradient_boosting
import ml_framework.methods.hist_gradient_boosting
import ml_framework.methods.sgd_classifier
import ml_framework.methods.mlp_classifier
import ml_framework.methods.lda
import ml_framework.methods.gaussian_nb
import ml_framework.methods.polynomial_lr
import ml_framework.methods.nystroem_sgd
import ml_framework.methods.bagging_classifier
import ml_framework.methods.voting_classifier
import ml_framework.methods.kmeans_5
import ml_framework.methods.nca_knn
import ml_framework.methods.xgb_classifier
import ml_framework.dataloaders.simple_loader
import ml_framework.dataloaders.standard_loader



def main():
    parser = argparse.ArgumentParser(description="Run General Experiment from Config")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs for methods")
    parser.add_argument("--save-models", action="store_true", help="Save trained models in pickle and ONNX format")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Get experiment name first
    experiment_name = config.get('experiment_name', 'unnamed_experiment')
    
    # Get base path from output config (default to "results")
    output_config = config.get("output", {})
    if isinstance(output_config, str):
        base_path = output_config if output_config else "results"
    else:
        base_path = output_config.get("dir", "results")
    
    # Combine base path with experiment name
    output_base_dir = os.path.join(base_path, experiment_name)
    
    datasources = config.get("datasource", [])
    if isinstance(datasources, dict): datasources = [datasources]
    if not datasources: datasources = [{}]
 
    normalizations = config.get("normalization", [])
    if isinstance(normalizations, dict): normalizations = [normalizations]
    if not normalizations: normalizations = [{}]
 
    method_configs = config.get("methods", [])
    method_n_jobs = config.get("method_n_jobs", None)
    
    tqdm.write(f"Starting experiment: {experiment_name}")
    tqdm.write(f"Output Directory: {output_base_dir}")
    
    if os.path.exists(output_base_dir):
        tqdm.write(f"Error: Output directory '{output_base_dir}' already exists. Exiting to prevent overwriting.")
        sys.exit(1)
        
    os.makedirs(output_base_dir, exist_ok=False)
    
    # Copy configuration file to output directory for referencing later
    import shutil
    config_dest = os.path.join(output_base_dir, "experiment_config.json")
    shutil.copy(args.config, config_dest)
    tqdm.write(f"Saved configuration to: {config_dest}")
    
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
 
    for i, ds in enumerate(tqdm(datasources, desc="Datasources")):
        ds_name = ds.get("name", f"datasource_{i}")
        
        aggregated_results = []
 
        for j, norm in enumerate(tqdm(normalizations, desc="Normalizations", leave=False)):
            norm_name = norm.get("name", f"norm_{j}")
            
            tqdm.write(f"\n--- Running Combination: {ds_name} + {norm_name} ---")
            
            # ... (Dataloader config construction same as before)
            loader_name = ds.get("loader", "standard_loader")
            loader_params = ds.copy()
            if "loader" in loader_params: del loader_params["loader"]
            if "name" in loader_params: del loader_params["name"]
                
            norm_params = norm.copy()
            if "name" in norm_params: del norm_params["name"]
            loader_params["preprocessing"] = norm_params
            
            dataloader_config = {"name": loader_name, "params": loader_params}
            current_output_dir = os.path.join(output_base_dir, ds_name, norm_name)
            
            # Get save_models setting from CLI or config
            save_models = args.save_models or config.get("save_models", False)
            
            runner = ExperimentRunner(
                dataloader_config=dataloader_config,
                method_configs=method_configs,
                output_dir=current_output_dir,
                n_jobs=args.n_jobs,
                method_n_jobs=method_n_jobs,
                save_model=save_models,
            )
            runner.run()
            
            # Collect results for aggregation
            summary_path = os.path.join(current_output_dir, "summary_results.csv")
            if os.path.exists(summary_path):
                df = pd.read_csv(summary_path)
                df["normalization"] = norm_name
                aggregated_results.append(df)

        # Process Aggregation for this Datasource
        if aggregated_results:
            full_df = pd.concat(aggregated_results, ignore_index=True)
            ds_output_dir = os.path.join(output_base_dir, ds_name)
            os.makedirs(ds_output_dir, exist_ok=True)
            
            # Save aggregated CSV
            full_df.to_csv(os.path.join(ds_output_dir, "aggregated_results.csv"), index=False)
            
            # Determine metric to plot
            metric_to_plot = None
            for metric in ["val_accuracy", "test_accuracy", "accuracy"]:
                if metric in full_df.columns:
                    metric_to_plot = metric
                    break
            
            if metric_to_plot:
                # Plot Accuracy Overview
                plt.figure(figsize=(14, 8))
                sns.barplot(x="method", y=metric_to_plot, hue="normalization", data=full_df)
                plt.title(f"Method {metric_to_plot} across Normalizations ({ds_name})")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(ds_output_dir, f"{metric_to_plot}_overview.png"))
                plt.close()
                tqdm.write(f"Created overview plot for {ds_name} at {ds_output_dir}/{metric_to_plot}_overview.png")
            else:
                tqdm.write(f"Warning: No accuracy metric found to plot for {ds_name}. Columns: {full_df.columns.tolist()}")

if __name__ == "__main__":
    main()
