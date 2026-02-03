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
import ml_framework.dataloaders.simple_loader
import ml_framework.dataloaders.standard_loader



def main():
    parser = argparse.ArgumentParser(description="Run General Experiment from Config")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs for methods")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # ... (Parsing logic remains the same)
    output_config = config.get("output", {})
    if isinstance(output_config, str):
        output_base_dir = "results/experiment" 
    else:
        output_base_dir = output_config.get("dir", "results/experiment")
    
    datasources = config.get("datasource", [])
    if isinstance(datasources, dict): datasources = [datasources]
    if not datasources: datasources = [{}]
 
    normalizations = config.get("normalization", [])
    if isinstance(normalizations, dict): normalizations = [normalizations]
    if not normalizations: normalizations = [{}]
 
    method_configs = config.get("methods", [])
    experiment_name = config.get('experiment_name', 'Unnamed')
    method_n_jobs = config.get("method_n_jobs", None)
    
    tqdm.write(f"Starting experiment: {experiment_name}")
    tqdm.write(f"Output Base Directory: {output_base_dir}")
    
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
            
            runner = ExperimentRunner(
                dataloader_config=dataloader_config,
                method_configs=method_configs,
                output_dir=current_output_dir,
                n_jobs=args.n_jobs,
                method_n_jobs=method_n_jobs,
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
            
            # Plot Accuracy Overview
            plt.figure(figsize=(14, 8))
            sns.barplot(x="method", y="accuracy", hue="normalization", data=full_df)
            plt.title(f"Method Accuracy across Normalizations ({ds_name})")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(ds_output_dir, "accuracy_overview.png"))
            plt.close()
            
            tqdm.write(f"Created overview plot for {ds_name} at {ds_output_dir}/accuracy_overview.png")

if __name__ == "__main__":
    main()
