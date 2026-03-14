import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_framework.core.registry import Registry
from ml_framework.core.metrics import calculate_metrics

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

def load_dataloader(config):
    datasources = config.get("datasource", [])
    if isinstance(datasources, dict): datasources = [datasources]
    if not datasources: datasources = [{}]
    
    # Just take the first for simplicity, assuming one datasource
    ds = datasources[0]
    
    loader_name = ds.get("loader", "standard_loader")
    loader_params = ds.copy()
    if "loader" in loader_params: del loader_params["loader"]
    if "name" in loader_params: del loader_params["name"]
        
    normalizations = config.get("normalization", [])
    if isinstance(normalizations, dict): normalizations = [normalizations]
    if not normalizations: normalizations = [{}]
    
    # We will need to re-init loader per normalization
    return loader_name, loader_params, ds.get("name", "datasource_0"), normalizations

def compute_ensembles(y_true, predictions_dict, probas_dict):
    """
    Given a dictionaries of method -> predictions, computes different ensemble strategies.
    Returns a dict of metrics.
    """
    n_samples = len(y_true)
    method_names = list(predictions_dict.keys())
    n_methods = len(method_names)
    metrics_out = {}
    
    if n_methods == 0:
        return metrics_out

    # 1. Hard Voting (Majority Vote)
    preds_stack = np.column_stack([predictions_dict[m] for m in method_names])
    hard_vote_preds = []
    for i in range(n_samples):
        unique, counts = np.unique(preds_stack[i], return_counts=True)
        hard_vote_preds.append(unique[np.argmax(counts)])
    hard_vote_preds = np.array(hard_vote_preds)
    
    metrics_out["hard_voting_accuracy"] = accuracy_score(y_true, hard_vote_preds)
    metrics_out["hard_voting_f1_macro"] = f1_score(y_true, hard_vote_preds, average='macro', zero_division=0)
    
    # 2. Soft Voting (Argmax of Avg Probabilities)
    soft_methods = list(probas_dict.keys())
    if len(soft_methods) > 0:
        avg_probas = np.mean([probas_dict[m] for m in soft_methods], axis=0)
        # We need the class labels to map argmax back to the actual class
        # Assuming unique classes from y_true or probas
        try:
            # unique classes across validation set should match
            classes = np.unique(y_true)
            classes.sort()
            soft_vote_preds = classes[np.argmax(avg_probas, axis=1)]
            metrics_out["soft_voting_accuracy"] = accuracy_score(y_true, soft_vote_preds)
            metrics_out["soft_voting_f1_macro"] = f1_score(y_true, soft_vote_preds, average='macro', zero_division=0)
        except Exception as e:
            pass
            # Soft voting fails gracefully if probas is misaligned, metrics left out.
            
    # 3. Logistic Regression Stacking (using validation probabilities or predictions as features)
    if len(soft_methods) > 1:
        # Flatten probabilities for features
        X_stack = np.hstack([probas_dict[m] for m in soft_methods])
        stacker = LogisticRegression(max_iter=1000)
        try:
            stacker.fit(X_stack, y_true)
            stack_preds = stacker.predict(X_stack)
            # This evaluates stacking ON THE VALIDATION SET it was just trained on.
            # Warning: Overfitting! Proper stacking requires K-fold cross-val.
            metrics_out["stacking_val_accuracy_overfit"] = accuracy_score(y_true, stack_preds)
            metrics_out["stacking_val_f1_macro_overfit"] = f1_score(y_true, stack_preds, average='macro', zero_division=0)
        except Exception as e:
            pass
            
    return metrics_out

def plot_ensemble_performance(results, out_dir, dataset_name, norm_name, metric_name, diversity_mode):
    """
    Plots the ensemble performance across different modes vs number of models.
    """
    plt.figure(figsize=(10, 6))
    
    best_single_score = None
    
    for mode_name, k_metrics in results.items():
        ks = sorted(k_metrics.keys())
        vals = [k_metrics[k] for k in ks]
        plt.plot(ks, vals, marker='o', label=mode_name)
        
        # k=1 represents the single best model (both Top-K and Diverse-K start with it)
        if 1 in k_metrics:
            if best_single_score is None or k_metrics[1] > best_single_score:
                best_single_score = k_metrics[1]
                
    if best_single_score is not None:
        plt.axhline(y=best_single_score, color='red', linestyle=':', linewidth=2, label='Best Individual Model')
    
    plt.xlabel("Number of Models in Ensemble")
    plt.ylabel(metric_name)
    plt.title(f"Ensemble Performance vs Number of Models ({dataset_name} / {norm_name})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # safe filename
    diversity_str = "_diverse" if diversity_mode else ""
    metric_str = metric_name.lower().replace(' ', '_')
    plot_path = os.path.join(out_dir, f"ensemble_{metric_str}{diversity_str}_{dataset_name}_{norm_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Ensemble Saved Methods")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    parser.add_argument("--results-dirs", nargs='+', required=True, help="Path(s) to results base directory for loading models")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the ensemble summary and plots (defaults to the first results-dir)")
    parser.add_argument("--drop-n-worst", type=int, default=0, help="Drop the N worst performing models based on selected metric before ensembling")
    parser.add_argument("--diversity-mode", action="store_true", help="Evaluate ensemble adding most diverse models iteratively")
    parser.add_argument("--metric", type=str, choices=['accuracy', 'f1_macro'], default='f1_macro', help="Metric to rank and select best models")
    args = parser.parse_args()

    # ensure results-dirs is a list
    if isinstance(args.results_dirs, str):
        args.results_dirs = [args.results_dirs]

    with open(args.config, 'r') as f:
        config = json.load(f)

    loader_name, loader_params_base, ds_name, normalizations = load_dataloader(config)
    
    all_ensemble_results = []

    for norm in normalizations:
        norm_name = norm.get("name", "norm_0")
        print(f"\n--- Processing {ds_name} / {norm_name} ---")
        
        # Setup specific dataloader for this normalization
        loader_params = loader_params_base.copy()
        norm_params = norm.copy()
        if "name" in norm_params: del norm_params["name"]
        loader_params["preprocessing"] = norm_params
        
        # We need to figure out what features the methods expect FIRST to decide on data loading
        all_models = [] # List of dicts
        
        for res_dir in args.results_dirs:
            norm_dir = os.path.join(res_dir, ds_name, norm_name)
            if not os.path.exists(norm_dir):
                print(f"Directory {norm_dir} not found. Skipping.")
                continue
                
            methods_in_dir = [d for d in os.listdir(norm_dir) if os.path.isdir(os.path.join(norm_dir, d))]
            
            # Read summary_results.csv to get val metric
            summary_path = os.path.join(norm_dir, "summary_results.csv")
            df_summary = None
            if os.path.exists(summary_path):
                df_summary = pd.read_csv(summary_path)
                
            for method in methods_in_dir:
                unique_name = f"{os.path.basename(os.path.normpath(res_dir))}_{method}"
                val_metric = 0.0
                metric_col = f"val_{args.metric}"
                
                if df_summary is not None and metric_col in df_summary.columns and 'method' in df_summary.columns:
                    row = df_summary[df_summary['method'] == method]
                    if not row.empty:
                        val_metric = row.iloc[0][metric_col]
                else:
                    print(f"Warning: {metric_col} not found for {method} in {summary_path}. Defaulting to 0.0")
                    
                all_models.append({
                    "unique_name": unique_name,
                    "method_cls_name": method,
                    "model_dir_path": os.path.join(norm_dir, method),
                    "val_metric": val_metric
                })
        
        if not all_models:
            print(f"No valid models found for {norm_name}. Skipping.")
            continue
            
        # Sort by validation metric descending
        all_models.sort(key=lambda x: x["val_metric"], reverse=True)
        
        # Drop worst methods if requested
        if args.drop_n_worst > 0:
            if len(all_models) > args.drop_n_worst:
                keep_n = len(all_models) - args.drop_n_worst
                dropped = all_models[keep_n:]
                all_models = all_models[:keep_n]
                dropped_names = [m["unique_name"] for m in dropped]
                print(f"Dropped {args.drop_n_worst} worst models: {dropped_names}")
            else:
                print(f"Warning: --drop-n-worst={args.drop_n_worst} is >= the number of available models ({len(all_models)}). Keeping all.")
                
        expected_features = None
        expected_feature_names = None
        
        if all_models:
            first_model_dir = all_models[0]["model_dir_path"]
            if os.path.exists(first_model_dir):
                # Try to extract expected features from model meta if saved by base_method
                meta_path = os.path.join(first_model_dir, "model_meta.pkl")
                if os.path.exists(meta_path):
                    import joblib
                    try:
                        meta = joblib.load(meta_path)
                        expected_features = meta.get("n_features_in_")
                        expected_feature_names = meta.get("feature_names_in_")
                    except Exception:
                        pass
                else:
                    # Fallback for older scikit-learn base methods saved as model.pkl
                    first_model_path = os.path.join(first_model_dir, "model.pkl")
                    if os.path.exists(first_model_path):
                        import joblib
                        try:
                            m = joblib.load(first_model_path)
                            expected_features = getattr(m, "n_features_in_", None)
                            expected_feature_names = getattr(m, "feature_names_in_", None)
                        except Exception:
                            pass
                    
        # Load Data Normally First
        print("Loading data...")
        dataloader_cls = Registry.get_dataloader(loader_name)
        dataloader = dataloader_cls({"params": loader_params})
        
        try:
            dataloader.load_data()
            val_data = dataloader.get_val_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            continue
            
        if val_data is None:
            print(f"No validation data found for {norm_name}. Skipping.")
            continue
            
        X_val, y_val = val_data
        
        # Check alignment
        if expected_features is not None and expected_features != X_val.shape[1]:
            print(f"Model expects {expected_features} features but data has {X_val.shape[1]}. Attempting feature alignment...")
             
            # Reload with no dropped columns
            loader_params_no_drop = loader_params.copy()
            loader_params_no_drop["drop_columns"] = []
            dataloader_no_drop = dataloader_cls({"params": loader_params_no_drop})
            
            try:
                dataloader_no_drop.load_data()
                X_val_all, y_val_all = dataloader_no_drop.get_val_data()
                
                if expected_feature_names is not None and hasattr(dataloader_no_drop, "feature_names"):
                     try:
                         available_features = dataloader_no_drop.feature_names
                         indices = [available_features.index(f) for f in expected_feature_names]
                         print(f"Aligned features to match {len(indices)} expected features.")
                         X_val = X_val_all[:, indices]
                         y_val = y_val_all
                     except ValueError as e:
                         print(f"Warning: Could not perfectly align features: {e}. Predictions might fail.")
                else:
                     print(f"Info: No feature names available. Slicing to {expected_features} features.")
                     if X_val_all.shape[1] >= expected_features:
                         X_val = X_val_all[:, :expected_features]
                         y_val = y_val_all
                     else:
                         print("Cannot slice due to not enough features.")
            except Exception as e:
                 print(f"Error reloading data for alignment: {e}")
                
        val_predictions = {}
        val_probas = {}
        valid_models = []
        
        for model_info in all_models:
            unique_name = model_info["unique_name"]
            method_cls_name = model_info["method_cls_name"]
            model_dir = model_info["model_dir_path"]
            
            # Predict
            try:
                method_cls = Registry.get_method(method_cls_name)
                # Need a bare-bones initialization just to hold the model
                method = method_cls({"name": method_cls_name, "params": {}})
                
                # Use the method's own load function directly without enforcing a specific model.pkl file check
                # Note: `load` expects a file path in some, but the base_method might add the filename. We handle standard load
                method.load(os.path.join(model_dir, "model.pkl")) # Kept for older methods
            except FileNotFoundError:
                 try:
                     # Some methods like MLP might override load to take a directory or look for different files
                     method.load(model_dir)
                 except Exception as e:
                     print(f"Failed to load model from {model_dir}: {e}")
                     continue
            except Exception as e:
                print(f"Failed to load model {method_cls_name} from {model_dir}: {e}")
                continue
                
            try:
                print(f"Evaluating {unique_name}...")
                preds = method.predict((X_val, y_val))
                val_predictions[unique_name] = preds
                
                if hasattr(method.model, "predict_proba"):
                    val_probas[unique_name] = method.model.predict_proba(X_val)
                    
                valid_models.append(model_info)
            except Exception as e:
                print(f"Failed to predict with {unique_name}: {e}")
                
        if len(val_predictions) == 0:
            print(f"No successful predictions for {norm_name}. Skipping ensemble.")
            continue
            
        print("Computing ensembles... (Soft voting and Stacking will be skipped if probabilities are not available)")
        
        modes_to_run = {}
        
        # 1. Top-K Evaluation
        top_k_order = [m["unique_name"] for m in valid_models]
        modes_to_run['Top-K'] = top_k_order
        
        # 2. Diversity Evaluation
        if args.diversity_mode and len(valid_models) > 1:
            diverse_order = [valid_models[0]["unique_name"]]
            remaining = [m["unique_name"] for m in valid_models[1:]]
            
            while remaining:
                best_candidate = None
                lowest_agreement = float('inf')
                
                for cand in remaining:
                    agreements = []
                    for sel in diverse_order:
                        # compute disagreement/agreement between hard predictions
                        agmt = accuracy_score(val_predictions[sel], val_predictions[cand])
                        agreements.append(agmt)
                    avg_agmt = np.mean(agreements)
                    
                    if avg_agmt < lowest_agreement:
                        lowest_agreement = avg_agmt
                        best_candidate = cand
                        
                diverse_order.append(best_candidate)
                remaining.remove(best_candidate)
                
            modes_to_run['Diverse-K'] = diverse_order

        plot_results = defaultdict(dict) # Plot metric per mode and k
        
        norm_ensemble_results = []
        for mode_name, order in modes_to_run.items():
            print(f"\n--- {mode_name} Ensemble ---")
            for k in range(1, len(order) + 1):
                subset_names = order[:k]
                subset_preds = {n: val_predictions[n] for n in subset_names}
                subset_probas = {n: val_probas[n] for n in subset_names if n in val_probas}
                
                metrics = compute_ensembles(y_val, subset_preds, subset_probas)
                metrics["normalization"] = norm_name
                metrics["datasource"] = ds_name
                metrics["mode"] = mode_name
                metrics["k_models"] = k
                metrics["models_list"] = "+".join(subset_names)
                
                norm_ensemble_results.append(metrics)
                all_ensemble_results.append(metrics)
                
                # Pick metric to plot
                metric_key_soft = f"soft_voting_{args.metric}"
                metric_key_hard = f"hard_voting_{args.metric}"
                
                val_plot = None
                if metric_key_soft in metrics:
                    val_plot = metrics[metric_key_soft]
                elif metric_key_hard in metrics:
                    val_plot = metrics[metric_key_hard]
                
                if val_plot is not None:
                    plot_results[mode_name][k] = val_plot
                    
                print(f"k={k:2d} | Soft {args.metric}: {metrics.get(metric_key_soft, 0.0):.4f} | Hard {args.metric}: {metrics.get(metric_key_hard, 0.0):.4f} | Added: {subset_names[-1]}")
                
        # Generate Plot
        out_dir = args.output_dir if args.output_dir else (args.results_dirs[0] if args.results_dirs else None)
        if plot_results and out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plot_ensemble_performance(plot_results, out_dir, ds_name, norm_name, f"Ensemble {args.metric.capitalize()}", args.diversity_mode)

    if all_ensemble_results:
        df = pd.DataFrame(all_ensemble_results)
        
        # Find best constellation
        best_metric_col = f"soft_voting_{args.metric}"
        if best_metric_col not in df.columns:
            best_metric_col = f"hard_voting_{args.metric}"
            
        if best_metric_col in df.columns:
            # Get row with max value for the chosen metric
            best_row = df.loc[df[best_metric_col].idxmax()]
            
            print("\n" + "="*60)
            print("🏆 BEST ENSEMBLE CONSTELLATION 🏆".center(60))
            print("="*60)
            print(f"Mode:          {best_row['mode']}")
            print(f"Num Models:    {best_row['k_models']}")
            
            soft_val = best_row.get(f'soft_voting_{args.metric}', None)
            hard_val = best_row.get(f'hard_voting_{args.metric}', None)
            if pd.notna(soft_val):
                print(f"Soft {args.metric}: {soft_val:.4f}")
            if pd.notna(hard_val):
                print(f"Hard {args.metric}: {hard_val:.4f}")
                
            print(f"\nModels Included:")
            for m in str(best_row['models_list']).split('+'):
                print(f"  - {m}")
            print("="*60 + "\n")

        out_dir = args.output_dir if args.output_dir else (args.results_dirs[0] if args.results_dirs else None)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            diversity_str = "_diverse" if args.diversity_mode else ""
            out_csv = os.path.join(out_dir, f"ensemble_summary_{args.metric}{diversity_str}.csv")
            df.to_csv(out_csv, index=False)
            print(f"Saved ensemble summary to {out_csv}")

if __name__ == "__main__":
    main()
