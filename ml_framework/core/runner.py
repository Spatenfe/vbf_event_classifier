import os
import pandas as pd
from .registry import Registry
from .metrics import calculate_metrics
from .plotting import plot_comparison, plot_confusion_matrix
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def _run_single_method(method_config, train_data, test_data, output_dir):
    """Worker function for parallel method execution."""
    if isinstance(method_config, str):
        method_name = method_config
        user_params = {}
    else:
        method_name = method_config["name"]
        user_params = method_config.get("params", {})
    
    method_cls = Registry.get_method(method_name)
    default_config = method_cls.get_default_config()
    full_params = default_config.get("params", {}).copy()
    full_params.update(user_params)
    
    config = {"name": method_name, "params": full_params}
    method = method_cls(config)
    
    method_output_dir = os.path.join(output_dir, method_name)
    os.makedirs(method_output_dir, exist_ok=True)
    
    # Train
    method.train(train_data)
    
    # Predict
    predictions = method.predict(test_data)
    
    # Evaluate
    X_test, y_test = test_data
    metrics = calculate_metrics(y_test, predictions, output_dir=method_output_dir)
    plot_confusion_matrix(y_test, predictions, output_dir=method_output_dir)
    
    metrics['method'] = method_name
    return metrics

class ExperimentRunner:
    def __init__(self, dataloader_config, method_configs, output_dir="results", n_jobs=1):
        self.output_dir = output_dir
        self.n_jobs = n_jobs
        
        # Load Dataloader
        if isinstance(dataloader_config, str):
            dataloader_name = dataloader_config
            dataloader_cls = Registry.get_dataloader(dataloader_name)
            self.dataloader = dataloader_cls({})
        else:
            dataloader_name = dataloader_config["name"]
            dataloader_params = dataloader_config.get("params", {})
            dataloader_cls = Registry.get_dataloader(dataloader_name)
            self.dataloader = dataloader_cls({"params": dataloader_params})
        
        self.dataloader_name = dataloader_name
        self.method_configs = method_configs if isinstance(method_configs, list) else [method_configs]
        self.results = []

    def run(self):
        tqdm.write(f"Loading data using {self.dataloader_name}...")
        self.dataloader.load_data()
        train_data = self.dataloader.get_train_data()
        test_data = self.dataloader.get_test_data()
        
        os.makedirs(self.output_dir, exist_ok=True)

        if self.n_jobs > 1:
            tqdm.write(f"Running methods in parallel (n_jobs={self.n_jobs})...")
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(_run_single_method, m_cfg, train_data, test_data, self.output_dir): m_cfg 
                    for m_cfg in self.method_configs
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Methods", leave=False):
                    try:
                        metrics = future.result()
                        self.results.append(metrics)
                        tqdm.write(f"  Finished {metrics['method']}.")
                    except Exception as e:
                        m_cfg = futures[future]
                        m_name = m_cfg if isinstance(m_cfg, str) else m_cfg.get('name')
                        tqdm.write(f"  Error running {m_name}: {e}")
        else:
            for method_config in tqdm(self.method_configs, desc="Methods", leave=False):
                method_name = method_config if isinstance(method_config, str) else method_config["name"]
                tqdm.write(f"Running method: {method_name}")
                metrics = _run_single_method(method_config, train_data, test_data, self.output_dir)
                self.results.append(metrics)
                tqdm.write(f"  Finished {method_name}.")

        # Generate summary
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(os.path.join(self.output_dir, "summary_results.csv"), index=False)
            
            plot_comparison(results_df, metric="accuracy", output_dir=self.output_dir)
            plot_comparison(results_df, metric="f1_macro", output_dir=self.output_dir)
        
        tqdm.write(f"Combination {self.dataloader_name} finished.")
