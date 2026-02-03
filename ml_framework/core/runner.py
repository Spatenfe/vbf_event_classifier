import os
import logging
import importlib
import warnings
from datetime import datetime
import time
import pandas as pd
from .registry import Registry
from .metrics import calculate_metrics
from .plotting import plot_comparison, plot_confusion_matrix
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


_logger = logging.getLogger(__name__)

class _TqdmLoggingHandler(logging.Handler):
    """Logging handler that plays nicely with tqdm progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            # Never let logging crash the run.
            pass


# If this module is used from a script that doesn't configure logging,
# ensure INFO logs are visible and don't break tqdm progress output.
if not logging.getLogger().handlers:
    _handler = _TqdmLoggingHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    _root = logging.getLogger()
    _root.setLevel(logging.INFO)
    _root.addHandler(_handler)


def _shorten_repr(value, max_len: int = 500) -> str:
    text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _append_line(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _get_registered_method_cls(method_name: str):
    """Get method class, importing its module if needed.

    When running methods in parallel, each worker is a fresh process and the
    Registry starts empty unless methods are imported in that process.
    """
    try:
        return Registry.get_method(method_name)
    except ValueError:
        # Best-effort: method names map to ml_framework.methods.<method_name> packages.
        importlib.import_module(f"ml_framework.methods.{method_name}")
        return Registry.get_method(method_name)

def _run_single_method(
    method_config,
    train_data,
    val_data,
    test_data,
    output_dir,
    test_mode=False,
    method_n_jobs=None,
    emit_console_logs=True,
):
    """Worker function for parallel method execution."""
    if isinstance(method_config, str):
        method_name = method_config
        user_params = {}
    else:
        method_name = method_config["name"]
        user_params = method_config.get("params", {})

    method_output_dir = os.path.join(output_dir, method_name)
    os.makedirs(method_output_dir, exist_ok=True)
    run_log_path = os.path.join(method_output_dir, "run.log")

    try:
        # Optional global default for per-method threading.
        # If a specific method config sets params.n_jobs, it wins.
        if method_n_jobs is not None and isinstance(user_params, dict) and "n_jobs" not in user_params:
            user_params = {**user_params, "n_jobs": method_n_jobs}

        method_cls = _get_registered_method_cls(method_name)
        default_config = method_cls.get_default_config()
        full_params = default_config.get("params", {}).copy()
        full_params.update(user_params)
        
        config = {"name": method_name, "params": full_params}
        method = method_cls(config)

        start_msg = (
            f"Starting method '{method_name}' ({getattr(method_cls, '__name__', type(method).__name__)})"
            f" | val={val_data is not None} test={test_data is not None} | params={_shorten_repr(full_params)}"
        )
        if emit_console_logs:
            _logger.info(start_msg)
        else:
            _append_line(run_log_path, f"{datetime.now().isoformat(timespec='seconds')} - INFO - {start_msg}")

        # Keep warnings from breaking tqdm output; route to console logger or run.log.
        def _showwarning(message, category, filename, lineno, file=None, line=None):
            text = warnings.formatwarning(message, category, filename, lineno, line)
            if emit_console_logs:
                _logger.warning(text.rstrip("\n"))
            else:
                for ln in text.rstrip("\n").splitlines():
                    _append_line(run_log_path, f"{datetime.now().isoformat(timespec='seconds')} - WARNING - {ln}")

        with warnings.catch_warnings():
            # Suppress a noisy sklearn warning (n_jobs is deprecated/ignored in LogisticRegression in newer versions).
            warnings.filterwarnings(
                "ignore",
                message=r".*'n_jobs' has no effect since 1\.8.*",
                category=FutureWarning,
            )
            warnings.showwarning = _showwarning

            # Train
            method.train(train_data)

            combined_metrics = {'method': method_name}

            # Evaluate on Validation Set
            if val_data is not None:
                X_val, y_val = val_data
                val_predictions = method.predict(val_data)
                val_output_dir = os.path.join(method_output_dir, "val")
                val_metrics = calculate_metrics(y_val, val_predictions, output_dir=val_output_dir)
                plot_confusion_matrix(y_val, val_predictions, output_dir=val_output_dir)
                for k, v in val_metrics.items():
                    combined_metrics[f"val_{k}"] = v

            # Evaluate on Test Set
            if test_data is not None:
                X_test, y_test = test_data
                test_predictions = method.predict(test_data)
                test_output_dir = os.path.join(method_output_dir, "test")
                test_metrics = calculate_metrics(y_test, test_predictions, output_dir=test_output_dir)
                plot_confusion_matrix(y_test, test_predictions, output_dir=test_output_dir)
                for k, v in test_metrics.items():
                    combined_metrics[f"test_{k}"] = v
                
                # Backward compatibility for 'accuracy' column in summary
                combined_metrics['accuracy'] = test_metrics.get('accuracy')
                combined_metrics['f1_macro'] = test_metrics.get('f1_macro')

                if test_mode and emit_console_logs:
                    tqdm.write(f"\n--- Test Results for {method_name} ---")
                    y_test_list = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
                    for pred, exp in zip(test_predictions, y_test_list):
                        tqdm.write(f"Predicted: {pred}, Expected: {exp}")
                    tqdm.write("----------------------------\n")

            return combined_metrics
    except Exception:
        import traceback
        try:
            with open(os.path.join(method_output_dir, "error.txt"), "w") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        raise

class ExperimentRunner:
    def __init__(
        self,
        dataloader_config,
        method_configs,
        output_dir="results",
        n_jobs=1,
        method_n_jobs=None,
    ):
        self.output_dir = output_dir
        self.n_jobs = n_jobs
        self.method_n_jobs = method_n_jobs
        
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
        val_data = self.dataloader.get_val_data()
        test_data = self.dataloader.get_test_data()
        test_mode = self.dataloader.has_test_set()
        
        os.makedirs(self.output_dir, exist_ok=True)

        if self.n_jobs > 1:
            tqdm.write(f"Running methods in parallel (n_jobs={self.n_jobs})...")
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {}
                start_times = {}
                for m_cfg in self.method_configs:
                    m_name = m_cfg if isinstance(m_cfg, str) else m_cfg.get("name")
                    tqdm.write(f"Submitting method: {m_name}")
                    future = executor.submit(
                        _run_single_method,
                        m_cfg,
                        train_data,
                        val_data,
                        test_data,
                        self.output_dir,
                        test_mode,
                        self.method_n_jobs,
                        False,
                    )
                    futures[future] = m_cfg
                    start_times[future] = time.monotonic()
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Methods", leave=False):
                    try:
                        metrics = future.result()
                        self.results.append(metrics)
                        elapsed = time.monotonic() - start_times.get(future, time.monotonic())
                        tqdm.write(f"  Finished {metrics['method']} after {elapsed:.2f}s.")
                    except Exception as e:
                        import traceback
                        m_cfg = futures[future]
                        m_name = m_cfg if isinstance(m_cfg, str) else m_cfg.get('name')
                        tqdm.write(f"  Error running {m_name}: {e}")
                        tqdm.write(traceback.format_exc())
        else:
            for method_config in tqdm(self.method_configs, desc="Methods", leave=False):
                method_name = method_config if isinstance(method_config, str) else method_config["name"]
                tqdm.write(f"Running method: {method_name}")
                try:
                    t0 = time.monotonic()
                    metrics = _run_single_method(
                        method_config,
                        train_data,
                        val_data,
                        test_data,
                        self.output_dir,
                        test_mode,
                        self.method_n_jobs,
                    )
                    self.results.append(metrics)
                    elapsed = time.monotonic() - t0
                    tqdm.write(f"  Finished {method_name} after {elapsed:.2f}s.")
                except Exception as e:
                    tqdm.write(f"  Error running {method_name}: {e}")
                    # Optionally log the traceback
                    import traceback
                    tqdm.write(traceback.format_exc())

        # Generate summary
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(os.path.join(self.output_dir, "summary_results.csv"), index=False)
            
            plot_comparison(results_df, metric="accuracy", output_dir=self.output_dir)
            plot_comparison(results_df, metric="f1_macro", output_dir=self.output_dir)
        
        tqdm.write(f"Combination {self.dataloader_name} finished.")
