# Classification Framework

A modular framework for evaluating various machine learning classification algorithms across different data distributions and normalization strategies.

## Overview

This project provides a structured environment to benchmark multiple classification methods (Random Forest, Gradient Boosting, MLP, etc.) against various datasets and preprocessing techniques. It supports parallel execution, runtime data balancing, and automatic generation of comparison results.

## Features

- **Modular Architecture**: Easy to add new algorithms and dataloaders via a registry system.
- **Cartesian Product Experiments**: Run experiments across all combinations of datasources and normalization strategies from a single JSON config.
- **Parallel Execution**: Support for running multiple models in parallel using the `--n-jobs` flag.
- **Runtime Data Balancing**: Optional undersampling or oversampling of training data to handle class imbalance without modifying source files.
- **Automated Reporting**: Generates summary CSVs, confusion matrices, and accuracy overview plots for every experiment run.
- **Progress Tracking**: Clear terminal progress bars using tqdm for long-running benchmarks.

## Project Structure

- `ml_framework/core/`: Core logic (runner, metrics, registry, plotting).
- `ml_framework/methods/`: Implementation of classification algorithms.
- `ml_framework/dataloaders/`: Data loading and preprocessing logic.
- `scripts/`: Entry point scripts for data splitting and running experiments.
- `results/`: Default output directory for experiment metadata and plots.

## Usage

### 1. Data Preparation

Use `scripts/split_data.py` to create balanced validation sets from raw CSV files.

```bash
python scripts/split_data.py --input path/to/raw_data.csv --output-dir ml_framework/data/my_dataset --balance-train
```

### 2. Running Experiments

Experiments are defined in JSON files. Run them using `scripts/run_experiment.py`.

```bash
python scripts/run_experiment.py --config benchmark_config.json --n-jobs 4
```

### 3. Adding New Components

#### New Methods
Create a new directory in `ml_framework/methods/` with:
- `model.py`: Inherit from `BaseAlgorithm` and use `@Registry.register_method`.
- `config.json`: Default parameters for the method.
- `__init__.py`: Import the model class to ensure registration.

#### New Dataloaders
Create a new file in `ml_framework/dataloaders/` inheriting from `BaseDataloader` and register it using `@Registry.register_dataloader`.

## Configuration

Example `datasource` configuration with runtime balancing:

```json
{
    "name": "my_data",
    "loader": "standard_loader",
    "train_path": "data/train.csv",
    "val_path": "data/val.csv",
    "balance_train": "undersample"
}
```

Available balancing strategies: `"undersample"`, `"oversample"`, or `null` (none).
