# VBF Event Classifier

A modular ML experimentation framework for benchmarking classification methods on Vector Boson Fusion (VBF) event datasets from particle physics simulations. It supports parallel execution, runtime data balancing, synthetic minority-class data generation, and automated reporting.

## Features

- **Modular registry system** — add new algorithms, dataloaders, and generators without touching core code
- **Cartesian-product experiments** — run all combinations of datasources × normalisation strategies from a single JSON config
- **Parallel execution** — run multiple methods simultaneously with `--n-jobs`
- **Class imbalance handling** — runtime oversampling/undersampling, Gaussian noise augmentation, and VAE-based minority class generation
- **Two-stage training** — pre-train on real + synthetic data, then fine-tune on clean data with a lower learning rate
- **PyTorch MLP methods** — residual connections, batch norm, SE blocks, focal loss, LR scheduling, warmup
- **Automated reporting** — summary CSVs, confusion matrices, accuracy/F1 comparison plots, method-agreement heatmaps

---

## Installation

```bash
# Clone and install in editable mode
git clone <repo-url>
cd vbf_event_classifier
pip install -e .

# Or with uv
uv sync
```

**Requirements:** Python ≥ 3.9, PyTorch (CPU or CUDA).

---

## Quick Start

```bash
# Run a benchmarking experiment
python scripts/run_experiment.py configs/ablation/all_std.json

# Run with 4 parallel methods
python scripts/run_experiment.py configs/ablation/all_std.json --n-jobs 4

# Generate synthetic minority-class samples with a VAE
python scripts/run_generation.py configs/generation/minority_vae.json
```

Results are written to the directory specified in the config's `output.dir` field.

---

## Project Structure

```
vbf_event_classifier/
├── configs/
│   ├── ablation/          # Experiment configs
│   └── generation/        # Data generation configs
├── ml_framework/
│   ├── core/              # Runner, registry, metrics, plotting
│   ├── dataloaders/       # CSV loaders with preprocessing
│   ├── generators/        # Generative models (VAE)
│   └── methods/           # Classification algorithms
│       ├── mlp_classifier/
│       ├── mlp_residual_classifier/
│       ├── random_forest/
│       └── ...            # sklearn-based methods
└── scripts/
    ├── run_experiment.py      # Main entry point
    ├── run_generation.py      # Synthetic data generation
    ├── analyze_dataset.py     # Dataset exploration
    ├── hyperparameter_search.py
    └── plot_summary.py
```

---

## Running Experiments

Experiments are defined as JSON config files:

```bash
python scripts/run_experiment.py <config.json> [--n-jobs N]
```

`--n-jobs` controls process-level parallelism (one process per method). Threading inside a method is controlled by `"n_jobs"` in that method's `params`.

---

## Configuration Reference

### Top-level keys

| Key | Type | Description |
|-----|------|-------------|
| `experiment_name` | string | Used as the output subdirectory name |
| `datasource` | list | One or more data source configs (see below) |
| `normalization` | list | One or more scaling/normalisation configs |
| `methods` | list | Method names or `{name, alias, params}` dicts |
| `method_n_jobs` | int | Default thread count for all methods |
| `save_models` | bool | Persist trained models to disk |
| `output.dir` | string | Root directory for results |

### Datasource config

```json
{
    "name": "my_data",
    "loader": "standard_loader",
    "data_path": "path/to/data.csv",
    "target_column": "label",
    "val_split": 0.2,
    "balance_val": true,
    "drop_columns": ["id", "irrelevant_col"],
    "merge_classes": [[0, 0.5, 1.5]],
    "discard_classes": [99],
    "balance_train": "oversample",
    "oversample_factor": 3.0,
    "augment_path": "path/to/synthetic.csv",
    "augmentation": {
        "noise_std": 0.01,
        "minority_only": true
    }
}
```

`balance_train` accepts `"oversample"`, `"undersample"`, or `false`.
`augment_path` appends a pre-generated synthetic CSV to the training split.
`augmentation.noise_std` applies relative Gaussian noise (`x *= 1 + N(0, σ)`) to training features.

### Normalization config

```json
{ "name": "std", "scaling": "standard" }
```

`scaling` options: `standard`, `minmax`, `robust`, `maxabs`, `yeo-johnson`, `quantile_normal`, `quantile_uniform`.

### Method config

Methods can be specified as a name string or a dict with parameter overrides:

```json
{
    "name": "mlp_residual_classifier",
    "alias": "mlp_residual_focal",
    "params": {
        "loss_fn": "focal",
        "warmup_epochs": 10,
        "finetune_epochs": 100,
        "finetune_lr": 0.0005
    }
}
```

Key PyTorch method params: `hidden_layer_sizes`, `activation`, `dropout_rate`, `batch_norm`, `lr_scheduler`, `label_smoothing`, `loss_fn` (`bce` / `weighted_bce` / `focal`), `warmup_epochs`, `finetune_epochs`, `finetune_lr`.

---

## Synthetic Data Generation

Train a VAE on the minority class and generate new samples:

```bash
python scripts/run_generation.py configs/generation/minority_vae.json
```

Generation config keys:

| Key | Description |
|-----|-------------|
| `data_path` | Source CSV |
| `target_column` | Label column name |
| `target_class` | Class value to generate (e.g. `"1.0"`) |
| `drop_columns` | Columns to exclude from generation |
| `n_samples` | Number of samples to generate |
| `output_path` | Where to save the generated CSV |
| `save_model` / `model_dir` | Persist the trained VAE |
| `generator.params` | VAE hyperparameters (`hidden_dims`, `latent_dim`, `epochs`, …) |

The output CSV contains only the kept features plus the target column, and can be passed directly to `augment_path` in an experiment config.

---

## Adding New Components

### New classification method

Create `ml_framework/methods/<name>/`:
- `model.py` — inherit from `BaseAlgorithm` (sklearn) or `PyTorchBaseMethod` (PyTorch), decorate with `@Registry.register_method("<name>")`
- `config.json` — default parameters
- `__init__.py` — import the model class to trigger registration

### New dataloader

Create a file in `ml_framework/dataloaders/`, inherit from `BaseDataloader`, register with `@Registry.register_dataloader("<name>")`.

### New generator

Create `ml_framework/generators/<name>/`, inherit from `BaseGenerator`, register with `@Registry.register_generator("<name>")`. Add the import to `ml_framework/generators/__init__.py`.

---

## License

MIT — see [LICENSE](LICENSE).
