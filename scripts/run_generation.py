"""
Generate synthetic minority-class samples using a configured generative model.

Usage:
    python scripts/run_generation.py configs/generation/minority_vae.json
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

# Make sure the project root is on the path regardless of where the script is called from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ml_framework.generators  # noqa: F401 — triggers Registry registration
from ml_framework.core.registry import Registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic minority-class samples.")
    parser.add_argument("config", help="Path to the generation config JSON file.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    generation_name = cfg.get("generation_name", "generation")
    data_path       = cfg["data_path"]
    target_column   = cfg.get("target_column", "cvv")
    target_class    = str(cfg["target_class"])
    drop_columns    = cfg.get("drop_columns", [])
    n_samples       = cfg.get("n_samples", 1000)
    output_path     = cfg["output_path"]
    save_model      = cfg.get("save_model", False)
    model_dir       = cfg.get("model_dir", f"results/generation/{generation_name}/")
    generator_cfg   = cfg["generator"]

    logger.info("=== Generation: %s ===", generation_name)

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors="ignore")
        logger.info("Dropped %d columns. Remaining features: %d", len(drop_columns), df.shape[1] - 1)

    # Filter to target class
    mask = df[target_column].astype(str) == target_class
    df_minority = df[mask].copy()
    logger.info(
        "Target class '%s': %d samples out of %d total.",
        target_class, len(df_minority), len(df),
    )

    if len(df_minority) == 0:
        logger.error("No samples found for target class '%s'. Aborting.", target_class)
        sys.exit(1)

    feature_cols  = [c for c in df_minority.columns if c != target_column]
    X             = df_minority[feature_cols].values.astype(np.float32)
    feature_names = feature_cols

    # ── Build generator ───────────────────────────────────────────────────────
    gen_name = generator_cfg["name"]
    gen_cls  = Registry.get_generator(gen_name)

    default_cfg   = gen_cls.get_default_config()
    merged_params = default_cfg.get("params", {}).copy()
    merged_params.update(generator_cfg.get("params", {}))
    full_cfg = {"name": gen_name, "params": merged_params}

    generator = gen_cls(full_cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    generator.fit(X, feature_names=feature_names)

    # ── Generate ──────────────────────────────────────────────────────────────
    logger.info("Generating %d synthetic samples...", n_samples)
    X_gen = generator.generate(n_samples)

    df_gen = pd.DataFrame(X_gen, columns=feature_names)
    df_gen[target_column] = target_class

    # ── Save output ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_gen.to_csv(output_path, index=False)
    logger.info("Saved %d generated samples to %s", len(df_gen), output_path)

    if save_model:
        generator.save(model_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
