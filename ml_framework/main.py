import json
import logging
import argparse
import sys
from data_processor import DataProcessor
from model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ml_framework.log"),
    ],
)


def main():
    parser = argparse.ArgumentParser(
        description="Classical ML Classification Framework"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ml_framework/config.json",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        return

    logging.info("Starting ML Framework run...")

    # 1. Data Preparation
    try:
        processor = DataProcessor(config)
        processor.load_data()
        processor.preprocess()
        X_train, X_test, y_train, y_test = processor.get_data()
    except Exception as e:
        logging.error(f"Data processing failed: {e}")
        return

    # 2. Model Training & Evaluation
    try:
        manager = ModelManager(config)
        manager.train_and_evaluate(X_train, X_test, y_train, y_test)
        manager.save_results()
        manager.visualize_results()
    except Exception as e:
        logging.error(f"Model execution failed: {e}")
        return

    logging.info("ML Framework run completed successfully.")


if __name__ == "__main__":
    main()
