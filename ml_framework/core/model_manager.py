import importlib
import logging
import json
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ModelManager:
    def __init__(self, config):
        self.config = config
        self.results = {}

    def get_algorithm(self, alg_config):
        try:
            module = importlib.import_module(alg_config["module"])
            model_class = getattr(module, alg_config["class"])
            return model_class(**alg_config.get("params", {}))
        except Exception as e:
            logging.error(f"Error loading algorithm {alg_config['name']}: {e}")
            return None

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        for alg_config in self.config["algorithms"]:
            alg_name = alg_config["name"]
            logging.info(f"Training {alg_name}...")

            model = self.get_algorithm(alg_config)
            if model is None:
                continue

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred)

                self.results[alg_name] = {
                    "accuracy": acc,
                    "classification_report": report,
                    "confusion_matrix": cm.tolist(),
                    "params": alg_config.get("params", {}),
                }

                logging.info(f"{alg_name} finished. Accuracy: {acc:.4f}")

            except Exception as e:
                logging.error(f"Error training {alg_name}: {e}")

    def save_results(self):
        output_dir = self.config.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON
        json_path = os.path.join(output_dir, "run_results.json")
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=4)
        logging.info(f"Results saved to {json_path}")

    def visualize_results(self):
        output_dir = self.config.get("output_dir", "results")

        # Accuracy comparison
        alg_names = list(self.results.keys())
        accuracies = [res["accuracy"] for res in self.results.values()]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=alg_names, y=accuracies)
        plt.title("Algorithm Comparison - Accuracy")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
        plt.close()

        # Confusion Matrices
        for alg_name, res in self.results.items():
            cm = np.array(res["confusion_matrix"])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {alg_name}")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"confusion_matrix_{alg_name}.png"))
            plt.close()

        logging.info(f"Visualizations saved to {output_dir}")
