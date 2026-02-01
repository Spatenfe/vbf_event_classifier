import pandas as pd
import numpy as np

class DataBalancer:
    """
    Component for balancing datasets at runtime.
    """
    @staticmethod
    def balance(X, y, strategy="undersample", random_state=42):
        if strategy not in ["undersample", "oversample"]:
            if strategy is True: # Backwards compatibility
                strategy = "undersample"
            else:
                return X, y
            
        # Convert X to DataFrame if it's a numpy array to handle indices easily
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
            
        y_series = pd.Series(y)
        counts = y_series.value_counts()
        
        np.random.seed(random_state)
        balanced_indices = []
        
        if strategy == "undersample":
            target_count = counts.min()
            replace = False
        else: # oversample
            target_count = counts.max()
            replace = True
            
        for label in counts.index:
            label_indices = y_series[y_series == label].index
            # For undersample, we pick min_count without replacement
            # For oversample, we pick max_count with replacement
            selected_indices = np.random.choice(label_indices, target_count, replace=replace)
            balanced_indices.extend(selected_indices)
            
        # Select balanced data
        X_balanced = X_df.iloc[balanced_indices]
        y_balanced = y_series.iloc[balanced_indices]
        
        # Shuffle result
        shuffle_idx = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced.iloc[shuffle_idx]
        y_balanced = y_balanced.iloc[shuffle_idx]
        
        if is_numpy:
            return X_balanced.values, y_balanced.values
        return X_balanced, y_balanced
