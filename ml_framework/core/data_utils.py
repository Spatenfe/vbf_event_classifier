import pandas as pd
import numpy as np

class DataBalancer:
    """
    Component for balancing datasets at runtime.
    """
    @staticmethod
    def balance(X, y, strategy="undersample", oversample_factor=None, random_state=42):
        if strategy not in ["undersample", "oversample"]:
            if strategy is True: # Backwards compatibility
                strategy = "undersample"
            else:
                return X, y
            
        is_numpy = isinstance(X, np.ndarray) # Define is_numpy at the beginning
        
        # Convert X to DataFrame if it's a numpy array to handle indices easily
        if is_numpy:
            X_df = pd.DataFrame(X)
            y_series = pd.Series(y)
        else:
            X_df = X.reset_index(drop=True)
            y_series = pd.Series(y).reset_index(drop=True)
        counts = y_series.value_counts()
        
        np.random.seed(random_state)
        balanced_indices = []
        
        if strategy == "undersample":
            target_count = counts.min()
        else: # oversample
            if oversample_factor is not None:
                # Limit oversampling based on the minority class size
                target_count = int(counts.min() * oversample_factor)
                # Ensure we don't go below the minority class size (though logically implied)
                target_count = max(target_count, counts.min())
            else:
                target_count = counts.max()
            
        for label in counts.index:
            label_indices = y_series[y_series == label].index
            current_count = len(label_indices)
            
            # Determine replacement strategy and count
            if current_count < target_count:
                # Need to oversample
                replace = True
                count_to_sample = target_count
            else:
                # Need to undersample or keep as is (but we balance to target_count)
                # If existing count is greater than target, we undersample down to target
                replace = False
                count_to_sample = target_count

            # Should we just always sample 'target_count'? 
            # If strategy is "undersample", target_count is min(), so everyone gets sampled down to min without replacement.
            # If strategy is "oversample" (without factor), target_count is max(), so small classes get sampled up with replacement, big classes stay same (sampled 100% without replacement effectively if count == target)
            # If strategy is "oversample" (with factor), target_count might be in between.
            #   - Small class (count < target): oversample with replacement.
            #   - Large class (count > target): undersample without replacement.
            
            selected_indices = np.random.choice(label_indices, count_to_sample, replace=replace)
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

    @staticmethod
    def merge_and_balance_classes(df, target_col, merge_config, random_state=42):
        """
        Merges classes based on configuration and balances them so each source class contributes equally.
        Undersamples larger classes to match the size of the smallest class in the merge group.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
            merge_config (list): List of lists, e.g. [['new_label', 'old1', 'old2'], ...]
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Modified dataframe with merged and balanced classes
        """
        if not merge_config:
            return df
            
        # localized import to avoid circular dependency issues if any
        # though standard import is fine here usually
        
        # specific logic
        df = df.copy()
        # convert target to string for consistency
        df[target_col] = df[target_col].astype(str)
        
        np.random.seed(random_state)
        
        indices_to_drop = []
        
        for merge_group in merge_config:
            if not merge_group or len(merge_group) < 2:
                continue
                
            target_label = str(merge_group[0])
            source_labels = [str(ls) for ls in merge_group[1:]]
            
            # Find counts for each source label in this group
            source_counts = {}
            min_count = float('inf')
            
            available_sources = []
            
            for source in source_labels:
                count = len(df[df[target_col] == source])
                if count > 0:
                    source_counts[source] = count
                    if count < min_count:
                        min_count = count
                    available_sources.append(source)
            
            if not available_sources:
                continue
                
            # If min_count remains inf (no matching rows), nothing to do for this group
            if min_count == float('inf'):
                continue
                
            # For each source class, select min_count random indices (if count > min_count)
            # The indices we DO NOT select are to be dropped
            
            for source in available_sources:
                source_indices = df[df[target_col] == source].index
                current_count = len(source_indices)
                
                if current_count > min_count:
                    # Randomly choose which ones to KEEP
                    keep_indices = np.random.choice(source_indices, min_count, replace=False)
                    # The rest must be dropped
                    drop_indices = np.setdiff1d(source_indices, keep_indices)
                    indices_to_drop.extend(drop_indices)
            
            # Now update the label for all rows of these source classes (that remain)
            # We do this after dropping, or we can just update all and let drop happen later.
            # Updating all is safe because dropped rows won't matter.
            df.loc[df[target_col].isin(source_labels), target_col] = target_label

        if indices_to_drop:
            # removing duplicates just in case
            unique_drop = np.unique(indices_to_drop)
            df.drop(index=unique_drop, inplace=True)
            
        return df
