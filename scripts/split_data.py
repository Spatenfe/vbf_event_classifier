import pandas as pd
import argparse
import os
import numpy as np

def split_data(input_path, output_dir, val_size=0.2, drop_labels=None, target_col="cvv", balance_train=False):
    if drop_labels is None:
        drop_labels = []
        
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Drop specified labels
    if drop_labels:
        print(f"Dropping labels: {drop_labels}")
        df = df[~df[target_col].isin(drop_labels)].copy()
        
    classes = df[target_col].unique()
    n_classes = len(classes)
    total_samples = len(df)
    
    target_val_samples = int(total_samples * val_size)
    n_val_per_class = int(target_val_samples / n_classes)
    
    print(f"Total samples: {total_samples}")
    print(f"Classes: {classes}")
    print(f"Target validation total: {target_val_samples}")
    print(f"Target validation per class: {n_val_per_class}")
    
    # Check feasibility for validation
    class_counts = df[target_col].value_counts()
    min_class_count = class_counts.min()
    
    if n_val_per_class >= min_class_count:
        print(f"Warning: Calculated validation samples per class ({n_val_per_class}) exceeds or equals the size of the smallest class ({min_class_count}).")
        # Adjusting to 50% of smallest class to ensure at least some train data
        n_val_per_class = int(min_class_count * 0.5)
        print(f"Adjusted validation samples per class to: {n_val_per_class}")
        
    val_indices = []
    
    for cls in classes:
        cls_indices = df[df[target_col] == cls].index
        # Randomly select n_val_per_class indices
        selected_indices = np.random.choice(cls_indices, n_val_per_class, replace=False)
        val_indices.extend(selected_indices)
        
    val_df = df.loc[val_indices].copy()
    train_df = df.drop(index=val_indices).copy()
    
    if balance_train:
        print("Balancing training data...")
        train_counts = train_df[target_col].value_counts()
        min_train_count = train_counts.min()
        print(f"Smallest class in training set has {min_train_count} samples. Downsampling all classes to this size.")
        
        balanced_train_indices = []
        for cls in classes:
            cls_train_indices = train_df[train_df[target_col] == cls].index
            selected_train_indices = np.random.choice(cls_train_indices, min_train_count, replace=False)
            balanced_train_indices.extend(selected_train_indices)
        
        train_df = train_df.loc[balanced_train_indices].copy()
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_out = os.path.join(output_dir, "train_balanced.csv")
    val_out = os.path.join(output_dir, "val_balanced.csv")
    
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    
    print(f"Saved training data to {train_out}: {len(train_df)} samples")
    print(f"Saved validation data to {val_out}: {len(val_df)} samples")
    
    # Verify Balance
    print("\nTraining Set Distribution:")
    print(train_df[target_col].value_counts())
    print("\nValidation Set Distribution:")
    print(val_df[target_col].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CSV into train and balanced validation sets.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output CSVs")
    parser.add_argument("--val-size", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--drop-labels", type=float, nargs="*", default=[1.0], help="Labels to drop")
    parser.add_argument("--target", type=str, default="cvv", help="Target column name")
    parser.add_argument("--balance-train", action="store_true", help="Balance the training set by downsampling to the smallest class size.")
    
    args = parser.parse_args()
    
    split_data(args.input, args.output_dir, args.val_size, args.drop_labels, args.target, args.balance_train)
