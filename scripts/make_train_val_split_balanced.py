import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = 'ml_framework/data/large/matched_events_large.csv'
TRAIN_OUTPUT = 'ml_framework/data/large/train.csv'
VAL_OUTPUT = 'ml_framework/data/large/val.csv'
EXCLUDED_CLASS = 1.0
VAL_SPLIT_RATIO = 0.20
RANDOM_SEED = 42

def main():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    original_count = len(df)
    print(f"Original shape: {df.shape}")

    # 1. Remove excluded class
    print(f"Removing class {EXCLUDED_CLASS}...")
    df_filtered = df[df['cvv'] != EXCLUDED_CLASS].copy()
    filtered_count = len(df_filtered)
    print(f"Filtered shape: {df_filtered.shape} (Removed {original_count - filtered_count} rows)")

    # Check remaining classes
    classes = df_filtered['cvv'].unique()
    print(f"Remaining classes: {sorted(classes)}")
    
    # 2. Calculate validation size
    n_total_filtered = len(df_filtered)
    n_val_total = int(n_total_filtered * VAL_SPLIT_RATIO)
    n_classes = len(classes)
    n_val_per_class = n_val_total // n_classes
    
    print(f"Total filtered samples: {n_total_filtered}")
    print(f"Target validation size: {n_val_total} ({VAL_SPLIT_RATIO*100}%)")
    print(f"Target validation samples per class ({n_classes} classes): {n_val_per_class}")

    # 3. Create split
    val_indices = []
    
    # Group by class
    grouped = df_filtered.groupby('cvv')
    
    for class_label, group in grouped:
        n_available = len(group)
        if n_available < n_val_per_class:
            raise ValueError(f"Not enough samples for class {class_label}. Needed {n_val_per_class}, found {n_available}")
        
        # Sample for validation
        selected_indices = group.sample(n=n_val_per_class, random_state=RANDOM_SEED).index
        val_indices.extend(selected_indices)

    # create val df
    val_df = df.loc[val_indices]
    
    # create train df (all filtered rows NOT in val)
    # Using index difference to ensure no overlap and complete coverage of filtered data
    train_indices = df_filtered.index.difference(val_df.index)
    train_df = df.loc[train_indices]

    # 4. Verification
    print("\n--- Verification ---")
    print(f"Train set shape: {train_df.shape}")
    print(f"Val set shape:   {val_df.shape}")
    
    # Check overlap
    overlap = train_df.index.intersection(val_df.index)
    print(f"Overlap between train and val indices: {len(overlap)}")
    if len(overlap) > 0:
        raise RuntimeError("CRITICAL ERROR: Overlap detected between train and val sets!")

    # Check conservation of data
    total_split = len(train_df) + len(val_df)
    print(f"Total split (Train + Val): {total_split}")
    print(f"Original Filtered: {filtered_count}")
    if total_split != filtered_count:
        raise RuntimeError(f"CRITICAL ERROR: Data loss! {filtered_count} -> {total_split}")

    # Check class balance in Val
    print("\nValidation Set Class Distribution:")
    print(val_df['cvv'].value_counts().sort_index())
    
    val_counts = val_df['cvv'].value_counts()
    if val_counts.nunique() != 1: # All counts should be the same
         print("WARNING: Validation set is not perfectly balanced (should strictly not happen with logic above unless n_val_per_class logic failed)")
    else:
        print("Validation set is perfectly balanced per class.")

    print("\nTraining Set Class Distribution:")
    print(train_df['cvv'].value_counts().sort_index())

    # 5. Saving
    print(f"\nSaving to {TRAIN_OUTPUT}...")
    train_df.to_csv(TRAIN_OUTPUT, index=False)
    print(f"Saving to {VAL_OUTPUT}...")
    val_df.to_csv(VAL_OUTPUT, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
