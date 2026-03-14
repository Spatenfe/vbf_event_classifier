import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import argparse

# Configuration
DATA_PATH = "ml_framework/data/large/matched_events_all_no_dup.csv"
OUTPUT_DIR = "analysis_plots"

def create_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    print(f"Loading data from {DATA_PATH}...")
    return pd.read_csv(DATA_PATH)

def analyze_basic_info(df):
    print("\n--- Basic Information ---")
    print(df.info())
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

def plot_distributions(df):
    print("\nGenerating distribution plots...")
    # Plot target variable 'cvv'
    plt.figure(figsize=(10, 6))
    sns.histplot(df['cvv'], kde=False, bins=30)
    plt.title('Distribution of Target Variable (cvv)')
    plt.xlabel('cvv')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(OUTPUT_DIR, 'cvv_distribution.png'))
    plt.close()

    # Plot distributions for other numerical columns
    # Select a subset of interesting columns to avoid too many plots if there are many
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Exclude 'cvv' and 'event_id' from this loop as 'cvv' is plotted separately and 'event_id' is an ID
    cols_to_plot = [col for col in numerical_cols if col not in ['cvv', 'event_id']]
    
    # Limit to first 9 for a grid view example, or just plot all individually. 
    # Let's plot top correlations or just some key features.
    # Given the column names seen in `head`, let's plot a few specific ones like pt, eta, phi
    
    for col in cols_to_plot[:10]: # Limiting to first 10 for now to avoid clutter
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=False, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{col}_distribution.png'))
        plt.close()

def plot_correlation_matrix(df):
    print("\nGenerating correlation matrix...")
    plt.figure(figsize=(20, 16))
    # Drop event_id as it's not useful for correlation
    df_corr = df.drop(columns=['event_id']) if 'event_id' in df.columns else df
    corr = df_corr.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
    plt.close()

def plot_target_correlations(df):
    print("\nGenerating target correlation plot...")
    if 'cvv' not in df.columns:
        print("Target column 'cvv' not found.")
        return

    # Calculate correlations with 'cvv'
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlations = df[numerical_cols].corrwith(df['cvv']).drop('cvv')
    
    # Sort by absolute correlation
    correlations_sorted = correlations.abs().sort_values(ascending=False)
    # Get original values for sorted index to keep the sign
    correlations = correlations[correlations_sorted.index]

    # Plot top 20 correlations
    plt.figure(figsize=(10, 8))
    sns.barplot(x=correlations.head(20).values, y=correlations.head(20).index, hue=correlations.head(20).index, palette='coolwarm', legend=False)
    plt.title('Top 20 Feature Correlations with Target (cvv)')
    plt.xlabel('Correlation Coefficient')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cvv_correlations.png'))
    plt.close()

def plot_binary_target_correlations(df):
    print("\nGenerating binary target (cvv=1 vs other) correlation plot...")
    if 'cvv' not in df.columns:
        print("Target column 'cvv' not found.")
        return

    # Create binary target
    binary_target = (df['cvv'] == 1.0).astype(int)
    
    # Calculate correlations with the new binary target
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlations = df[numerical_cols].drop(columns=['cvv'], errors='ignore').corrwith(binary_target)
    
    # Sort by absolute correlation
    correlations_sorted = correlations.abs().sort_values(ascending=False)
    # Get original values for sorted index to keep the sign
    correlations = correlations[correlations_sorted.index]

    # Plot top 20 correlations
    plt.figure(figsize=(10, 8))
    sns.barplot(x=correlations.head(20).values, y=correlations.head(20).index, hue=correlations.head(20).index, palette='coolwarm', legend=False)
    plt.title('Top 20 Feature Correlations with Binary Target (cvv=1 vs other)')
    plt.xlabel('Correlation Coefficient')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cvv_binary_correlations.png'))
    plt.close()

def plot_pca(df):
    print("\nGenerating PCA plot...")
    if 'cvv' not in df.columns:
        print("Target column 'cvv' not found, skipping PCA.")
        return

    # Prepare features and target
    # Drop event_id and target for PCA features
    cols_to_drop = ['event_id', 'cvv']
    features = [col for col in df.columns if col not in cols_to_drop]
    
    X = df[features].fillna(0) # Handle NaN if any
    y = df['cvv']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, y.reset_index(drop=True)], axis=1)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cvv', palette='viridis', alpha=0.7)
    plt.title('PCA of Dataset (2 Components)')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_2d.png'))
    plt.close()

def plot_feature_importance(df):
    print("\nGenerating Feature Importance plot...")
    if 'cvv' not in df.columns:
        print("Target column 'cvv' not found, skipping Feature Importance.")
        return

    # Prepare features and target
    cols_to_drop = ['event_id', 'cvv']
    features = [col for col in df.columns if col not in cols_to_drop]
    X = df[features].fillna(0)
    # Ensure y is categorical/integer for Classifier purposes if 'cvv' represents classes
    # If 'cvv' is continuous, we should use Regressor. The config suggested classification classes 0 and 1, but cvv is float.
    # Looking at the head, cvv is 1.5. Wait - head showed 1.5. Config said target_column "cvv", discard classes [1.0].
    # Is it regression or classification? "vbf_event_classifier" suggests classification.
    # Let's check unique values of cvv first.
    unique_vals = df['cvv'].unique()
    print(f"Unique target values: {unique_vals}")
    
    # If few unique values, treat as classification.
    # Cast to string to ensure Classifier treats it as discrete classes
    y = df['cvv'].astype(str)
    
    # Using Classifier for now as it's a "classifier" project.
    # Limit depth and trees for speed
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = rf.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df, hue='Feature', palette='magma', legend=False)
    plt.title('Top 20 Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    plt.close()

def plot_structured_correlation_matrix(df):
    print("\nGenerating Structured Correlation Matrix...")
    # Drop event_id and cvv
    cols = [col for col in df.columns if col not in ['event_id', 'cvv']]
    
    # Priority order for properties: m, eta, pt, ... (based on user request)
    # We will sort by: 1. Index in property_priority (if found), 2. Property name (alphabetical), 3. Particle prefix
    property_priority = ['m', 'eta', 'pt', 'phi', 'e', 'dR']
    
    def get_sort_key(col_name):
        parts = col_name.split('_')
        # Assuming the last part is the property
        prop = parts[-1]
        prefix = '_'.join(parts[:-1])
        
        try:
            prop_idx = property_priority.index(prop)
        except ValueError:
            prop_idx = len(property_priority) # Put at the end
            
        return (prop_idx, prop, prefix)

    sorted_cols = sorted(cols, key=get_sort_key)
    
    plt.figure(figsize=(24, 20))
    # Correlation of only these sorted columns
    corr = df[sorted_cols].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Structured Correlation Matrix (Sorted by Property: m, eta, pt...)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'structured_correlation_matrix.png'))
    plt.close()

def plot_properties_by_target(df):
    print("\nGenerating property distributions by target...")
    prop_dir = os.path.join(OUTPUT_DIR, "property_plots")
    if not os.path.exists(prop_dir):
        os.makedirs(prop_dir)
        print(f"Created output directory: {prop_dir}")

    if 'cvv' not in df.columns:
        print("Target column 'cvv' not found, skipping property plots.")
        return

    cols = [col for col in df.columns if col not in ['event_id', 'cvv']]
    properties = set()
    for col in cols:
        if '_' in col:
            properties.add(col.split('_')[-1])
        else:
            properties.add(col)

    import math
    for prop in properties:
        prop_cols = [col for col in cols if col.split('_')[-1] == prop or col == prop]
        if not prop_cols:
            continue

        n_cols = len(prop_cols)
        cols_grid = min(4, n_cols)
        rows_grid = math.ceil(n_cols / cols_grid)

        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(cols_grid * 5, rows_grid * 4), squeeze=False)
        axes_flat = axes.flatten()

        mask_cvv1 = (df['cvv'] == 1.0)

        for i, col in enumerate(prop_cols):
            ax = axes_flat[i]
            data_red = df[mask_cvv1][col].dropna()
            data_green = df[~mask_cvv1][col].dropna()

            # Using kdeplot to create smooth line charts instead of bar charts
            try:
                sns.kdeplot(data_green, ax=ax, color='green', label='Other', linewidth=2)
                sns.kdeplot(data_red, ax=ax, color='red', label='cvv=1', linewidth=2)
            except Exception as e:
                # Fallback to step histplot if KDE fails (e.g. zero variance)
                sns.histplot(data_green, ax=ax, color='green', label='Other', stat='density', element='poly', fill=False, bins=30)
                sns.histplot(data_red, ax=ax, color='red', label='cvv=1', stat='density', element='poly', fill=False, bins=30)

            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.legend()

        for i in range(len(prop_cols), len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.suptitle(f'Property: {prop} distributions (cvv=1 vs other)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(prop_dir, f'property_{prop}.png'))
        plt.close()

def main():
    global DATA_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Analyze dataset features and distributions.")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the dataset CSV file.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save analysis plots.")
    parser.add_argument("--exclude_columns", nargs='+', default=[], help="List of columns to exclude from analysis.")
    args = parser.parse_args()

    # Update globals with arguments if provided
    DATA_PATH = args.data_path
    OUTPUT_DIR = args.output_dir

    create_output_dir()
    df = load_data()

    # Exclude columns
    if args.exclude_columns:
        print(f"Excluding columns: {args.exclude_columns}")
        # Check if columns exist before dropping to avoid errors, or just use ignore errors
        existing_cols = [col for col in args.exclude_columns if col in df.columns]
        missing_cols = [col for col in args.exclude_columns if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Columns not found in dataset: {missing_cols}")
            
        if existing_cols:
            df = df.drop(columns=existing_cols)
            print(f"Dropped {len(existing_cols)} columns.")

    analyze_basic_info(df)
    plot_distributions(df)
    plot_correlation_matrix(df)
    plot_target_correlations(df)
    plot_binary_target_correlations(df)
    plot_pca(df)
    plot_feature_importance(df)
    plot_structured_correlation_matrix(df)
    plot_properties_by_target(df)
    print(f"\nAnalysis complete. Plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
