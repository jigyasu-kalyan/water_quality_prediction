"""
Module: Data Ingestion & Exploratory Data Analysis (EDA)
Author: Jigyasu
Project: Water Quality Prediction via Gradient Boosting (Phase 1)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(file_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Loads the water quality dataset with optional subsampling for rapid EDA.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    print(f"[+] Loading dataset from: {file_path}")
    if sample_size:
        df = pd.read_csv(file_path, nrows=sample_size)
    else:
        df = pd.read_csv(file_path)

    print(f"[+] Loaded {len(df):,} records across {df.shape[1]} attributes.")
    return df


def generate_eda_report(df: pd.DataFrame, output_dir: str = "reports") -> dict:
    """
    Generates a full statistical overview of the dataset:
    - Missing value analysis
    - Physical & chemical parameter ranges
    - Target distributions (CCME_Values and CCME_WQI)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    summary_file = os.path.join(output_dir, "eda_statistical_summary.txt")
    with open(summary_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("WATER QUALITY MONITORING DATASET - EXPLORATORY DATA ANALYSIS\n")
        f.write("Author: Jigyasu (Phase 1 Data Ingestion & Profiling)\n")
        f.write("=" * 70 + "\n\n")

        # 1. Dataset Shape & Columns
        f.write(f"1. Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
        f.write("2. Features Identified:\n")
        for col in df.columns:
            f.write(f"   - {col} (dtype: {df[col].dtype})\n")
        f.write("\n")

        # 2. Missing Values Analysis
        f.write("3. Missing Value Audit:\n")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        miss_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct})
        f.write(miss_df.to_string() + "\n\n")

        # 3. Numeric Summary
        f.write("4. Summary Statistics (Numeric Physical-Chemical Parameters):\n")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        f.write(df[numeric_cols].describe().T.to_string() + "\n\n")

        # 4. Target Distribution
        if "CCME_WQI" in df.columns:
            f.write("5. Categorical Target (CCME_WQI Class Distribution):\n")
            wqi_dist = df["CCME_WQI"].value_counts(dropna=False)
            wqi_pct = df["CCME_WQI"].value_counts(normalize=True, dropna=False) * 100
            wqi_summary = pd.DataFrame({"Count": wqi_dist, "Percentage (%)": wqi_pct})
            f.write(wqi_summary.to_string() + "\n\n")

    print(f"[+] Written EDA summary text report to: {summary_file}")
    return {"numeric_cols": numeric_cols.tolist()}


def plot_distributions(df: pd.DataFrame, output_dir: str = "reports/figures"):
    """
    Renders and saves publication-quality figures:
    - Target CCME_WQI class frequency
    - Chemical parameter correlation heatmap
    - Distribution histograms for physical/chemical features
    """
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. CCME_WQI Class Countplot
    if "CCME_WQI" in df.columns:
        plt.figure(figsize=(8, 4.5))
        order = ["Poor", "Marginal", "Fair", "Good", "Excellent"]
        existing_order = [o for o in order if o in df["CCME_WQI"].values]
        ax = sns.countplot(
            data=df,
            x="CCME_WQI",
            order=existing_order if existing_order else df["CCME_WQI"].value_counts().index,
            palette="Blues_r"
        )
        plt.title("Water Quality Index (CCME_WQI) Category Distribution", fontsize=12, fontweight="bold")
        plt.xlabel("CCME Water Quality Index Category")
        plt.ylabel("Sample Count")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:,}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_wqi_distribution.png", dpi=300)
        plt.close()

    # 2. Correlation Matrix of Chemical-Physical Sensors
    chem_features = [
        "Ammonia (mg/l)",
        "Biochemical Oxygen Demand (mg/l)",
        "Dissolved Oxygen (mg/l)",
        "Orthophosphate (mg/l)",
        "pH (ph units)",
        "Temperature (cel)",
        "Nitrogen (mg/l)",
        "Nitrate (mg/l)"
    ]
    available_chem = [c for c in chem_features if c in df.columns]

    if available_chem:
        plt.figure(figsize=(9, 7))
        corr = df[available_chem].corr()
        sns.heatmap(corr, annot=True, cmap="mako", fmt=".2f", linewidths=0.5)
        plt.title("Sensor-to-Sensor Multicollinearity Matrix", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sensor_correlation_matrix.png", dpi=300)
        plt.close()

    print(f"[+] Visualizations saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA for Water Quality Prediction")
    parser.add_argument("--data", type=str, default="data/water_quality_dataset.csv", help="Dataset path")
    parser.add_argument("--sample", type=int, default=50000, help="Number of rows to sample for fast profiling")
    args = parser.parse_args()

    df_data = load_dataset(args.data, sample_size=args.sample)
    generate_eda_report(df_data)
    plot_distributions(df_data)
