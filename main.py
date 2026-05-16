import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class BiomedicalECGPipeline:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.df = None
        self.cleaned_df = None
        self.output_dir = "outputs"
        self.data_dir = "data"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def find_dataset(self):
        possible_files = [
            self.csv_path,
            "data/dataset_original.csv",
            "mHealth_subject1.csv",
            "mhealth_subject1.csv",
            "MHEALTHDATASET.csv",
        ]

        for file in possible_files:
            if file and os.path.exists(file):
                return file

        raise FileNotFoundError(
            "Dataset not found. Put your CSV in data/dataset_original.csv "
            "or place mHealth_subject1.csv beside main.py."
        )

    def load_data(self):
        try:
            file_path = self.find_dataset()
            self.df = pd.read_csv(file_path)
            print(f"[OK] Loaded dataset: {file_path}")
            print(f"[INFO] Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"[ERROR] Could not load dataset: {e}")
            sys.exit(1)

    def assign_columns_if_missing(self):
        if self.df is None:
            raise ValueError("No dataset loaded.")

        if all(isinstance(c, int) for c in self.df.columns):
            pass

        if len(self.df.columns) >= 24:
            self.df.columns = [
                "chest_acc_x", "chest_acc_y", "chest_acc_z",
                "ecg_1", "ecg_2",
                "left_ankle_acc_x", "left_ankle_acc_y", "left_ankle_acc_z",
                "left_ankle_gyro_x", "left_ankle_gyro_y", "left_ankle_gyro_z",
                "left_ankle_mag_x", "left_ankle_mag_y", "left_ankle_mag_z",
                "right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z",
                "right_arm_gyro_x", "right_arm_gyro_y", "right_arm_gyro_z",
                "right_arm_mag_x", "right_arm_mag_y", "right_arm_mag_z",
                "activity"
            ][:len(self.df.columns)]

    def clean_data(self):
        try:
            self.assign_columns_if_missing()
            df = self.df.copy()

            df = df.drop_duplicates()

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

            if "activity" in df.columns:
                df = df[df["activity"] != 0]

            df["time_index"] = np.arange(len(df))

            self.cleaned_df = df
            cleaned_path = os.path.join(self.data_dir, "dataset_cleaned.csv")
            df.to_csv(cleaned_path, index=False)

            print(f"[OK] Cleaned dataset saved: {cleaned_path}")
            print(f"[INFO] Cleaned shape: {df.shape}")
            return df

        except Exception as e:
            print(f"[ERROR] Cleaning failed: {e}")
            sys.exit(1)

    def select_ecg_column(self):
        possible_ecg = ["ecg_1", "ecg_2", "ECG_1", "ECG_2", "ecg"]
        for col in possible_ecg:
            if col in self.cleaned_df.columns:
                return col

        numeric_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        return numeric_cols[0]

    def compute_statistics(self):
        try:
            df = self.cleaned_df
            ecg_col = self.select_ecg_column()
            ecg = df[ecg_col].to_numpy(dtype=float)

            mean_val = np.mean(ecg)
            median_val = np.median(ecg)
            std_val = np.std(ecg)
            var_val = np.var(ecg)

            q1, q3 = np.percentile(ecg, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ecg[(ecg < lower) | (ecg > upper)]

            skewness = np.mean(((ecg - mean_val) / std_val) ** 3) if std_val != 0 else 0

            print("\n===== ECG DESCRIPTIVE STATISTICS =====")
            print(f"ECG Column Used: {ecg_col}")
            print(f"Mean: {mean_val:.4f}")
            print(f"Median: {median_val:.4f}")
            print(f"Standard Deviation: {std_val:.4f}")
            print(f"Variance: {var_val:.4f}")
            print(f"Skewness: {skewness:.4f}")
            print(f"Outlier Count: {len(outliers)}")

            stats_df = pd.DataFrame({
                "Metric": ["Mean", "Median", "Standard Deviation", "Variance", "Skewness", "Outlier Count"],
                "Value": [mean_val, median_val, std_val, var_val, skewness, len(outliers)]
            })

            stats_df.to_csv(os.path.join(self.output_dir, "ecg_statistics.csv"), index=False)
            return ecg_col

        except Exception as e:
            print(f"[ERROR] Statistics failed: {e}")
            sys.exit(1)

    def comparative_analysis(self, ecg_col):
        try:
            df = self.cleaned_df

            if "activity" not in df.columns:
                print("[WARNING] No activity column found for comparative analysis.")
                return

            comparison = df.groupby("activity")[ecg_col].agg(
                ["mean", "median", "std", "var", "count"]
            )

            comparison.to_csv(os.path.join(self.output_dir, "activity_comparison.csv"))

            print("\n===== COMPARATIVE ANALYSIS BY ACTIVITY =====")
            print(comparison)

        except Exception as e:
            print(f"[ERROR] Comparative analysis failed: {e}")

    def correlation_analysis(self, ecg_col):
        try:
            df = self.cleaned_df
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()

            corr.to_csv(os.path.join(self.output_dir, "correlation_matrix.csv"))

            plt.figure(figsize=(10, 8))
            plt.imshow(corr, aspect="auto")
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
            plt.yticks(range(len(corr.columns)), corr.columns)
            plt.title("Correlation Heatmap of Wearable Biomedical Signals")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "static_3_correlation_heatmap.png"))
            plt.close()

            print("[OK] Correlation analysis completed.")

        except Exception as e:
            print(f"[ERROR] Correlation analysis failed: {e}")

    def create_static_plots(self, ecg_col):
        try:
            df = self.cleaned_df

            plt.figure(figsize=(8, 5))
            plt.hist(df[ecg_col], bins=40)
            plt.title("ECG Signal Distribution")
            plt.xlabel("ECG Signal Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "static_1_ecg_histogram.png"))
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.boxplot(df[ecg_col], vert=False)
            plt.title("ECG Signal Outlier Distribution")
            plt.xlabel("ECG Signal Value")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "static_2_ecg_boxplot.png"))
            plt.close()

            if "activity" in df.columns:
                plt.figure(figsize=(8, 5))
                plt.scatter(df["activity"], df[ecg_col], alpha=0.4)
                plt.title("ECG Signal Behavior Across Physical Activities")
                plt.xlabel("Activity Label")
                plt.ylabel("ECG Signal Value")
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "static_4_ecg_activity_scatter.png"))
                plt.close()

            print("[OK] Static plots saved in outputs folder.")

        except Exception as e:
            print(f"[ERROR] Static plotting failed: {e}")

    def create_animations(self, ecg_col):
        try:
            df = self.cleaned_df.head(800).copy()

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_xlim(0, len(df))
            ax.set_ylim(df[ecg_col].min(), df[ecg_col].max())
            ax.set_title("Animated ECG Signal Trend Over Time")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("ECG Signal")

            line, = ax.plot([], [])

            def update(frame):
                x = df["time_index"].iloc[:frame]
                y = df[ecg_col].iloc[:frame]
                line.set_data(x, y)
                return line,

            anim = FuncAnimation(fig, update, frames=len(df), interval=20, blit=True)
            anim.save(os.path.join(self.output_dir, "animation_1_ecg_trend.gif"), writer=PillowWriter(fps=20))
            plt.close()

            if "activity" in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.set_xlim(df["activity"].min() - 1, df["activity"].max() + 1)
                ax.set_ylim(df[ecg_col].min(), df[ecg_col].max())
                ax.set_title("Animated ECG Response Across Activity Labels")
                ax.set_xlabel("Activity Label")
                ax.set_ylabel("ECG Signal")

                scatter = ax.scatter([], [])

                def update_scatter(frame):
                    x = df["activity"].iloc[:frame]
                    y = df[ecg_col].iloc[:frame]
                    scatter.set_offsets(np.column_stack((x, y)))
                    return scatter,

                anim2 = FuncAnimation(fig, update_scatter, frames=len(df), interval=20, blit=True)
                anim2.save(os.path.join(self.output_dir, "animation_2_ecg_activity.gif"), writer=PillowWriter(fps=20))
                plt.close()

            print("[OK] Animations saved in outputs folder.")

        except Exception as e:
            print(f"[ERROR] Animation creation failed: {e}")

    def run_pipeline(self):
        self.load_data()
        self.clean_data()
        ecg_col = self.compute_statistics()
        self.comparative_analysis(ecg_col)
        self.correlation_analysis(ecg_col)
        self.create_static_plots(ecg_col)
        self.create_animations(ecg_col)

        print("\nDONE. Your ML/data pipeline successfully finished.")
        print("Check the data and outputs folders.")


if __name__ == "__main__":
    pipeline = BiomedicalECGPipeline()
    pipeline.run_pipeline()