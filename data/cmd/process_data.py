import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import shutil
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from logs.cmd.logger_config import setup_logger

logger = setup_logger("process_data", save_to_file=True)

SEED = 42
RAW_PATH = Path("data/raw/bar_pass_prediction.csv")
PROCESSED_DIR = Path("data/processed")


def raw_data_exists(dest_dir: Path = Path("data/raw")) -> bool:
    """
    Check if the raw data directory exists and contains CSV files.
    """
    if not dest_dir.exists():
        logger.info("Raw data directory does not exist: %s", dest_dir)
        return False

    csv_files = list(dest_dir.glob("*.csv"))
    if not csv_files:
        logger.info("No CSV files found in raw data directory: %s", dest_dir)
        return False

    logger.info("Raw data directory exists and contains CSV files: %s", dest_dir)

    return True


def extract_raw_data(dest_dir: Path = Path("data/raw")) -> Path:
    """
    Download LSAC dataset via kagglehub and copy CSV files to data/raw.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(kagglehub.dataset_download("danofer/law-school-admissions-bar-passage"))

    copied_files = []
    for csv_file in cache_path.glob("*.csv"):
        target_path = dest_dir / csv_file.name
        shutil.copy2(csv_file, target_path)
        copied_files.append(target_path)
        logger.info("File copied to: %s", target_path)

    return copied_files[0]


def main(raw_path: Path = RAW_PATH, out_dir: Path = PROCESSED_DIR) -> None:
    """
    Extracts raw data if needed and processes it into train, validation, and test splits,
    saving them to the processed directory.
    """
    if not raw_data_exists():
        logger.info("Starting data extraction...")
        raw_data_path = extract_raw_data()
        logger.info("Data extraction completed. Raw data saved to: %s", raw_data_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path)

    # 1. removes registers without label and inverts target (1 = Risk/Failed, 0 = Passed)
    df = df.dropna(subset=["pass_bar"]).copy()
    df["target_risk"] = 1 - df["pass_bar"].astype(int)

    # 2. maintains only attributes available until the end of the 1st year
    # to avoid data leakage
    selected_cols = [
        "lsat",
        "ugpa",
        "zfygpa",
        "tier",
        "cluster",
        "fulltime",
        "fam_inc",
        "age",
        "gender",
        "race1",
    ]
    df_clean = df[selected_cols + ["target_risk"]].copy()

    # 3. stateless padronization
    df_clean["age"] = np.where(
        df_clean["age"] < 0,
        91 + df_clean["age"],
        df_clean["age"],
    )
    df_clean["fulltime"] = df_clean["fulltime"].map({1: 1, 2: 0})

    for col in ["gender", "race1"]:
        df_clean[col] = df_clean[col].astype("string").str.strip().str.lower()

    # 4. division: 65% train, 25% validation, 10% test
    temp_df, test_df = train_test_split(
        df_clean,
        test_size=0.10,
        stratify=df_clean["target_risk"],
        random_state=SEED,
    )
    # 0.25 / 0.90 = 5/18 (~0.2778) remaining corresponds to 25% of the total
    train_df, val_df = train_test_split(
        temp_df,
        test_size=(0.25 / 0.90),
        stratify=temp_df["target_risk"],
        random_state=SEED,
    )

    train_df.to_csv(out_dir / "pre_train.csv", index=False)
    val_df.to_csv(out_dir / "pre_val.csv", index=False)
    test_df.to_csv(out_dir / "pre_test.csv", index=False)

    logger.info(f"Total cleaned rows: {len(df_clean)}")
    logger.info(
        f"Splits -> pre_train: {len(train_df)} ({len(train_df)/len(df_clean):.1%}) | "
        f"pre_val: {len(val_df)} ({len(val_df)/len(df_clean):.1%}) | "
        f"pre_test: {len(test_df)} ({len(test_df)/len(df_clean):.1%})"
    )


if __name__ == "__main__":
    main()
