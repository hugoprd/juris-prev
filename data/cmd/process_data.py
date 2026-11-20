import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import shutil
import kagglehub

from logs.cmd.logger_config import setup_logger

logger = setup_logger("evaluate_log", save_to_file=True)


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


def main():
    if not raw_data_exists():
        logger.info("Starting data extraction...")
        raw_data_path = extract_raw_data()
        logger.info("Data extraction completed. Raw data saved to: %s", raw_data_path)


if __name__ == "__main__":
    main()
