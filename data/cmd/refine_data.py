import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from logs.cmd.logger_config import setup_logger  # noqa: E402

logger = setup_logger("refine_data", save_to_file=True)

PROCESSED_DIR = Path("data/processed")
REFINED_DIR = Path("data/refined")

CONT_COLS = ["lsat", "ugpa", "zfygpa", "age"]
ORD_COLS = ["tier", "fam_inc"]
BIN_COLS = ["fulltime"]
NOM_COLS = ["cluster"]
SENSITIVE_COLS = ["race1", "gender", "fam_inc", "age"]
TARGET_COL = "target_risk"


def add_engineered_features(
    df: pd.DataFrame, admission_stats: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """
    Create engineered features using only statistics from the training set.
    """
    out = df.copy()

    z_lsat = (out["lsat"] - admission_stats["lsat"][0]) / admission_stats["lsat"][1]
    z_ugpa = (out["ugpa"] - admission_stats["ugpa"][0]) / admission_stats["ugpa"][1]

    out["academic_momentum"] = out["zfygpa"] - ((z_lsat + z_ugpa) / 2.0)
    out["lsat_ugpa_ratio"] = out["lsat"] / (out["ugpa"] + 0.5)

    return out


def main(in_dir: Path = PROCESSED_DIR, out_dir: Path = REFINED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, pd.DataFrame] = {
        "train": pd.read_csv(in_dir / "pre_train.csv"),
        "val": pd.read_csv(in_dir / "pre_val.csv"),
        "test": pd.read_csv(in_dir / "pre_test.csv"),
    }

    for name, df in splits.items():
        df["age"] = df["age"].clip(lower=17, upper=85)
        df["ugpa"] = df["ugpa"].clip(lower=0.0, upper=4.0)
        splits[name] = df

    # 2. Imputer adjusted exclusively on train
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_all = ORD_COLS + BIN_COLS + NOM_COLS + ["race1", "gender"]

    splits["train"].loc[:, CONT_COLS] = np.asarray(
        num_imputer.fit_transform(splits["train"][CONT_COLS])
    )
    splits["train"].loc[:, cat_all] = np.asarray(
        cat_imputer.fit_transform(splits["train"][cat_all])
    )

    for name in ["val", "test"]:
        splits[name].loc[:, CONT_COLS] = np.asarray(
            num_imputer.transform(splits[name][CONT_COLS])
        )
        splits[name].loc[:, cat_all] = np.asarray(cat_imputer.transform(splits[name][cat_all]))

    # 3. Export sensitive attributes aligned for fairness auditing
    for name, df in splits.items():
        df[SENSITIVE_COLS + [TARGET_COL]].to_csv(out_dir / f"sensitive_{name}.csv", index=False)

    # 4. Engineering attributes with parameters calculated only on the training set
    admission_stats = {
        "lsat": (float(splits["train"]["lsat"].mean()), float(splits["train"]["lsat"].std())),
        "ugpa": (float(splits["train"]["ugpa"].mean()), float(splits["train"]["ugpa"].std())),
    }
    for name in ["train", "val", "test"]:
        splits[name] = add_engineered_features(splits[name], admission_stats)

    engineered_cols = ["academic_momentum", "lsat_ugpa_ratio"]
    scale_cols = CONT_COLS + ORD_COLS + engineered_cols

    scaler = StandardScaler()
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

    scaler.fit(splits["train"][scale_cols])
    ohe.fit(splits["train"][NOM_COLS])
    ohe_names = list(ohe.get_feature_names_out(NOM_COLS))

    for name, df in splits.items():
        scaled_df = pd.DataFrame(
            np.asarray(scaler.transform(df[scale_cols])),
            columns=scale_cols,
            index=df.index,
        )
        bin_df = df[BIN_COLS].astype(int).reset_index(drop=True)
        ohe_df = pd.DataFrame(
            np.asarray(ohe.transform(df[NOM_COLS])),
            columns=ohe_names,
        )
        target_df = df[[TARGET_COL]].astype(int).reset_index(drop=True)

        refined_df = pd.concat(
            [scaled_df.reset_index(drop=True), bin_df, ohe_df, target_df],
            axis=1,
        )
        refined_df.to_csv(out_dir / f"{name}.csv", index=False)

    logger.info(
        f"Refine concluded -> {len(scale_cols) + len(ohe_names)} features | "
        f"train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
