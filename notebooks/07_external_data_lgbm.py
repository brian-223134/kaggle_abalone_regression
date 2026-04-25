# %% [markdown]
# # External UCI Data + Target Encoding + LGBM
#
# This experiment appends the original UCI Abalone rows to each training fold.
# Validation is still measured only on the Kaggle fold validation rows, so the
# score remains comparable to earlier Kaggle-only CV.

# %%
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42
N_SPLITS = 5
ID_COL = "id"
TARGET = "Rings"
SEX_CATEGORIES = ["F", "I", "M"]

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
PROCEED_DIR = DATA_DIR / "proceed"
EXTERNAL_DIR = DATA_DIR / "external"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_TRAIN_PATH = PROCEED_DIR / "train_fe_v2.csv"
KAGGLE_TEST_PATH = PROCEED_DIR / "test_fe_v2.csv"
UCI_PATH = EXTERNAL_DIR / "abalone.data"

EXTERNAL_FEATURE_PATH = PROCEED_DIR / "external_uci_fe_v2.csv"
OOF_PATH = PROCEED_DIR / "oof_v4_lgbm_te_external.csv"
TEST_PRED_PATH = PROCEED_DIR / "test_pred_v4_lgbm_te_external.csv"
SCORES_PATH = PROCEED_DIR / "scores_v4_lgbm_te_external.csv"
SUBMISSION_PATH = SUBMISSION_DIR / "submission_v4_lgbm_te_external.csv"


# %%
def rmsle(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def make_stratification_bins(target: pd.Series, q: int = 10) -> np.ndarray:
    return pd.qcut(target, q=q, labels=False, duplicates="drop").astype(int).to_numpy()


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Height_is_zero"] = out["Height"].eq(0).astype(int)
    out["Volume"] = out["Length"] * out["Diameter"] * out["Height"]
    out["Density"] = safe_divide(out["Whole_weight"], out["Volume"])
    out["Shucked_ratio"] = safe_divide(out["Shucked_weight"], out["Whole_weight"])
    out["Viscera_ratio"] = safe_divide(out["Viscera_weight"], out["Whole_weight"])
    out["Shell_ratio"] = safe_divide(out["Shell_weight"], out["Whole_weight"])
    out["Shell_to_shucked"] = safe_divide(out["Shell_weight"], out["Shucked_weight"])

    sex_dummies = pd.get_dummies(out["Sex"], prefix="Sex", dtype=int)
    sex_dummies = sex_dummies.reindex(columns=[f"Sex_{value}" for value in SEX_CATEGORIES], fill_value=0)
    out = pd.concat([out.drop(columns=["Sex"]), sex_dummies], axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def add_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Area"] = out["Length"] * out["Diameter"]
    out["Diameter_to_Length"] = safe_divide(out["Diameter"], out["Length"])
    out["Height_to_Length"] = safe_divide(out["Height"], out["Length"])
    out["Height_to_Diameter"] = safe_divide(out["Height"], out["Diameter"])

    out["Component_weight_sum"] = out["Shucked_weight"] + out["Viscera_weight"] + out["Shell_weight"]
    out["Component_sum_ratio"] = safe_divide(out["Component_weight_sum"], out["Whole_weight"])
    out["Residual_weight"] = out["Whole_weight"] - out["Component_weight_sum"]
    out["Residual_weight_ratio"] = safe_divide(out["Residual_weight"], out["Whole_weight"])

    out["Whole_weight_to_Length"] = safe_divide(out["Whole_weight"], out["Length"])
    out["Whole_weight_to_Area"] = safe_divide(out["Whole_weight"], out["Area"])
    out["Whole_weight_to_Volume"] = safe_divide(out["Whole_weight"], out["Volume"])
    out["Shucked_to_Shell"] = safe_divide(out["Shucked_weight"], out["Shell_weight"])
    out["Viscera_to_Shell"] = safe_divide(out["Viscera_weight"], out["Shell_weight"])

    for col in [
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
        "Volume",
        "Density",
    ]:
        out[f"log1p_{col}"] = np.log1p(out[col].clip(lower=0))
    return out.replace([np.inf, -np.inf], np.nan)


def load_uci_v2(kaggle_feature_cols: list[str], fill_values: pd.Series) -> pd.DataFrame:
    columns = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
        TARGET,
    ]
    uci = pd.read_csv(UCI_PATH, names=columns)
    uci.insert(0, ID_COL, -np.arange(1, len(uci) + 1))
    uci = add_v2_features(add_base_features(uci))
    uci[kaggle_feature_cols] = uci[kaggle_feature_cols].fillna(fill_values)
    return uci[[ID_COL, *kaggle_feature_cols, TARGET]]


def make_lgbm(seed: int = RANDOM_STATE, n_estimators: int = 5000) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.025,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def height_bounds(df: pd.DataFrame, alpha: float = 2.0) -> tuple[float, float]:
    q1 = df["Height"].quantile(0.25)
    q3 = df["Height"].quantile(0.75)
    iqr = q3 - q1
    return float(q1 - alpha * iqr), float(q3 + alpha * iqr)


def add_target_encoding_features(
    X: pd.DataFrame,
    fit_X: pd.DataFrame,
    fit_y: np.ndarray | pd.Series,
    *,
    alpha: float = 2.0,
) -> pd.DataFrame:
    X = X.copy()
    fit_X = fit_X.copy()
    fit_y = np.asarray(fit_y, dtype=float)
    global_mean = float(np.mean(fit_y))

    lower, upper = height_bounds(fit_X, alpha=alpha)
    X["Height_clipped_iqr2"] = X["Height"].clip(lower, upper)
    fit_X["Height_clipped_iqr2"] = fit_X["Height"].clip(lower, upper)

    tmp = fit_X.copy()
    tmp[TARGET] = fit_y

    sweight_map = tmp.groupby("Shell_weight")[TARGET].mean()
    height_map = tmp.groupby("Height_clipped_iqr2")[TARGET].mean()
    weight_map = tmp.groupby("Whole_weight")[TARGET].mean()
    viscera_map = tmp.groupby("Viscera_weight")[TARGET].mean()
    sex_map = tmp.groupby(["Sex_F", "Sex_I", "Sex_M"])[TARGET].mean()

    out = X.copy()
    out["Sweight_avg"] = out["Shell_weight"].map(sweight_map).fillna(global_mean)
    out["Height_avg"] = out["Height_clipped_iqr2"].map(height_map)
    out["Weight_avg"] = out["Whole_weight"].map(weight_map)
    out["Viscera_weight_avg"] = out["Viscera_weight"].map(viscera_map)

    sex_key = pd.MultiIndex.from_frame(out[["Sex_F", "Sex_I", "Sex_M"]])
    out["Sex_avg"] = pd.Series(sex_key.map(sex_map), index=out.index).fillna(global_mean)

    out["Height_avg"] = out["Height_avg"].fillna(out["Sweight_avg"]).fillna(global_mean)
    out["Viscera_weight_avg"] = (
        out["Viscera_weight_avg"].fillna(out["Sweight_avg"]).fillna(global_mean)
    )
    out["Weight_avg"] = (
        out["Weight_avg"]
        .fillna(out["Sweight_avg"])
        .fillna(out["Viscera_weight_avg"])
        .fillna(global_mean)
    )
    return out


# %%
kaggle_train = pd.read_csv(KAGGLE_TRAIN_PATH)
kaggle_test = pd.read_csv(KAGGLE_TEST_PATH)
feature_cols = [c for c in kaggle_train.columns if c not in [ID_COL, TARGET]]
uci_train = load_uci_v2(feature_cols, kaggle_train[feature_cols].median(numeric_only=True))
uci_train.to_csv(EXTERNAL_FEATURE_PATH, index=False)

y = kaggle_train[TARGET].astype(float).to_numpy()
y_bins = make_stratification_bins(kaggle_train[TARGET])
X = kaggle_train[feature_cols]
X_test = kaggle_test[feature_cols]

print(f"kaggle_train: {kaggle_train.shape}")
print(f"uci_train   : {uci_train.shape}")
print(f"kaggle_test : {kaggle_test.shape}")
print(f"saved: {EXTERNAL_FEATURE_PATH.relative_to(PROJECT_ROOT)}")


# %%
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(kaggle_train), dtype=float)
score_rows: list[dict[str, float | int | str]] = []
best_iterations: list[int] = []
start = perf_counter()

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y_bins), start=1):
    fold_train = kaggle_train.iloc[train_idx]
    fold_valid = kaggle_train.iloc[valid_idx]
    train_with_external = pd.concat([fold_train, uci_train], ignore_index=True)

    X_train = train_with_external[feature_cols]
    y_train = train_with_external[TARGET].astype(float).to_numpy()
    X_valid = fold_valid[feature_cols]
    y_valid = fold_valid[TARGET].astype(float).to_numpy()

    X_train = add_target_encoding_features(X_train, X_train, y_train)
    X_valid = add_target_encoding_features(X_valid, X_train[feature_cols], y_train)

    model = make_lgbm(seed=RANDOM_STATE + fold)
    model.fit(
        X_train,
        np.log1p(y_train),
        eval_set=[(X_valid, np.log1p(y_valid))],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )

    best_iteration = int(model.best_iteration_ or model.n_estimators)
    pred = np.clip(np.expm1(model.predict(X_valid, num_iteration=best_iteration)), 0, None)
    oof[valid_idx] = pred
    score = rmsle(y_valid, pred)
    best_iterations.append(best_iteration)
    score_rows.append(
        {
            "experiment": "v2_te_lgbm_log_external",
            "fold": fold,
            "rmsle": score,
            "best_iteration": best_iteration,
        }
    )
    print(f"v2_te_lgbm_log_external fold {fold}: RMSLE={score:.6f}, best_iter={best_iteration}")

summary = {
    "experiment": "v2_te_lgbm_log_external",
    "fold": "mean",
    "rmsle": float(np.mean([row["rmsle"] for row in score_rows])),
    "rmsle_std": float(np.std([row["rmsle"] for row in score_rows])),
    "oof_rmsle": rmsle(y, oof),
    "best_iteration": int(round(np.mean(best_iterations))),
    "fit_seconds": perf_counter() - start,
}
score_rows.append(summary)
scores = pd.DataFrame(score_rows)
print(scores.to_string(index=False))


# %%
best_iteration = int(summary["best_iteration"])
full_train = pd.concat([kaggle_train, uci_train], ignore_index=True)
full_X = full_train[feature_cols]
full_y = full_train[TARGET].astype(float).to_numpy()

final_X = add_target_encoding_features(full_X, full_X, full_y)
final_X_test = add_target_encoding_features(X_test, full_X, full_y)

final_model = make_lgbm(seed=RANDOM_STATE, n_estimators=best_iteration)
final_model.fit(final_X, np.log1p(full_y))
test_pred = np.clip(np.expm1(final_model.predict(final_X_test)), 0, None)

oof_df = kaggle_train[[ID_COL, TARGET]].copy()
oof_df["pred_v2_te_lgbm_log_external"] = oof

test_pred_df = kaggle_test[[ID_COL]].copy()
test_pred_df["pred_v2_te_lgbm_log_external"] = test_pred

submission = kaggle_test[[ID_COL]].copy()
submission[TARGET] = test_pred

oof_df.to_csv(OOF_PATH, index=False)
test_pred_df.to_csv(TEST_PRED_PATH, index=False)
scores.to_csv(SCORES_PATH, index=False)
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"saved: {OOF_PATH.relative_to(PROJECT_ROOT)}")
print(f"saved: {TEST_PRED_PATH.relative_to(PROJECT_ROOT)}")
print(f"saved: {SCORES_PATH.relative_to(PROJECT_ROOT)}")
print(f"saved: {SUBMISSION_PATH.relative_to(PROJECT_ROOT)}")
print(submission[TARGET].describe(percentiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]))
