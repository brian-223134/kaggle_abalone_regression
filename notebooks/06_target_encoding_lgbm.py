# %% [markdown]
# # Target Encoding + LGBM
#
# Blog-inspired experiment for the high-impact features described in:
# https://here-lives-mummy.tistory.com/25
#
# The key rule here is that target encodings are fit inside each CV fold. This
# prevents validation rows from seeing target statistics computed from
# themselves.

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

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROCEED_DIR = PROJECT_ROOT / "data" / "proceed"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCEED_DIR / "train_fe_v2.csv"
TEST_PATH = PROCEED_DIR / "test_fe_v2.csv"

OOF_PATH = PROCEED_DIR / "oof_v3_lgbm_target_encoding.csv"
TEST_PRED_PATH = PROCEED_DIR / "test_pred_v3_lgbm_target_encoding.csv"
SCORES_PATH = PROCEED_DIR / "scores_v3_lgbm_target_encoding.csv"
SUBMISSION_PATH = SUBMISSION_DIR / "submission_v3_lgbm_target_encoding.csv"


# %%
def rmsle(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def make_stratification_bins(target: pd.Series, q: int = 10) -> np.ndarray:
    return pd.qcut(target, q=q, labels=False, duplicates="drop").astype(int).to_numpy()


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
    """Add target mean features fit from fit_X/fit_y only."""
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
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

y = train[TARGET].astype(float).to_numpy()
feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]
X = train[feature_cols]
X_test = test[feature_cols]
y_bins = make_stratification_bins(train[TARGET])

print(f"train: {train.shape}")
print(f"test : {test.shape}")
print(f"features: {len(feature_cols)}")


# %%
experiment_specs = [
    {"experiment": "v2_lgbm_log", "use_target_encoding": False},
    {"experiment": "v2_te_lgbm_log", "use_target_encoding": True},
]

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_predictions: dict[str, np.ndarray] = {}
score_rows: list[dict[str, float | int | str]] = []

for spec in experiment_specs:
    experiment = spec["experiment"]
    use_target_encoding = bool(spec["use_target_encoding"])
    start = perf_counter()
    oof = np.zeros(len(train), dtype=float)
    fold_scores: list[float] = []
    best_iterations: list[int] = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y_bins), start=1):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        if use_target_encoding:
            X_train = add_target_encoding_features(X_train, X_train, y[train_idx])
            X_valid = add_target_encoding_features(X_valid, X.iloc[train_idx], y[train_idx])

        model = make_lgbm(seed=RANDOM_STATE + fold)
        model.fit(
            X_train,
            np.log1p(y[train_idx]),
            eval_set=[(X_valid, np.log1p(y[valid_idx]))],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )

        best_iteration = int(model.best_iteration_ or model.n_estimators)
        pred = np.expm1(model.predict(X_valid, num_iteration=best_iteration))
        pred = np.clip(pred, 0, None)
        oof[valid_idx] = pred

        score = rmsle(y[valid_idx], pred)
        fold_scores.append(score)
        best_iterations.append(best_iteration)
        score_rows.append(
            {
                "experiment": experiment,
                "fold": fold,
                "rmsle": score,
                "best_iteration": best_iteration,
            }
        )
        print(f"{experiment:18s} fold {fold}: RMSLE={score:.6f}, best_iter={best_iteration}")

    oof_predictions[experiment] = oof
    summary = {
        "experiment": experiment,
        "fold": "mean",
        "rmsle": float(np.mean(fold_scores)),
        "rmsle_std": float(np.std(fold_scores)),
        "oof_rmsle": rmsle(y, oof),
        "best_iteration": int(round(np.mean(best_iterations))),
        "fit_seconds": perf_counter() - start,
    }
    score_rows.append(summary)
    print(
        f"{experiment:18s} mean={summary['rmsle']:.6f}, "
        f"std={summary['rmsle_std']:.6f}, oof={summary['oof_rmsle']:.6f}"
    )

scores = pd.DataFrame(score_rows)
print(scores.to_string(index=False))


# %%
best_experiment = min(
    (row for row in score_rows if row["fold"] == "mean"),
    key=lambda row: float(row["oof_rmsle"]),
)
best_iteration = int(best_experiment["best_iteration"])
best_name = str(best_experiment["experiment"])

print(f"best experiment: {best_name}")
print(f"best OOF RMSLE: {best_experiment['oof_rmsle']:.6f}")
print(f"final n_estimators: {best_iteration}")


# %%
final_X = X
final_X_test = X_test
if best_name == "v2_te_lgbm_log":
    final_X = add_target_encoding_features(X, X, y)
    final_X_test = add_target_encoding_features(X_test, X, y)

final_model = make_lgbm(seed=RANDOM_STATE, n_estimators=best_iteration)
final_model.fit(final_X, np.log1p(y))
test_pred = np.clip(np.expm1(final_model.predict(final_X_test)), 0, None)

submission = test[[ID_COL]].copy()
submission[TARGET] = test_pred

oof_df = train[[ID_COL, TARGET]].copy()
for experiment, pred in oof_predictions.items():
    oof_df[f"pred_{experiment}"] = pred

test_pred_df = test[[ID_COL]].copy()
test_pred_df[f"pred_{best_name}"] = test_pred

oof_df.to_csv(OOF_PATH, index=False)
test_pred_df.to_csv(TEST_PRED_PATH, index=False)
scores.to_csv(SCORES_PATH, index=False)
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"saved: {OOF_PATH.relative_to(PROJECT_ROOT)}")
print(f"saved: {TEST_PRED_PATH.relative_to(PROJECT_ROOT)}")
print(f"saved: {SCORES_PATH.relative_to(PROJECT_ROOT)}")
print(f"saved: {SUBMISSION_PATH.relative_to(PROJECT_ROOT)}")
print(submission[TARGET].describe(percentiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]))
