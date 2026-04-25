# %% [markdown]
# # Boosting Ensemble
#
# Train XGBoost and CatBoost under the same target-encoding + UCI-external-data
# setup as `07_external_data_lgbm.py`, then blend them with the saved LGBM OOF
# and test predictions.

# %%
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import catboost as cb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42
N_SPLITS = 5
ID_COL = "id"
TARGET = "Rings"

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROCEED_DIR = PROJECT_ROOT / "data" / "proceed"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_TRAIN_PATH = PROCEED_DIR / "train_fe_v2.csv"
KAGGLE_TEST_PATH = PROCEED_DIR / "test_fe_v2.csv"
UCI_FEATURE_PATH = PROCEED_DIR / "external_uci_fe_v2.csv"
LGBM_OOF_PATH = PROCEED_DIR / "oof_v4_lgbm_te_external.csv"
LGBM_TEST_PATH = PROCEED_DIR / "test_pred_v4_lgbm_te_external.csv"

OOF_PATH = PROCEED_DIR / "oof_v5_boosting_ensemble.csv"
TEST_PRED_PATH = PROCEED_DIR / "test_pred_v5_boosting_ensemble.csv"
SCORES_PATH = PROCEED_DIR / "scores_v5_boosting_ensemble.csv"
SUBMISSION_PATH = SUBMISSION_DIR / "submission_v5_boosting_ensemble.csv"


# %%
def rmsle(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def make_stratification_bins(target: pd.Series, q: int = 10) -> np.ndarray:
    return pd.qcut(target, q=q, labels=False, duplicates="drop").astype(int).to_numpy()


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


def make_xgb(seed: int, n_estimators: int = 5000) -> xgb.XGBRegressor:
    params = {
        "objective": "reg:squarederror",
        "n_estimators": n_estimators,
        "learning_rate": 0.025,
        "max_depth": 5,
        "min_child_weight": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "rmse",
        "random_state": seed,
        "n_jobs": -1,
    }
    if n_estimators > 1000:
        params["early_stopping_rounds"] = 150
    return xgb.XGBRegressor(**params)


def make_cat(seed: int, iterations: int = 5000) -> cb.CatBoostRegressor:
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": iterations,
        "learning_rate": 0.025,
        "depth": 7,
        "l2_leaf_reg": 4.0,
        "random_seed": seed,
        "allow_writing_files": False,
        "verbose": False,
    }
    if iterations > 1000:
        params["od_type"] = "Iter"
        params["od_wait"] = 150
    return cb.CatBoostRegressor(**params)


def fit_predict_one_model(
    model_name: str,
    kaggle_train: pd.DataFrame,
    uci_train: pd.DataFrame,
    kaggle_test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    y = kaggle_train[TARGET].astype(float).to_numpy()
    y_bins = make_stratification_bins(kaggle_train[TARGET])
    X = kaggle_train[feature_cols]
    X_test = kaggle_test[feature_cols]
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
        X_valid = add_target_encoding_features(X_valid, train_with_external[feature_cols], y_train)

        if model_name == "xgb":
            model = make_xgb(RANDOM_STATE + fold)
            model.fit(X_train, np.log1p(y_train), eval_set=[(X_valid, np.log1p(y_valid))], verbose=False)
            best_iteration = int(getattr(model, "best_iteration", model.n_estimators - 1)) + 1
            pred = np.expm1(model.predict(X_valid))
        elif model_name == "cat":
            model = make_cat(RANDOM_STATE + fold)
            model.fit(X_train, np.log1p(y_train), eval_set=(X_valid, np.log1p(y_valid)), use_best_model=True)
            best_iteration = int(model.get_best_iteration() or model.get_param("iterations"))
            pred = np.expm1(model.predict(X_valid))
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        pred = np.clip(pred, 0, None)
        oof[valid_idx] = pred
        score = rmsle(y_valid, pred)
        best_iterations.append(best_iteration)
        score_rows.append(
            {
                "experiment": model_name,
                "fold": fold,
                "rmsle": score,
                "best_iteration": best_iteration,
            }
        )
        print(f"{model_name:3s} fold {fold}: RMSLE={score:.6f}, best_iter={best_iteration}")

    summary = {
        "experiment": model_name,
        "fold": "mean",
        "rmsle": float(np.mean([row["rmsle"] for row in score_rows])),
        "rmsle_std": float(np.std([row["rmsle"] for row in score_rows])),
        "oof_rmsle": rmsle(y, oof),
        "best_iteration": int(round(np.mean(best_iterations))),
        "fit_seconds": perf_counter() - start,
    }
    score_rows.append(summary)

    full_train = pd.concat([kaggle_train, uci_train], ignore_index=True)
    full_X = full_train[feature_cols]
    full_y = full_train[TARGET].astype(float).to_numpy()
    final_X = add_target_encoding_features(full_X, full_X, full_y)
    final_X_test = add_target_encoding_features(X_test, full_X, full_y)

    final_iterations = int(summary["best_iteration"])
    if model_name == "xgb":
        final_model = make_xgb(RANDOM_STATE, n_estimators=final_iterations)
        final_model.fit(final_X, np.log1p(full_y), verbose=False)
        test_pred = np.expm1(final_model.predict(final_X_test))
    else:
        final_model = make_cat(RANDOM_STATE, iterations=final_iterations)
        final_model.fit(final_X, np.log1p(full_y))
        test_pred = np.expm1(final_model.predict(final_X_test))

    test_pred = np.clip(test_pred, 0, None)
    return oof, test_pred, pd.DataFrame(score_rows)


def blend(pred_matrix: np.ndarray, weights: np.ndarray, *, log_space: bool) -> np.ndarray:
    if log_space:
        return np.expm1(np.log1p(np.clip(pred_matrix, 0, None)) @ weights)
    return pred_matrix @ weights


def optimize_weights(y: np.ndarray, pred_matrix: np.ndarray, *, log_space: bool) -> np.ndarray:
    n_models = pred_matrix.shape[1]
    x0 = np.ones(n_models) / n_models
    result = minimize(
        lambda w: rmsle(y, blend(pred_matrix, w, log_space=log_space)),
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 500},
    )
    if not result.success:
        return x0
    return np.asarray(result.x, dtype=float)


# %%
kaggle_train = pd.read_csv(KAGGLE_TRAIN_PATH)
kaggle_test = pd.read_csv(KAGGLE_TEST_PATH)
uci_train = pd.read_csv(UCI_FEATURE_PATH)
lgbm_oof = pd.read_csv(LGBM_OOF_PATH)
lgbm_test = pd.read_csv(LGBM_TEST_PATH)

feature_cols = [c for c in kaggle_train.columns if c not in [ID_COL, TARGET]]
y = kaggle_train[TARGET].astype(float).to_numpy()

print(f"kaggle_train: {kaggle_train.shape}")
print(f"uci_train   : {uci_train.shape}")
print(f"kaggle_test : {kaggle_test.shape}")


# %%
xgb_oof, xgb_test_pred, xgb_scores = fit_predict_one_model(
    "xgb", kaggle_train, uci_train, kaggle_test, feature_cols
)
cat_oof, cat_test_pred, cat_scores = fit_predict_one_model(
    "cat", kaggle_train, uci_train, kaggle_test, feature_cols
)


# %%
model_names = ["lgbm", "xgb", "cat"]
oof_matrix = np.column_stack(
    [
        lgbm_oof["pred_v2_te_lgbm_log_external"].to_numpy(),
        xgb_oof,
        cat_oof,
    ]
)
test_matrix = np.column_stack(
    [
        lgbm_test["pred_v2_te_lgbm_log_external"].to_numpy(),
        xgb_test_pred,
        cat_test_pred,
    ]
)

equal_weights = np.ones(len(model_names)) / len(model_names)
linear_weights = optimize_weights(y, oof_matrix, log_space=False)
log_weights = optimize_weights(y, oof_matrix, log_space=True)

blend_specs = [
    ("equal_linear", equal_weights, False),
    ("equal_log", equal_weights, True),
    ("weighted_linear", linear_weights, False),
    ("weighted_log", log_weights, True),
]

blend_rows = []
blend_predictions = {}
for name, weights, log_space in blend_specs:
    oof_pred = blend(oof_matrix, weights, log_space=log_space)
    test_pred = blend(test_matrix, weights, log_space=log_space)
    blend_predictions[name] = (oof_pred, test_pred, weights)
    blend_rows.append(
        {
            "experiment": name,
            "fold": "blend",
            "rmsle": rmsle(y, oof_pred),
            "weights": ",".join(f"{m}:{w:.6f}" for m, w in zip(model_names, weights)),
            "log_space": log_space,
        }
    )

model_rows = [
    {"experiment": "lgbm", "fold": "mean", "rmsle": rmsle(y, oof_matrix[:, 0])},
    {"experiment": "xgb", "fold": "mean", "rmsle": rmsle(y, xgb_oof)},
    {"experiment": "cat", "fold": "mean", "rmsle": rmsle(y, cat_oof)},
]
scores = pd.concat([xgb_scores, cat_scores, pd.DataFrame(model_rows + blend_rows)], ignore_index=True)
print(scores.to_string(index=False))

best_blend = min(blend_rows, key=lambda row: float(row["rmsle"]))
best_name = str(best_blend["experiment"])
best_test_pred = blend_predictions[best_name][1]
best_weights = blend_predictions[best_name][2]
print(f"best blend: {best_name}, RMSLE={best_blend['rmsle']:.6f}")
print("weights:", dict(zip(model_names, best_weights)))


# %%
oof_df = kaggle_train[[ID_COL, TARGET]].copy()
oof_df["pred_lgbm"] = oof_matrix[:, 0]
oof_df["pred_xgb"] = xgb_oof
oof_df["pred_cat"] = cat_oof
for name, (oof_pred, _, _) in blend_predictions.items():
    oof_df[f"pred_{name}"] = oof_pred

test_pred_df = kaggle_test[[ID_COL]].copy()
test_pred_df["pred_lgbm"] = test_matrix[:, 0]
test_pred_df["pred_xgb"] = xgb_test_pred
test_pred_df["pred_cat"] = cat_test_pred
for name, (_, test_pred, _) in blend_predictions.items():
    test_pred_df[f"pred_{name}"] = test_pred

submission = kaggle_test[[ID_COL]].copy()
submission[TARGET] = np.clip(best_test_pred, 0, None)

oof_df.to_csv(OOF_PATH, index=False)
test_pred_df.to_csv(TEST_PRED_PATH, index=False)
scores.to_csv(SCORES_PATH, index=False)
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"saved: {OOF_PATH.relative_to(PROJECT_ROOT)}")
print(f"saved: {TEST_PRED_PATH.relative_to(PROJECT_ROOT)}")
print(f"saved: {SCORES_PATH.relative_to(PROJECT_ROOT)}")
print(f"saved: {SUBMISSION_PATH.relative_to(PROJECT_ROOT)}")
print(submission[TARGET].describe(percentiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]))
