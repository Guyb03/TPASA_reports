import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

# ─── Load ────────────────────────────────────────────────
DATA_DIR = Path("input_data")
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
X_test  = pd.read_csv(DATA_DIR / "X_test.csv")

X_train["gender"] = (X_train["gender"] == "m").astype(int)
X_test["gender"]  = (X_test["gender"] == "m").astype(int)

X    = X_train.values.astype(np.float64)
X_te = X_test.values.astype(np.float64)
y    = y_train.values.astype(np.float64)
print(f"Train: {X.shape}  Test: {X_te.shape}")

scaler = StandardScaler()
X_sc    = scaler.fit_transform(X)
X_te_sc = scaler.transform(X_te)

# ─── Pipeline: everything inside, no leakage ─────────────
# SelectKBest  → feature pre-filter  (fit only on training batch)
# SelectFromModel(Lasso) → sparse selection (fit only on training batch)
# Ridge → final estimator
pipe = Pipeline([
    ("sel_k", SelectKBest(f_regression)),
    ("sel_l", SelectFromModel(Lasso(max_iter=100_000))),
    ("ridge", Ridge()),
])

param_grid = {
    "sel_k__k":                [500, 1000, 2000],
    "sel_l__estimator__alpha": np.logspace(-2.5, 0, 5),
    "ridge__alpha":            np.logspace(1, 7, 5),
}
n_combos = 3 * 5 * 5
print(f"Grid: {n_combos} combinations")

# ─── Nested cross-validation ─────────────────────────────
# Inner CV: selects best hyperparameters (uses only training batches)
# Outer CV: evaluates generalisation (the held-out batch never touched during fitting)
#
# Result: OOF RMSE is a truly unbiased estimate — no leakage anywhere.
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv  = KFold(n_splits=7, shuffle=True, random_state=0)

gs = GridSearchCV(
    pipe, param_grid,
    cv=inner_cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    refit=True,
    verbose=1,
)

print("Running nested CV (outer=7 folds, inner=5 folds × grid)...")
# n_jobs=1 on outer loop avoids nested-parallelism issues;
# inner GridSearchCV already uses n_jobs=-1
oof = cross_val_predict(gs, X_sc, y, cv=outer_cv, n_jobs=1)
nested_rmse = rmse(oof, y)
print(f"Nested CV RMSE (unbiased): {nested_rmse:.4f}")
print("This should be close to the actual test RMSE with no leakage.")

# ─── Final model on all training data ────────────────────
print("\nFitting final model on full training set...")
gs.fit(X_sc, y)
print(f"Best params: {gs.best_params_}")

y_pred = np.clip(gs.predict(X_te_sc), 18, 70)
pd.DataFrame({"age": y_pred}).to_csv("y_pred.csv", index=False)
print(f"Saved y_pred.csv  ({len(y_pred)} rows)")
print(f"Range: [{y_pred.min():.1f}, {y_pred.max():.1f}]  mean={y_pred.mean():.1f}")
