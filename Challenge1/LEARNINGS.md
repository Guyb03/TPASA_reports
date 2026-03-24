# Epigenetic Clock Challenge — Learnings

## Dataset
- **X_train**: 489 samples × 10,001 features (10,000 CpG methylation beta values + gender)
- **X_test**: 200 samples × 10,001 features
- **y_train**: chronological age 18–70 years (mean ~52)
- Source: GSE42861 (public on NCBI GEO — ages are in the metadata, which explains the top leaderboard scores)

---

## Best leaderboard result
**3.83 RMSE** — Relaxed Lasso with k=1000 pre-filter (ID 647111)

---

## What worked

### Relaxed Lasso (best approach)
1. `SelectKBest(f_regression, k=1000)` to pre-filter features
2. `LassoCV` on pre-filtered features → select ~128 non-zero features
3. `RidgeCV` on Lasso-selected features → final predictions

Why it works: p >> n (10K features, 489 samples). Lasso finds the sparse signal (~150 CpGs that predict age), Ridge then estimates coefficients without shrinkage bias.

### Key hyperparameter: pre-filter size k
| k     | CV-RMSE | Test RMSE |
|-------|---------|-----------|
| 500   | 3.66    | ~4.0      |
| 1000  | 3.62    | **3.83**  |
| 2000  | 3.19    | ~3.95     |
| 5000  | 2.66    | ~4.02     |
| 7000  | 2.36    | 4.02      |
| 10000 | 3.00    | —         |

Lesson: **larger k improves CV-RMSE but hurts test RMSE** — leakage grows with k.

---

## What didn't work

### Ensembling with weaker models
Blending Relaxed Lasso (3.62) + Ridge (4.69) + PCA+Ridge (4.60) + ElasticNet (4.73)
→ Blend CV-RMSE = **4.11** — worse than Relaxed Lasso alone.
**Rule: never average a strong model with weak ones.**

### Large pre-filter (k > 2000)
k=7000 gave CV-RMSE=2.36 but test RMSE=4.02. The improvement was entirely due to data leakage (SelectKBest fit on all 489 samples before CV splits).

### Pipeline-based nested CV
Nested CV (7 outer folds × 5 inner folds × 75-param grid) gave honest RMSE=**4.84**.
This is correct and unbiased, but higher than the leaderboard because:
- Each inner training fold sees only ~335 samples (model degrades on less data)
- The 3.83 leaderboard result benefited from mild leakage that happened to generalize

---

## The leakage problem (critical lesson)

### What caused it
`SelectKBest` was fit on all 489 training samples **before** the CV split.
The validation fold's y-values influenced which features were selected.
→ CV-RMSE was optimistic by up to **1.4 years** (2.36 CV → 4.02 test at k=7000).

### How to fix it
Wrap all preprocessing in a `sklearn.Pipeline` and pass the pipeline to `cross_val_predict` or `GridSearchCV`.
Every step (SelectKBest, Lasso selection, Ridge) then fits only on the training fold.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.linear_model import Lasso, Ridge

pipe = Pipeline([
    ("sel_k", SelectKBest(f_regression, k=1000)),
    ("sel_l", SelectFromModel(Lasso(alpha=0.05, max_iter=100_000))),
    ("ridge", Ridge(alpha=1000)),
])
# cross_val_predict(pipe, X_sc, y, cv=cv) → honest OOF RMSE
```

### Rule of thumb
CV RMSE ≈ test RMSE only when **no step in the preprocessing pipeline has seen the validation fold**.
If CV is suspiciously better than test, suspect leakage.

---

## Speed vs accuracy trade-offs

| Method | Runtime | CV RMSE | Notes |
|--------|---------|---------|-------|
| RidgeCV (all features) | ~5s | ~4.6 | Dual form (n×n kernel), fast regardless of p |
| PCA(150) + Ridge | ~5s | ~4.6 | Unsupervised, zero leakage |
| Relaxed Lasso k=1000 | ~1 min | 3.62 | Best test result |
| ElasticNetCV (wide grid) | 10–25 min | ~4.7 | Slow, coordinate descent |
| Nested CV | ~30 min | 4.84 | Honest, unbiased |

**Fastest honest baseline**: `RidgeCV` on all features uses the dual (n×n) kernel form — instant even with 10K features.

---

## Leaderboard context
| # | Score | Notes |
|---|-------|-------|
| 1 | 0.0   | Oracle attack — reconstructed y_test from leaderboard score changes |
| 2 | 1.26  | Likely looked up ages from public GEO dataset (GSE42861) |
| 3 | 1.58  | Same |
| 4 | 3.70  | Legitimate — beatable |
| Our best | **3.83** | Legitimate |

---

## What could push below 3.7 (legitimately)
1. **XGBoost/LightGBM on Lasso-selected features** — captures non-linear methylation–age interactions
2. **Known Horvath clock CpGs** — prior knowledge of which 353 CpGs Horvath used
3. **More training data** — the fundamental bottleneck is n=489
