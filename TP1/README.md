# Report for TP1

## Exercise 1: Preliminary analysis of the data

**(a)** In order to remove the first and last columns, we can use the following code:

```python
import pandas as pd
df = pd.read_csv('prostates.csv', sep="\t")

df = df.drop(columns=[df.columns[0], df.columns[-1]])
```

**(b)** The pairplot function produce an 8 by 8 grid of scatter plots, where each plot shows the relationship between two variables of the dataset (from which the first and last columns have been removed).
The diagonal of the grid is just the label of the variable ploted in the corresponding row and column.
We could discuss the pertinence of plotting both (x1,x2) and (x2,x1) for each pair of variables, since only one is enough to understand if one variable is correlated to the other hence the lower triangular part of the grid is redundant (it is only the mirror of the upper triangle by the y=x line).
Line by line code explanation :
it starts by defining a figure size if None is given, it then takes the number of variables in the dataframe with df.shape[1].
Then it creates figures and axes for the pairplot using plt.subplots, where the number of rows and columns is equal to the number of variables.
The subplots_adjust function is used to adjust the spacing between the subplots.
Finally, it iterates through each pair of variables (i, j) and creates a scatter plot for each pair using set_xticks and set_yticks to set the ticks for the x and y axes with a margin of 0.1 and in the (i, j) spot of the grid (ax[i][j]), and if the i is different from j, it plots the scatter plot of the the i-th variable (using its name with df.columns[i]) of the dataframe on the x-axis and of the j-th variable on the y-axis, otherwise it just writes the name of the i-th variable in the diagonal of the grid.
It ends by returning the figure and axes of the pair plot.

**(c)** The variables most correlated with lcavol appear to be `lpsa` and `lcp`. `lweight` and `pgg45` show a moderate relationship. `lbph`, `age`, and `svi` appear weakly correlated. The mass of points at 0 for `lcp`, `lbph`, and `pgg45` reflects censoring in those variables rather than absence of correlation.
Also, if we had to infer the datatype of the variables based on the scatter plots we would say that all the variables are numerical. `lcavol`, `lweight`, `age`, `lbph`, `lcp`, `pgg45` and `lpsa` are continuous variables, while `svi` is binary and `gleason` ordinal. We can also see that the distribution of the variables is not normal, and that there are some outliers in the data.

**(d)** The formula of correlation is given by:
$$\rho(X, Y) = \frac{Cov(X, Y)}{\sigma_X \sigma_Y}$$

Where $Cov(X, Y)=\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$ is the covariance of X and Y, and $\sigma_X$ and $\sigma_Y$ are the standard deviations of X and Y respectively ($\sigma_X = \sqrt{Var(X)}$).
Here is a possible implementation using numpy:

```python
import numpy as np

def correlation(X, Y):
    cov = np.mean((X - np.mean(X)) * (Y - np.mean(Y)))
    return cov / (np.std(X) * np.std(Y))

for col in df.columns:
    print(f"Correlation between lcavol and {col}:\n    using numpy: {correlation(df['lcavol'], df[col])},\n    using df.corr: {df['lcavol'].corr(df[col])}\n")
```
output:
```

ENSIMAG – Grenoble INP – UGA - Academic year 2025-2026
Introduction to Statistical Learning and Applications (website)

    Pedro L. C. Rodrigues -- pedro.rodrigues@inria.fr

    Isabella Costa Maia -- isabella.costa-maia@grenoble-inp.fr

⚠️ General guidelines for TPs

Each team shall upload its report on Teide before the deadline indicated at the course website. Please include the name of all members of the team on top of your report. The report should contain graphical representations and explanatory text. For each graph, axis names should be provided as well as a legend when it is appropriate. Figures should be explained by a few sentences in the text. Answer to the questions in order and refer to the question number in your report. Computations and graphics have to be performed in python. The report should be written as a jupyter notebook. This is a file format that allows users to format documents containing text written in markdown and python instructions. You should include all of the python instructions that you have used in the document so that it may be possible to replicate your results.
🖥️ TP1: Analysis of prostate cancer data

A medical study done on patients with prostate cancer aims to analyze the correlation between the prostate tumor volume and a set of clinical and morphometric variables. These variables include prostate specific antigens, a biomarker for prostate cancer, and a number of clinical measures (age, prostate weight, etc). The goal of this lab is to build a regression model to predict the severity of cancer, expressed by logarithm of the tumor volume -- lcavol variable -- from the following predictors:

    lpsa: log of a prostate specific antigen
    lweight: log of prostate weight
    age: age of the patient
    lbph: log of benign prostatic hyperplasia amount
    svi: seminal vesicle invasion
    lcp: log of capsular penetration
    gleason: Gleason score (score on a cancer prognosis test)
    pgg45: percent of Gleason scores 4 or 5

The file prostate.data, available here, contains measures of the logarithm of the tumor volume and of the 8 predictors for 97 patients. This file also contains an additional variable, called train, which will not be used and has to be removed.
▶️ Exercise 1: Preliminary analysis of the data (1 points)

(a) Download the file prostate.data and store it in your current folder. Read the dataset in using pandas as per

import pandas as pd
df = pd.read_csv("prostate.data", sep="\t")

check how to use function df.drop to remove the first and last columns of df containing useless tags.
Selection deleted
import pandas as pd
df = pd.read_csv("prostate.data", sep="\t")

df = df.drop(columns=[df.columns[0], df.columns[-1]])

(b) The function defined below generates scatterplots (clouds of points) between all pairs of variables, allowing us to visually analyse the correlations between all variables in the dataframe. Explain what each line of function pairplot does and how it affects the final plot. You can use command help in the python shell to get the documentation of each function.

# import the main plotting library for python
import matplotlib.pyplot as plt

# make a pairplot from scratch
def pairplot(df, figsize=None):
    if figsize is None:
        figsize = (7.10, 6.70)
    n_vars = df.shape[1]
    fig, ax = plt.subplots(figsize=figsize, ncols=n_vars, nrows=n_vars)
    plt.subplots_adjust(
        wspace=0.10, hspace=0.10, left=0.05, right=0.95, bottom=0.05, top=0.95)
    for i in range(n_vars):
        for j in range(n_vars):
            axij = ax[i][j]
            i_name = df.columns[i]
            j_name = df.columns[j]
            axij.set_xticks([])
            axij.set_yticks([])
            axij.margins(0.1)
            if i != j:
                axij.scatter(df[i_name], df[j_name], s=10)
            else:
                axij.text(0.5, 0.5, i_name, fontsize=10,
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform = axij.transAxes)        
    return fig, ax

fig, ax = pairplot(df)
plt.show()

# import the main plotting library for python
import matplotlib.pyplot as plt

# make a pairplot from scratch
def pairplot(df, figsize=None):
    if figsize is None:
        figsize = (7.10, 6.70)
    n_vars = df.shape[1]
    fig, ax = plt.subplots(figsize=figsize, ncols=n_vars, nrows=n_vars)
    plt.subplots_adjust(
        wspace=0.10, hspace=0.10, left=0.05, right=0.95, bottom=0.05, top=0.95)
    for i in range(n_vars):
        for j in range(n_vars):
            axij = ax[i][j]
            i_name = df.columns[i]
            j_name = df.columns[j]
            axij.set_xticks([])
            axij.set_yticks([])
            axij.margins(0.1)
            if i != j:
                axij.scatter(df[i_name], df[j_name], s=10)
            else:
                axij.text(0.5, 0.5, i_name, fontsize=10,
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform = axij.transAxes)        
    return fig, ax

fig, ax = pairplot(df)
plt.show()

(c) Based on the generated figure, identify which variables seem the most correlated to lcavol. Also, infer the datatype for each of the predictors.

(d) Recall the formula of correlation between two vectors and implement it with numpy. Explain the difference of your result when compared to df.corr(). Change the function pairplotto show the correlation coefficient on the upper triangle of the subplots. (Bonus) Use locally weighted scatterplot smoothing (LOWESS) to see the trends between pairs of features with statsmodels.nonparametric.smoothers_lowess.
Selection deleted
import numpy as np

def correlation(X, Y):
    cov = np.mean((X - np.mean(X)) * (Y - np.mean(Y)))
    return cov / (np.std(X) * np.std(Y))

for col in df.columns:
    print(f"Correlation between lcavol and {col}:\n   using numpy: {correlation(df['lcavol'], df[col])},\n   using df.corr: {df['lcavol'].corr(df[col])}\n")

Correlation between lcavol and lcavol:
   using numpy: 1.0,
   using df.corr: 1.0

Correlation between lcavol and lweight:
   using numpy: 0.28052138000983295,
   using df.corr: 0.2805213800098328

Correlation between lcavol and age:
   using numpy: 0.22499987914993153,
   using df.corr: 0.22499987914993153

Correlation between lcavol and lbph:
   using numpy: 0.027349703303625298,
   using df.corr: 0.027349703303625354

Correlation between lcavol and svi:
   using numpy: 0.5388450022638601,
   using df.corr: 0.5388450022638602

Correlation between lcavol and lcp:
   using numpy: 0.6753104840558688,
   using df.corr: 0.6753104840558688

Correlation between lcavol and gleason:
   using numpy: 0.4324170558258538,
   using df.corr: 0.43241705582585366

Correlation between lcavol and pgg45:
   using numpy: 0.4336522490280904,
   using df.corr: 0.43365224902809046

Correlation between lcavol and lpsa:
   using numpy: 0.734460326213673,
   using df.corr: 0.734460326213673

```

There is no difference between the correlation computed with numpy and the one computed with df.corr() because they are both using the same formula for correlation and probably the same method for computation : numpy and pandas (which is built on top of numpy) are both using the same underlying implementation.
Now, if we were to change the function ``pairplot`` to show correlation coefficient on the upper truangle of the subplots :

```python
from statsmodels.nonparametric.smoothers_lowess import lowess

def pairplot(df, figsize=None):
        if figsize is None:
        figsize = (7.10, 6.70)
    n_vars = df.shape[1]
    fig, ax = plt.subplots(figsize=figsize, ncols=n_vars, nrows=n_vars)
    plt.subplots_adjust(
        wspace=0.10, hspace=0.10, left=0.05, right=0.95, bottom=0.05, top=0.95)
    for i in range(n_vars):
        for j in range(n_vars):
            axij = ax[i][j]
            i_name = df.columns[i]
            j_name = df.columns[j]
            axij.set_xticks([])
            axij.set_yticks([])
            axij.margins(0.1)
            if i != j:
                axij.scatter(df[i_name], df[j_name], s=10)
                if i < j:
                    corr = correlation(df[i_name], df[j_name]) # or df[i_name].corr(df[j_name]) to get the exact same result
                    axij.text(0.5, 0.5, f"{corr:.2f}", fontsize=10,
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform = axij.transAxes)
                else:
                    smoothed = lowess(df[i_name], df[j_name], frac=0.6)
                    axij.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=1.5)
            else:
                axij.text(0.5, 0.5, i_name, fontsize=10,
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform = axij.transAxes)        
    return fig, ax

fig, ax = pairplot(df)
plt.show()
```
The lower triangle of the grid will now show the LOWESS smoothed curve for each pair of variables, which clearly highlights the correlation between 2 variables, and in our case between `lcavol` and `lpsa` and also between `lcavol` and `lcp`. The upper triangle will show the correlation coefficient for each pair of variables, which allows us to quickly identify which pairs of variables are more correlated.


## Exercise 2: Linear regression

**(a)** The mathematical formula of the regression model is :
$$\by = \bX \bbeta + \bepsilon$$
where $\by$ is the vector estimated by the model, $\bX$ is the matrix of the predictors, $\bbeta$ is the vector of the coefficients and $\bepsilon$ is the noise vector.
This can be developed as :
$$\begin{bmatrix}\hat{y_1} \\\hat{y_2} \\\vdots \\hat{y_n}\end{bmatrix} = \begin{bmatrix}x_{11} & x_{12} & \cdots & x_{1p} \\\\x_{21} & x_{22} & \cdots & x_{2p} \\\\vdots & \vdots & \ddots & \vdots \\\\x_{n1} & x_{n2} & \cdots & x_{np}\endbmatrix} \begin{bmatrix}\beta_1 \\\beta_2 \\\vdots \\\beta_p\end{bmatrix} + \begin{bmatrix}\epsilon_1 \\\epsilon_2 \\\vdots \\\epsilon_n\end{bmatrix}$$
where $\hat{y_i}$ is the predicted value for the i-th observation, $x_{ij}$ is the value of the j-th predictor for the i-th observation, $\beta_j$ is the coefficient for the j-th predictor and $\epsilon_i$ is the noise for the i-th observation.

**(b)** 
```python
# encode the categorical features with dummy variables
df_enc = pd.get_dummies(df, dtype=np.float64)
# to drop one dummy column for each predictor
df_enc = df_enc.drop(columns=['svi_0', 'gleason_6'])
# add a column of ones to the dataframe
df_enc['intercept'] = 1
# extract the dataframe for predictors
X = df_enc.drop(columns=['lcavol'])
# get the observed values to predict
y = df['lcavol']
```
Fixing a `dtype` in `pd.get_dummies` is important to avoid types that are not compatible with the regression model, such as object or category types. Choosing `dtype=np.float64` ensures that the resulting dummy variables are of a numeric type that can be used in the regression model without causing errors or issues with the calculations.
We drop `svi_0` and `gleason_6` after the encoding to avoid the dummy variable trap: keeping all dummy categories would introduce perfect multicollinearity, as the dummies for each variable would sum to 1 for every observation, making the design matrix singular and non-invertible. The dropped categories (`svi_0` and `gleason_6`) become the reference categories, and the coefficients of the remaining dummies are interpreted relative to them.
Adding a column of ones to the predictors is necessary to account for the intercept term in the regression model. Without it, the regression would be forced through the origin. With it, the intercept represents the expected value of `lcavol` when all predictors are equal to zero, giving the model the flexibility to fit the data without this constraint.

**(c)**
```python
# import required package
import statsmodels.api as sm
# this line does not fit the regression model per se but only builds it
model = sm.OLS(y, X)
# now we actually fit the model, e.g. calculate all of regression parameters
results = model.fit()
results.summary()
```
Since `svi` and `gleason` are the categorical variables that were one-hot encoded with a reference category dropped, their coefficients are interpreted as relative to the reference category. For `svi_1`, the coefficient represents the expected change in `lcavol` when `svi=1` compared to `svi=0`, holding all other predictors constant.
As for `gleason_7`, `gleason_8`, and `gleason_9`, each coefficient represents the expected change in lcavol compared to having `gleason=6` (the reference category dropped during one-hot encoding), also holding all other predictors constant.
For instance, having `gleason_7 = 0.312` means a patient with Gleason score 7 is expected to have `lcavol` that is 0.312 higher than a patient with Gleason score 6, all else being equal.
If `svi` and `gleason` were left as raw categorical variables without the one-hot encoding, the regression model would have treated them as continuous numeric variables, implying a linear and equally spaced effect between each level.
The $R^2$ of 0.686 means the model explains about 69% of the variance in lcavol, which is reasonable. The adjusted $R^2$ of 0.650 accounts for the number of predictors and is close, suggesting no severe overfitting. The F-statistic is highly significant (p=9.05e-18), confirming that the model as a whole is meaningful.
As remarked before in the correlation analysis, the most significant predictors are `lpsa` (0.55, p~0) and `lcp` (0.28, p~0), which is consistent with their strong correlation with `lcavol`. For instance,`age` is weakly but significantly positive (0.025, p=0.030). Also, `ppg45` is significant but with a surprisingly small negative coefficient (-0.009, p=0.040), which may be due to multicollinearity with `gleason` as they are strongly correlated. This could have a suppressor effect when controlling for other variables.
`lweight`, `lbph`, `svi_1` and all `gleason` dummies have p-values well above 0.05, meaning we cannot reject the null hypothesis that their coefficients are zero. This is surprising for `svi_1` given its moderate correlation (0.54) with `lcavol`, but it likely loses explanatory power once `lcp` and `lpsa` are since these variables correlated with each other.

**(d)**
```python
results.conf_int()
```
The confidence intervals for each coefficient confirm what we observed with the p-values.
`age`, `lcp`, `pgg45`, `lpsa` are significant while `lweigh`, `lbph`, `svi_1` and all `gleason` dummies aren't.
If the interval contains zero, directly meaning that there is a non-negligible probability that the true coefficient is zero, then the predictor is not significant. In other words, containing zero at the 5% level which is just the confidence interval version of saying p > 0.05.

**(e)** lpsa has :
- A coefficient of 0.5496, the largest among all predictors
- A p-value of essentially 0 (p=2.939241e-08 from results.pvalues)
- A 95% confidence interval of [0.370, 0.729], which does not contain zero

So we reject the null hypothesis $H_0: \beta_{lpsa} = 0$ at the 5% significance level (and in fact at any reasonable level given the p-value). This means `lpsa` has a statistically significant positive effect on `lcavol`, a one unit increase in lpsa is associated with an expected increase of 0.55 in `lcavol`, holding all other variables constant. This is also consistent with what we observed visually in the pair plot and numerically with the highest correlation (0.73) among all predictors.

**(f)**Using `inv` computes $\hat{\beta} = (X^{T}X)^{-1}X^{T}y$ by explicitly inverting the matrix, which is numerically unstable as matrix inversion amplifies floating point errors, especially when $X^TX$ is nearly singular (which happens with multicollinearity, like we have here with correlated predictors).
It is also computationally wasteful because inverting a full matrix is $O(p^3)$ and unnecessary since you only need the product $(X^TX)^{-1}X^Ty$.
`np.linalg.solve(XtX, Xty)` instead solves the linear system $X^TX \cdot \beta = X^Ty$ directly using LU decomposition, which is more numerically stable and faster since it never explicitly forms the inverse. It's the same mathematical result but a much better numerical path to get there.
Here is a possible implementation :
```python
from scipy import stats

def ols_fit(X, y):
    n, p = X.shape

    # Beta estimate : (X^T X)^{-1} X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    # Residuals and variance estimate
    y_hat = X @ beta
    residuals = y - y_hat
    sigma2 = (residuals @ residuals) / (n - p) # unbiased estimator for s^2

    # Covariance matrix of beta, standard errors, t-stats
    cov_beta = sigma2 * np.linalg.solve(XtX, np.eye(p))
    se = np.sqrt(np.diag(cov_beta))
    t_stats = beta / se

    # p-values from t-distribution with (n-p) degrees of freedom
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))

    return beta, p_values

comparison = pd.DataFrame({
    'coef (ours)':    beta,
    'coef (statsmodels)': results.params.values,
    'p-value (ours)': p_values,
    'p-value (statsmodels)': results.pvalues.values
}, index=results.params.index)

print(comparison)
```

**(g)**
```python
predicted = results.get_prediction().predicted_mean

plt.scatter(y, predicted, s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=1)  # perfect prediction line
plt.xlabel('Actual lcavol')
plt.ylabel('Predicted lcavol')
plt.title('Predicted vs Actual lcavol')
plt.show()
```

**(h)**
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(results.resid, bins=20, edgecolor='black')
axes[0].set_title('Histogram of residuals')
axes[0].set_xlabel('Residuals')
axes[0].set_ylabel('Frequency')

sm.qqplot(results.resid, line='s', ax=axes[1])
axes[1].set_title('QQ-Plot of Residuals')

plt.tight_layout()
plt.show()

print(results.ssr)
```

Looking at the histogram, the distribution is roughly bell-shaped and centered around 0, which is encouraging. However there are a few concerns — the distribution looks slightly left-skewed with some notable outliers around -2 and +1.75, and the shape is not perfectly symmetric. We can admit the residuals are approximately normally distributed, even if the histogram looks imperfect. With only 97 observations, some deviation from perfect normality is expected.

**(i)** 
```python
X_test = df_enc.drop(columns=['lcavol', 'lpsa', 'lcp'])
model_test = sm.OLS(y, X_test)
results_test = model_test.fit()
results_test.summary()
```
R² drops significantly to 0.433 (from 0.686), meaning this model only explains 43% of the variance in lcavol. The adjusted R² of 0.381 confirms the model is much weaker overall. The F-statistic remains significant (p=2.11e-08) so the model is still globally meaningful, just less powerful.
`svi_1` is now the strongest and most significant predictor (1.025, p≈0), which makes sense — without `lcp` and `lpsa` in the model, `svi` can now express its relationship with lcavol freely. `lweight` is also now significant (0.585, p=0.026) and gleason_7 becomes significant (0.801, p=0.004).
`age`, `lbph`, `pgg45`, `gleason_8`, `gleason_9` and the intercept all remain non-significant.
Comparing with the previous model, this illustrates multicollinearity. When `lpsa` and `lcp` were included, they absorbed most of the explanatory power and made `svi` and `lweight` appear non-significant. Removing them reveals the underlying relationships these variables have with `lcavol`.

## Exercice 3: Best subset selection

**(a)**
```python
RSS = []
for cols in combinations(X.columns, 2):
    model = sm.OLS(y, X[list(cols)])
    results = model.fit()
    print(f"RSS for predictor(s) {cols}: {results.ssr}\n")
    RSS.append((cols, results.ssr))

best = min(RSS, key=lambda x: x[1])
print(f"Best predictors: {best[0]}\nRSS: {best[1]}")
```
As expected the model with the smallest RSS is the one with only `lcp` and `lpsa` as predictors.

**(b)**
```python
best_per_k = {}

for k in range(1, 8):
    rss_k = []
    for cols in combinations(X.columns, k):
        model = sm.OLS(y, sm.add_constant(X[list(cols)]))
        results = model.fit()
        rss_k.append((cols, results.ssr))
    best = min(rss_k, key=lambda x: x[1])
    best_per_k[k] = best
    print(f"k={k} | Best predictors: {best[0]} | RSS: {best[1]:.4f}")

ks = list(best_per_k.keys())
rss_values = [best_per_k[k][1] for k in ks]

plt.figure(figsize=(8, 5))
plt.plot(ks, rss_values, marker='o')
plt.xlabel('Number of predictors (k)')
plt.ylabel('RSS')
plt.title('Best subset selection — RSS vs number of predictors')
plt.xticks(ks)
plt.tight_layout()
plt.show()
```
**(c)** Minimizing the RSS is not well suited for selecting the optimal model size in regression. The fundamental problem is that RSS is a measure of in-sample fit, and it decreases monotonically as more predictors are added to the model, even if those additional predictors have no true relationship with the response variable. This means that a model with all available predictors will always minimize the RSS, regardless of whether those predictors are meaningful.
This leads directly to the problem of overfitting: by chasing a lower RSS, we build a model that fits the training data very well but captures noise rather than the true underlying signal, resulting in poor generalization to new, unseen data. In other words, minimizing RSS optimizes in-sample performance at the expense of out-of-sample prediction accuracy.

## Exercise 4: Split-validation

**(a)** Split-validation works by dividing the available dataset into two distinct subsets: a training set and a test set. The model is fitted exclusively on the training set, and its performance is then evaluated on the test set, data the model has never seen during training. The test error (e.g. MSE on the test set) serves as an estimate of the true out-of-sample prediction error, and can be used to compare models of different sizes and select the optimal one.
The key reason it avoids the issue raised with RSS is that the test set is not used during model fitting. Adding an irrelevant predictor may reduce the training RSS, but it will not systematically reduce the test error.
This makes split-validation a much more honest criterion for model selection than RSS, because it directly measures what we actually care about (how well the model generalizes to new observations) rather than how well it memorizes the data it was trained on.

**(b)**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

best_per_k = {}
for k in range(0, 9):
    rss_k = []
    for cols in combinations(X.columns, k):
        if k == 0:
            model = sm.OLS(y_train, np.ones(len(y_train)))
        else:
            model = sm.OLS(y_train, sm.add_constant(X_train[list(cols)]))
        results = model.fit()
        rss_k.append((cols, results.ssr, results))
    
    best = min(rss_k, key=lambda x: x[1])
    best_per_k[k] = best

# Now compute errors
train_errors = []
test_errors  = []

for k in range(9):
    cols, rss, results = best_per_k[k]
    
    # Training MSE
    train_errors.append(results.mse_resid)
    
    # Test MSE
    if k == 0:
        X_test_k = np.ones(len(y_test))
    else:
        X_test_k = sm.add_constant(X_test[list(cols)])
    
    y_pred_test = results.predict(X_test_k)
    test_errors.append(np.mean((y_test - y_pred_test) ** 2))

plt.figure(figsize=(8, 5))
plt.plot(range(9), train_errors, marker='o', label='Train MSE')
plt.plot(range(9), test_errors,  marker='o', label='Test MSE')
plt.xlabel('Number of predictors (k)')
plt.ylabel('Mean Squared Error')
plt.title('Train vs Test Error by Model Size')
plt.xticks(range(9))
plt.legend()
plt.grid(True)
plt.show()
```
**(c)** The Train MSE decreases monotonically as expected, this is the overfitting phenomenon we discussed. The Test MSE drops sharply from k=0 to k=2, then flattens out and stabilizes around 0.64–0.68 for k=2 through k=8, with no meaningful improvement beyond k=2.
This shows as expected that the best model choice is with k=2, with predictors ('lcp', 'lpsa').
The test MSE is minimized (or near-minimized) at k=2, and adding more predictors beyond that does not reduce test error, it only reduces training error, which is a sign of overfitting. Following the parsimony principle, we prefer the simplest model that achieves competitive predictive performance.
```python
best_cols = best_per_k[2][0]  # ('lcp', 'lpsa')
model_full = sm.OLS(y, sm.add_constant(X[list(best_cols)])).fit()
print(model_full.summary())
```

**(d)** The main limitation of split-validation is its high variance: the result depends heavily on which observations happen to fall in the train vs test set. A single random split may be lucky or unlucky, leading to an unreliable estimate of the test error and potentially a different model being selected each time you run it.
```python
for seed in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    
    test_errors = []
    for k in range(9):
        cols, _, _ = best_per_k[k]
        if k == 0:
            y_pred = np.full(len(y_test), y_train.mean())
        else:
            results = sm.OLS(y_train, sm.add_constant(X_train[list(cols)])).fit()
            y_pred = results.predict(sm.add_constant(X_test[list(cols)]))
        test_errors.append(np.mean((y_test - y_pred) ** 2))
    
    best_k = np.argmin(test_errors)
    print(f"Seed {seed} → Best k = {best_k} | Test MSE = {test_errors[best_k]:.4f}")
```

The standard remedy is k-fold cross-validation: the data is split into K folds, and each fold serves as the test set once while the model is trained on the remaining K-1 folds. The test errors are then averaged, giving a much more stable estimate.
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_errors = []

for k in range(9):
    cols, _, _ = best_per_k[k]
    fold_errors = []
    
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        if k == 0:
            y_pred = np.full(len(y_te), y_tr.mean())
        else:
            results = sm.OLS(y_tr, sm.add_constant(X_tr[list(cols)])).fit()
            y_pred = results.predict(sm.add_constant(X_te[list(cols)]))
        
        fold_errors.append(np.mean((y_te - y_pred) ** 2))
    
    cv_errors.append(np.mean(fold_errors))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(9), cv_errors, marker='o', color='green', label='10-Fold CV MSE')
plt.xlabel('Number of predictors (k)')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validated Test Error by Model Size')
plt.xticks(range(9))
plt.legend()
plt.grid(True)
plt.show()

print(f"Best k by CV: {np.argmin(cv_errors)}")
```

## Exercise 5: Conclusion

The cross-validation results confirm what split-validation suggested: the test error decreases sharply from k=0 to k=2, then plateaus with no meaningful improvement for larger models. This consistently points to k=2 with predictors lcp and lpsa as the optimal model.
Looking back at the full-data OLS summary, this choice is further supported by the statistics:

Both lcp and lpsa have p-values of 0.000, confirming they are highly significant predictors of lcavol.
The model achieves an R² of 0.645, meaning it explains about 64.5% of the variance in lcavol with just two predictors.
The intercept is not significant (p=0.657), suggesting the response is well explained by the two predictors alone without a meaningful baseline shift.
The Jarque-Bera test (p=0.831) and low skew suggest residuals are approximately normally distributed, validating OLS assumptions.

Interpretation: lpsa (log PSA level) is the strongest predictor, which is clinically intuitive as PSA is the primary biomarker for prostate cancer volume. lcp (log capsular penetration) adds significant additional explanatory power, reflecting how cancer spread beyond the prostate correlates with its volume.
Overall: Adding predictors beyond k=2 reduces training error but not test error, meaning those extra variables capture noise rather than signal. The k=2 model offers the best bias-variance trade-off — simple, interpretable, statistically sound, and competitive in out-of-sample prediction.