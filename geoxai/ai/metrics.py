import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance
import statsmodels.api as sm
from typing import Dict, Any, Optional

def stats_summary(df):
    min_ = df.min().to_frame().T
    Q1 = df.quantile(0.25).to_frame().T
    median_ = df.quantile(0.5).to_frame().T
    mean_ = df.mean().to_frame().T
    Q3 = df.quantile(0.75).to_frame().T
    max_ = df.max().to_frame().T
    df_stats = pd.concat([min_, Q1, median_, mean_, Q3, max_])
    df_stats.index = ["Min", "Q1", "Median", "Mean", "Q3", "Max"]
    return df_stats

def stats_measures(x, y):
    # from sklearn.metrics import mean_absolute_percentage_error
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
    mse = mean_squared_error(x, y)
    r2 = rvalue ** 2
    rmse = np.sqrt(mse)
    mbe = (y - x).mean()
    # ----------------------------------------------------------------
    pearsonr = stats.pearsonr(x, y)
    evs = explained_variance_score(x, y)
    me = max_error(x, y)
    mae = mean_absolute_error(x, y)
    msle = mean_squared_log_error(x, y)
    meae = median_absolute_error(x, y)
    r2_score = r2_score(x, y)
    mpd = mean_poisson_deviance(x, y)
    mgd = mean_gamma_deviance(x, y)
    mtd = mean_tweedie_deviance(x, y)
    mean_ = np.mean(x)
    return {
        "r2": r2,
        "SLOPE": slope,
        "RMSE": rmse,
        "MBE": mbe,
        "INTERCEPT": intercept,
        "PVALUE": pvalue,
        "STDERR": stderr,
        "PEARSON": pearsonr,
        "EXPLAINED_VARIANCE": evs,
        "MAXERR": me,
        "MAE": mae,
        "MSLE": msle,
        "MEDIAN_AE": meae,
        "R2": r2_score,
        "MPD": mpd,
        "MGD": mgd,
        "MTD": mtd,
        "MEAN": mean_
    }

def stats_measures_df(df, name1, name2, return_dict = False):
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(df[name1], df[name2])
    mse = mean_squared_error(df[name1], df[name2])
    r2 = rvalue ** 2
    rmse = np.sqrt(mse)
    mbe = (df[name2] - df[name1]).mean()
    if return_dict:
        return {
            "R2": r2,
            "SLOPE": slope,
            "RMSE": rmse,
            "MBE": mbe
        }
    else:
        return [r2, slope, rmse, mbe]

def get_r2(x, y):
    try:
        x_bar = x.mean()
    except:
        x_bar = np.mean(x)

    r2 = 1 - np.sum((x - y)**2) / np.sum((x - x_bar)**2)
    return r2

def get_rmse(observations, estimates):
    return np.sqrt(((estimates - observations) ** 2).mean())

def calculate_R2(y_true, y_pred):
    """
    Calculate the R^2 (coefficient of determination).

    Args:
        y_true (array-like): Actual values of the dependent variable.
        y_pred (array-like): Predicted values of the dependent variable.

    Returns:
        float: The R^2 value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # R^2 calculation
    R2 = 1 - (ss_res / ss_tot)
    return R2

def regress2(
    x: np.ndarray, 
    y: np.ndarray, 
    method_type_1: str = "ols",
    method_type_2: str = "rma",
    weight_x: Optional[np.ndarray] = None, 
    weight_y: Optional[np.ndarray] = None, 
    intercept: bool = True
) -> Dict[str, Any]:
    """
    Model type I and II regression, including RMA (reduced major axis).
    Original from: UMaine MISC Lab; emmanuel.boss@maine.edu
    Example:
    # 1. Generate synthetic data
    np.random.seed(42)
    true_signal = np.linspace(0, 10, 50)
    x = true_signal + np.random.normal(0, 1, 50)  # Sensor A with noise
    y = true_signal + np.random.normal(0, 1, 50)  # Sensor B with noise

    # 2. Run the refactored Type II (RMA) regression
    results = regress2(x, y, method_type_2="rma")

    # 3. View the results
    print(f"Slope (RMA):     {results['slope']:.4f}")
    print(f"Intercept:       {results['intercept']:.4f}")
    print(f"Correlation (r): {results['r']:.4f}")

    """
    x, y = np.asarray(x), np.asarray(y)
    
    # Normalize method names for easier matching
    m1 = method_type_1.lower().strip()
    m2 = method_type_2.lower().strip()

    # 1. Model Type I Strategy Mapping
    # Maps method names to (Statsmodels Class, weight_key)
    m1_strategies = {
        "ols": (sm.OLS, None),
        "ordinary least square": (sm.OLS, None),
        "wls": (sm.WLS, "weights"),
        "weighted least square": (sm.WLS, "weights"),
        "rlm": (sm.RLM, None),
        "robust linear model": (sm.RLM, None)
    }

    if m1 not in m1_strategies:
        raise ValueError(f"Invalid Type I method: {method_type_1}")

    model_cls, weight_attr = m1_strategies[m1]

    def fit_model(indep, dep, weights=None):
        instr_x = sm.add_constant(indep) if intercept else indep
        kwargs = {weight_attr: 1.0 / weights} if weight_attr and weights is not None else {}
        return model_cls(dep, instr_x, **kwargs).fit().params

    # 2. Computation Logic
    if m2 in ["reduced major axis", "rma", "geometric mean"]:
        # Type I regressions for both (y~x) and (x~y)
        params_a = fit_model(x, y, weight_y)
        params_b = fit_model(y, x, weight_x)
        
        slope_a = params_a[-1]
        slope_b = params_b[-1]
        
        if np.sign(slope_a) != np.sign(slope_b):
            raise RuntimeError("Type I regressions have opposite signs; RMA undefined.")

        # RMA Math
        slope = np.sign(slope_a) * np.sqrt(slope_a * (1.0 / slope_b))
        
        if intercept:
            # Use mean for OLS, median for robust methods
            center_y, center_x = (np.mean(y), np.mean(x)) if m1 == "ols" else (np.median(y), np.median(x))
            intercept_val = center_y - slope * center_x
        else:
            intercept_val = 0.0

        r = np.sign(slope_a) * np.sqrt(slope_a / (1.0 / slope_b))
        predict = slope * x + intercept_val
        
        # Statistics
        n = len(x)
        residuals = y - predict
        ss_x = np.sum(x**2)
        denom = n * ss_x - np.sum(x)**2
        s2 = np.sum(residuals**2) / (n - 2)
        
        std_slope = np.sqrt(n * s2 / denom)
        std_int = np.sqrt(ss_x * s2 / denom) if intercept else 0.0

    elif m2 in ["major axis", "ma", "pearson's major axis"]:
        if not intercept:
            raise ValueError("Major Axis requires an intercept.")
            
        xm, ym = np.mean(x), np.mean(y)
        xp, yp = x - xm, y - ym
        
        s_xx, s_yy = np.sum(xp**2), np.sum(yp**2)
        s_xy = np.sum(xp * yp)
        
        # Eigenvector-based slope for Major Axis
        slope = (s_yy - s_xx + np.sqrt((s_yy - s_xx)**2 + 4 * s_xy**2)) / (2 * s_xy)
        intercept_val = ym - slope * xm
        
        r = s_xy / np.sqrt(s_xx * s_yy)
        predict = slope * x + intercept_val
        
        n = len(x)
        std_slope = (slope / r) * np.sqrt((1 - r**2) / n)
        # Simplified standard error for intercept
        std_int = np.sqrt(((np.std(y) - np.std(x) * slope)**2) / n) # Approximation

    else:
        raise ValueError(f"Method {m2} not supported or requires implementation.")

    return {
        "slope": float(slope),
        "intercept": float(intercept_val),
        "r": float(r),
        "std_slope": float(std_slope),
        "std_intercept": float(std_int),
        "predict": predict
    }


def concordance_correlation_coefficient(y_true, y_pred):
    """
    Compute Lin's Concordance Correlation Coefficient (CCC).

    CCC measures agreement between two variables by combining
    precision (correlation) and accuracy (closeness to 1:1 line).

    https://rowannicholls.github.io/python/statistics/agreement/correlation_coefficients.html#lins-concordance-correlation-coefficient-ccc
    Lin LIK (1989). “A concordance correlation coefficient to evaluate reproducibility”. Biometrics. 45 (1):255-268.
    
    Example:
    y_true = [3, -0.5, 2, 7, np.nan]
    y_pred = [2.5, 0.0, 2, 8, 3]
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    print(ccc)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        return np.nan

    mean_true = y_true.mean()
    mean_pred = y_pred.mean()

    diff_true = y_true - mean_true
    diff_pred = y_pred - mean_pred

    var_true = np.mean(diff_true ** 2)
    var_pred = np.mean(diff_pred ** 2)
    cov = np.mean(diff_true * diff_pred)

    return (2 * cov) / (
        var_true + var_pred + (mean_true - mean_pred) ** 2
    )