import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Union, Optional, Dict, Any, Tuple

# -------------------
# Summary statistics
# -------------------
def stats_summary(
    data: Union[np.ndarray, pd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    output_type: str = "dict"
) -> Union[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Compute summary statistics (Min, Q1, Median, Mean, Q3, Max).

    Parameters:
        data: np.ndarray or pd.DataFrame
        x: column name for DataFrame input (ignored for numpy)
        y: column name for DataFrame input (ignored for numpy)
        output_type: 'dict' (default) or 'df'

    Returns:
        dict or pd.DataFrame with summary statistics

    Example:
    np.random.seed(42)
    true_signal = np.linspace(0, 10, 50)
    x = true_signal + np.random.normal(0, 1, 50)  # Sensor A with noise
    y = true_signal + np.random.normal(0, 1, 50)  # Sensor B with noise

    df = pd.DataFrame(data={"x": x, "y": y})
    summary = stats_summary(np.column_stack([x, y]), output_type="x")
    """
    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        if x is None or y is None:
            raise ValueError("For DataFrame input, provide column names x and y.")
        arr = data[[x, y]].to_numpy(dtype=float)
        cols = [x, y]
    else:
        arr = np.asarray(data, dtype=float)
        # If 1D array, reshape
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [f"Var{i}" for i in range(arr.shape[1])]

    # Compute statistics ignoring NaNs
    min_ = np.nanmin(arr, axis=0)
    q1 = np.nanpercentile(arr, 25, axis=0)
    median_ = np.nanmedian(arr, axis=0)
    mean_ = np.nanmean(arr, axis=0)
    q3 = np.nanpercentile(arr, 75, axis=0)
    max_ = np.nanmax(arr, axis=0)

    stats_dict = {
        "Min": min_,
        "Q1": q1,
        "Median": median_,
        "Mean": mean_,
        "Q3": q3,
        "Max": max_
    }

    if output_type == "df":
        df = pd.DataFrame(stats_dict, index=cols).T
        return df
    else:
        return stats_dict


# -------------------
# Regression metrics
# -------------------
def stats_measures(
    x: Union[np.ndarray, pd.Series, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    df: Optional[pd.DataFrame] = None,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    output_type: str = "dict"
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Compute regression metrics: R2, RMSE, MAE, MBE, Pearson r, explained variance, etc.

    Parameters:
        x, y: numpy arrays or pd.Series if using numpy input
        df: pd.DataFrame if using column input
        x_col, y_col: column names for df
        output_type: 'dict' (default) or 'df'

    Returns:
        dict or pd.DataFrame of regression metrics

    Example:
    np.random.seed(42)
    true_signal = np.linspace(0, 10, 50)
    x = true_signal + np.random.normal(0, 1, 50)  # Sensor A with noise
    y = true_signal + np.random.normal(0, 1, 50)  # Sensor B with noise

    df = pd.DataFrame(data={"x": x, "y": y})
    metrics = stats_measures(x, y, output_type="x")
    """
    # Determine input type
    if df is not None:
        if x_col is None or y_col is None:
            raise ValueError("Provide x_col and y_col for DataFrame input.")
        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    # Linear regression
    slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)

    # Helper functions
    def r2_func(a, b):
        mean_a = np.mean(a)
        return 1 - np.sum((b - a) ** 2) / np.sum((a - mean_a) ** 2)

    def rmse_func(a, b):
        return np.sqrt(np.mean((b - a) ** 2))

    def mae_func(a, b):
        return np.mean(np.abs(b - a))

    def median_absolute_error(a, b):
        return np.median(np.abs(b - a))

    def max_error(a, b):
        return np.max(np.abs(b - a))

    def explained_variance(a, b):
        return 1 - np.var(a - b) / np.var(a) if np.var(a) != 0 else np.nan

    def pearsonr_func(a, b) -> Tuple[float, float]:
        r = np.corrcoef(a, b)[0, 1]
        n = a.size
        if n < 3:
            return np.nan, np.nan
        t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
        df_ = n - 2
        if df_ > 30:
            from math import erf, sqrt
            p = 2 * (1 - 0.5 * (1 + erf(abs(t_stat)/sqrt(2))))
        else:
            p = np.nan
        return r, p

    Pearsonr, Pearsonp = pearsonr_func(x, y)
    metrics_dict = {
        "r2": r_value ** 2,
        "R2": r2_func(x, y),
        "slope": slope,
        "intercept": intercept,
        "pvalue": p_value,
        "RMSE": rmse_func(x, y),
        "MBE": np.mean(y - x),
        "MAE": mae_func(x, y),
        "stderr": stderr,
        "Pearsonr": Pearsonr,
        'Pearsonp': Pearsonp,
        "explained_variance": explained_variance(x, y),
        "MAXERR": max_error(x, y),
        "MEAE": median_absolute_error(x, y),
        "mean": np.mean(x)
    }

    if output_type == "df":
        return pd.DataFrame(metrics_dict, index=[0])
    else:
        return metrics_dict

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