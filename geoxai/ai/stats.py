import numpy as np
import statsmodels.api as sm
from typing import Dict, Any, Optional

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