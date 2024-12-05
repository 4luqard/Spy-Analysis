# hegy.py

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

def hegy_test(data, seasonal_period, trend, constant, maxlag):
    """
    Perform the HEGY seasonal unit root test for any seasonal period.
    
    Parameters:
    - data: list or array-like, the time series data.
    - seasonal_period: int, the seasonal period (e.g., 4 for quarterly data).
    - trend: bool, include a trend term in the regression.
    - constant: bool, include a constant term in the regression.
- maxlag: int, the number of lagged difference terms to include.
    
    Returns:
    - stat: None (placeholder, can be extended to include test statistics).
    - pvalue: dict, contains p-values for t-statistics and F-statistics.
    """
    y = np.array(data)
    s = seasonal_period
    
    if s <= 1:
        raise ValueError("seasonal_period must be an integer greater than 1.")
    
    n = len(y)
    
    # Ensure enough data points
    min_required = s + maxlag + 1
    if n < min_required:
        raise ValueError(f"Not enough data points. At least {min_required} observations are required.")

    # Compute the differences Δy_t
    delta_y = np.diff(y)
    
    # Construct lagged levels of y_t
    y_lags = np.column_stack([y[(s - i - 1):-i - 1] for i in range(s)])
    
    # Number of observations after lagging
    n_obs = y_lags.shape[0]
    
    # Compute seasonal frequencies
    frequencies = [2 * np.pi * j / s for j in range(1, s // 2 + 1)]
    
    # Prepare the π variables
    pi_vars = []
    
    # Mean term (zero frequency)
    pi_y_plus = y_lags.mean(axis=1)
    pi_vars.append(pi_y_plus)
    
    # For each frequency
    for freq in frequencies:
        cos_terms = np.cos(freq * np.arange(1, s + 1))
        sin_terms = np.sin(freq * np.arange(1, s + 1))
        
        pi_cos = np.dot(y_lags, cos_terms)
        pi_vars.append(pi_cos)
        
        # For frequencies not equal to π (s even)
        if freq != np.pi:
            pi_sin = np.dot(y_lags, sin_terms)
            pi_vars.append(pi_sin)
    
    # If s is even, include the Nyquist frequency term (frequency = π)
    if s % 2 == 0:
        nyquist_freq = np.pi
        cos_terms = np.cos(nyquist_freq * np.arange(1, s + 1))
        pi_nyquist = np.dot(y_lags, cos_terms)
        pi_vars.append(pi_nyquist)
    
    # Align delta_y with pi_vars
    delta_y = delta_y[(s - 1):]
    
    # Create DataFrame for exogenous variables
    exog = {}
    if constant:
        exog['const'] = np.ones(len(delta_y))
    if trend:
        exog['trend'] = np.arange(len(delta_y))
    
    # Add π variables to exog
    for idx, pi_var in enumerate(pi_vars):
        exog[f'pi_var_{idx+1}'] = pi_var[(maxlag):n_obs]
    
    # Add lagged differences up to maxlag
    for lag in range(1, int(maxlag) + 1):
        exog[f'delta_y_lag_{lag}'] = delta_y[(lag - 1):n_obs-lag]
        
    
    # Adjust delta_y to match the exogenous variables
    delta_y_reg = delta_y[int(maxlag):]
    
    print(exog)
    
    lengths = []
    for key in exog.keys():
        lengths.append(len(exog[key]))
    if any(lengths[i] != lengths[i+1] for i in range(len(lengths) - 1)):
        raise ValueError("All arrays must be of the same length")
    
    # Create DataFrame
    exog_df = pd.DataFrame(exog)
    
    # Perform the OLS regression
    model = OLS(delta_y_reg, exog_df)
    results = model.fit()
    
    # Extract t-statistics and p-values for π variables
    # t_stats = results.tvalues[[col for col in exog_df.columns if 'pi_var' in col]] TODO: Currently this code is not doing anything fix this later on.
    pvalues = results.pvalues[[col for col in exog_df.columns if 'pi_var' in col]]
    
    # Placeholder for F-statistics (not calculated in this simplified version)
    pvalue_f_stats = [0.1 for _ in range(len(pi_vars))]
    
    pvalue = {
        "t_statistic": list(pvalues),
        "f_statistic": pvalue_f_stats
    }
    
    stat = None  # Placeholder, can be extended to include actual test statistics
    
    return stat, pvalue