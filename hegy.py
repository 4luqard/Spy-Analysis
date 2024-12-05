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
    
    # Number of observations after differencing
    n_diff = len(delta_y)

    # Adjust maxlag if it's too large for the data
    maxlag = min(int(maxlag), n_diff - s)
    
    # Construct lagged levels of y_t for s periods
    y_lags = np.column_stack([y[(s - i - 1):-(i + 1) if (i + 1) != 0 else None] for i in range(s)])
    
    # Number of observations after lagging
    n_obs = y_lags.shape[0]
    
    # Compute seasonal frequencies
    frequencies = [2 * np.pi * j / s for j in range(1, s // 2 + 1)]
    
    # Prepare the π variables
    pi_vars = []
    
    # Mean term (zero frequency)
    pi_y_plus = y_lags.sum(axis=1) / s
    pi_vars.append(pi_y_plus)
    
    # For each frequency
    for freq in frequencies:
        cos_terms = np.cos(freq * np.arange(1, s + 1))
        sin_terms = np.sin(freq * np.arange(1, s + 1))

        pi_cos = np.dot(y_lags, cos_terms) / s
        pi_vars.append(pi_cos)

        # Include sine terms only if frequency is not π or s is odd
        if freq != np.pi or s % 2 != 0:
            pi_sin = np.dot(y_lags, sin_terms) / s
            pi_vars.append(pi_sin)
    
    # If s is even, include the Nyquist frequency term (frequency = π)
    # if s % 2 == 0:
    #     nyquist_freq = np.pi
    #     cos_terms = np.cos(nyquist_freq * np.arange(1, s + 1))
    #     pi_nyquist = np.dot(y_lags, cos_terms)
    #     pi_vars.append(pi_nyquist)
    
    # Align all arrays to have the same length
    min_len = min(len(delta_y), len(pi_vars[0]))
    delta_y = delta_y[-min_len:]
    for i in range(len(pi_vars)):
        pi_vars[i] = pi_vars[i][-min_len:]
        
    # Adjust for maxlag
    delta_y_reg = delta_y[maxlag:]
    for i in range(len(pi_vars)):
        pi_vars[i] = pi_vars[i][maxlag:]

    # Prepare lagged differences
    delta_y_lags = []
    for lag in range(1, maxlag + 1):
        delta_y_lag = delta_y[maxlag - lag:-lag if lag != 0 else None]
        delta_y_lags.append(delta_y_lag)
    
    # Truncate arrays to the same length
    min_len = len(delta_y_reg)
    delta_y_reg = delta_y_reg[:min_len]
    for i in range(len(pi_vars)):
        pi_vars[i] = pi_vars[i][:min_len]
    for i in range(len(delta_y_lags)):
        delta_y_lags[i] = delta_y_lags[i][:min_len]
    
    # Align delta_y with pi_vars
    # delta_y = delta_y[(s - 1):]
    # delta_y_reg = delta_y[int(maxlag):n_obs]
    
    # Create DataFrame for exogenous variables
    exog = {}
    if constant:
        exog['const'] = np.ones(min_len)
    if trend:
        exog['trend'] = np.arange(min_len)

    # Add π variables to exog
    for idx, pi_var in enumerate(pi_vars):
        exog[f'pi_var_{idx+1}'] = pi_var

    # Add lagged differences up to maxlag
    for idx, delta_y_lag in enumerate(delta_y_lags):
        exog[f'delta_y_lag_{idx+1}'] = delta_y_lag
        
    
    # Adjust delta_y to match the exogenous variables
    # delta_y_reg = delta_y[int(maxlag):]
    
    lengths = []
    for key in exog.keys():
        lengths.append(len(exog[key]))
    if any(lengths[i] != lengths[i+1] for i in range(len(lengths) - 1)):
        raise ValueError("All arrays must be of the same length")
    
    # Create DataFrame
    exog_df = pd.DataFrame(exog)
    
    # Ensure that delta_y_reg and exog_df have the same number of observations
    # min_length = min(len(delta_y_reg), len(exog_df))
    # delta_y_reg = delta_y_reg[:min_length]
    # exog_df = exog_df.iloc[:min_length]
    
    # Perform the OLS regression
    model = OLS(delta_y_reg, exog_df)
    results = model.fit()
    
    print(results.summary())
    print(results.f_test(np.identity(len(results.params))))
    
    # Extract t-statistics and p-values for π variables
    pi_var_names = [f'pi_var_{idx+1}' for idx in range(len(pi_vars))]
    t_stats = results.tvalues[pi_var_names] # TODO: Currently this code is not doing anything fix this later on.
    pvalues_s = results.pvalues[['const', 'trend']]
    pvalues_t = results.pvalues[pi_var_names]
    
    # F-statistic for the joint hypothesis that all π variables are zero.
    pvalue_f_stats = results.f_test(np.identity(len(results.params))).pvalue
    
    pvalues = {
        "p_values_s": list(pvalues_s),
        "p_values_t": list(pvalues_t),
        "p_value_f": pvalue_f_stats
    }
    
    stat = None  # TODO: Placeholder, can be extended to include actual test statistics
    
    return stat, pvalues