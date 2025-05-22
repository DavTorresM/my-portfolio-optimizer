import string
import numpy as np
import pandas as pd
#from prophet import Prophet
from statsforecast.adapters.prophet import AutoARIMAProphet
from typing import List, Tuple, Dict, Optional

# Constants for risk metric weighting, these are hyperparameters that could be optimized.
ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0


def generate_column_labels(n: int) -> List[str]:
    """
    Generate a list of column labels using letters (A, B, ..., Z, AA, BB, ...).

    Parameters:
    ----------
    n : int
        Number of labels to generate.

    Returns:
    -------
    List[str]
        List of column labels.
    """
    labels = []
    for i in range(n):
        if i < 26:
            labels.append(string.ascii_uppercase[i])
        else:
            repetitions = (i // 26) + 1
            base_letter = string.ascii_uppercase[i % 26]
            labels.append(base_letter * repetitions)
    return labels


def get_traditional_risk(cov_matrix: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate portfolio risk as the standard deviation of returns.

    Parameters:
    ----------
    cov_matrix : np.ndarray
        Covariance matrix of asset returns (NxN).
    weights : np.ndarray
        Portfolio weights (length N).

    Returns:
    -------
    float
        Portfolio volatility (standard deviation of returns).

    Raises:
    ------
    ValueError
        If dimensions of weights and covariance matrix do not match.
    """
    weights = np.array(weights)
    cov_matrix = np.array(cov_matrix)

    if weights.shape[0] != cov_matrix.shape[0]:
        raise ValueError("Weights dimension must match covariance matrix dimension.")

    return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))


def get_weighted_average_stddev_hat(stddev_hat: List[float], weights: List[float]) -> Optional[float]:
    """
    Calculate weighted average of estimated standard deviations.

    Parameters:
    ----------
    stddev_hat : List[float]
        List of estimated standard deviations.
    weights : List[float]
        Asset weights.

    Returns:
    -------
    Optional[float]
        Weighted average standard deviation, or None if error occurs.
    """
    try:
        stddev_hat = np.array(stddev_hat)
        weights = np.array(weights)

        if len(stddev_hat) != len(weights):
            raise ValueError("Length of stddev_hat and weights must be equal.")

        normalized_weights = weights / np.sum(weights)
        return float(np.dot(normalized_weights, stddev_hat))
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def min_max_scaling(series: pd.Series) -> pd.Series:
    """
    Apply min-max scaling to a pandas Series to scale values to [0, 1].

    Parameters:
    ----------
    series : pd.Series
        Series to scale.

    Returns:
    -------
    pd.Series
        Scaled series.

    Raises:
    ------
    ValueError
        If the series is empty or contains constant values.
    """
    if series.empty:
        raise ValueError("Input series is empty.")
    if series.min() == series.max():
        raise ValueError("Input series contains constant values.")
    
    return (series - series.min()) / (series.max() - series.min())


def simulate_volatility_scenarios(cov_matrix: np.ndarray, 
                                   num_simulations: int = 1_000_000, 
                                   seed: int = 1994) -> Tuple[float, float]:
    """
    Simulate portfolio volatility across many random weight combinations.

    Parameters:
    ----------
    cov_matrix : np.ndarray
        Covariance matrix (NxN).
    num_simulations : int
        Number of portfolio simulations.
    seed : int
        Random seed for reproducibility.

    Returns:
    -------
    Tuple[float, float]
        Minimum and maximum volatility observed.
    """
    np.random.seed(seed)
    n_assets = cov_matrix.shape[0]
    volatilities = []

    for _ in range(num_simulations):
        weights = np.random.rand(n_assets)
        weights /= np.sum(weights)
        vol = get_traditional_risk(cov_matrix, weights)
        volatilities.append(vol)

    return float(np.min(volatilities)), float(np.max(volatilities))


def my_balanced_risk_metric(cov_matrix: np.ndarray, 
                            vol_min: float, 
                            vol_max: float,
                            vect_cvar: pd.Series,
                            stddev_hat: pd.Series,
                            weights: np.ndarray) -> float:
    """
    Composite risk metric combining CVaR, estimated stddev, and portfolio volatility.

    Parameters:
    ----------
    cov_matrix : np.ndarray
        Covariance matrix (NxN).
    vol_min : float
        Minimum simulated volatility.
    vol_max : float
        Maximum simulated volatility.
    vect_cvar : pd.Series
        Series of conditional VaR estimates per asset.
    stddev_hat : pd.Series
        Series of estimated standard deviations per asset.
    weights : np.ndarray
        Portfolio weights.

    Returns:
    -------
    float
        Balanced risk score in range [0, 1].
    """
    scaled_stddev_hat = min_max_scaling(stddev_hat)
    scaled_cvar = min_max_scaling(vect_cvar)
    vol_actual = get_traditional_risk(cov_matrix, weights)
    scaled_vol = (vol_actual - vol_min) / (vol_max - vol_min)

    balanced_risk = (ALPHA * np.dot(scaled_cvar, weights) +
                     BETA * np.dot(scaled_stddev_hat, weights) +
                     GAMMA * scaled_vol)

    return float(balanced_risk / (ALPHA + BETA + GAMMA))


def get_stats_returns_hat(df: pd.DataFrame, periods_future: int) -> pd.DataFrame:
    """
    Use AutoARIMA to forecast future returns and return descriptive stats for each asset.

    Parameters:
    ----------
    df : pd.DataFrame
        Historical return time series, one column per asset.
    periods_future : int
        Number of future periods to forecast.

    Returns:
    -------
    pd.DataFrame
        DataFrame with statistical summaries of predicted returns.
    """
    stats_dict = {}

    for ticker in df.columns:
        df_ticker = df[ticker].reset_index().rename(columns={"index": "ds", ticker: "y"})
        
        #model = Prophet(backend='pystan')
        model = AutoARIMAProphet()
        model.fit(df_ticker)
        future = model.make_future_dataframe(periods=periods_future)
        forecast = model.predict(future)

        stats_dict[ticker] = (forecast.tail(periods_future)["yhat"]*100).describe()

    return pd.DataFrame(stats_dict)


def get_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute risk metrics (VaR, CVaR) for each asset.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame of historical returns.

    Returns:
    -------
    pd.DataFrame
        Risk metrics including VaR, CVaR, and relative difference.
    """
    risk_dict = {}

    for ticker in df.columns:
        series = df[ticker]
        var = np.percentile(series, 5)
        losses_beyond_var = series[series < var]
        cvar = losses_beyond_var.mean()

        risk_dict[ticker] = {
            "VaR": var,
            "CVaR": cvar,
            "|CVaR - VaR|": abs(cvar - var),
            "|CVaR - VaR| / VaR": (cvar - var) / var
        }

    return pd.DataFrame(risk_dict)