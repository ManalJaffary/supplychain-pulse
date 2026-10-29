import numpy as np
import pandas as pd


def generate_synthetic_supply_chain_data(
    num_days: int = 365,
) -> pd.DataFrame:
  """Generates realistic daily procurement demand data for testing and local training."""
  np.random.seed(42)
  dates = pd.date_range(start="2025-01-01", periods=num_days, freq="D")

  trend = np.linspace(50, 120, num_days)
  seasonality = 20 * np.sin(2 * np.pi * dates.day_of_year / 365.25)
  weekly = 10 * (dates.day_of_week >= 5)
  noise = np.random.normal(0, 5, num_days)

  demand = np.maximum(10, trend + seasonality + weekly + noise)
  return pd.DataFrame({"date": dates, "demand": demand})


def create_time_series_features(
    df: pd.DataFrame, target_col: str = "demand"
) -> pd.DataFrame:
  """Generates lag features, rolling statistics, and calendar attributes without target leakage."""
  data = df.copy()
  data["date"] = pd.to_datetime(data["date"])
  data = data.sort_values("date").reset_index(drop=True)

  # Calendar attributes
  data["day_of_week"] = data["date"].dt.day_of_week
  data["month"] = data["date"].dt.month

  # Lag features (past values only)
  data["lag_7"] = data[target_col].shift(7)
  data["lag_14"] = data[target_col].shift(14)
  data["lag_28"] = data[target_col].shift(28)

  # Rolling statistics shifted by 1 step to ensure no target leakage
  data["rolling_mean_7"] = data[target_col].shift(1).rolling(window=7).mean()
  data["rolling_std_7"] = data[target_col].shift(1).rolling(window=7).std()

  # Drop incomplete rows resulting from shifts
  data = data.dropna().reset_index(drop=True)
  return data