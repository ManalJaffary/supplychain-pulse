from typing import Any, Dict
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap


class DemandForecaster:

  def __init__(self):
    self.feature_names = [
        "day_of_week",
        "month",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_std_7",
    ]
    self.model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        random_state=42,
        verbosity=-1,
    )
    self.explainer = None

  def train(self, df: pd.DataFrame, target_col: str = "demand") -> None:
    X = df[self.feature_names]
    y = df[target_col]

    self.model.fit(X, y)
    self.explainer = shap.TreeExplainer(self.model)

  def predict_with_explanations(
      self, input_features: pd.DataFrame
  ) -> Dict[str, Any]:
    if self.explainer is None:
      raise ValueError("Model must be trained before inference.")

    X = input_features[self.feature_names]
    pred = self.model.predict(X)[0]
    shap_vals = self.explainer.shap_values(X)[0]

    return {
        "predicted_demand": round(float(pred), 2),
        "base_value": round(float(self.explainer.expected_value), 2),
        "shap_contributions": {
            feat: round(float(val), 3)
            for feat, val in zip(self.feature_names, shap_vals)
        },
    }