from src.data_pipeline import (
    create_time_series_features,
    generate_synthetic_supply_chain_data,
)


def test_data_generation_and_features():
  raw_data = generate_synthetic_supply_chain_data(num_days=100)
  assert not raw_data.empty
  assert "demand" in raw_data.columns

  featured_data = create_time_series_features(raw_data)
  expected_columns = [
      "day_of_week",
      "month",
      "lag_7",
      "lag_14",
      "lag_28",
      "rolling_mean_7",
      "rolling_std_7",
  ]
  for col in expected_columns:
    assert col in featured_data.columns

  # Verify no NaN values exist in calculated features
  assert featured_data.isnull().sum().sum() == 0