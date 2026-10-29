from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from src.data_pipeline import (
    create_time_series_features,
    generate_synthetic_supply_chain_data,
)
from src.forecaster import DemandForecaster
from src.nlp_extractor import (
    IncidentLog,
    ParsedIncidentReport,
    RuleBasedIncidentExtractor,
)

app = FastAPI(
    title="SupplyChain Pulse API",
    description="Intelligent Demand Forecaster, SHAP Explainer, and Incident Extractor",
    version="1.0.0",
)

# Eagerly initialize and pre-train model so both TestClient and Uvicorn work seamlessly
forecaster = DemandForecaster()
_raw_df = generate_synthetic_supply_chain_data(num_days=300)
_train_df = create_time_series_features(_raw_df)
forecaster.train(_train_df)


class ForecastPayload(BaseModel):
  day_of_week: int = Field(
      ..., ge=0, le=6, json_schema_extra={"example": 2}
  )
  month: int = Field(..., ge=1, le=12, json_schema_extra={"example": 10})
  lag_7: float = Field(..., json_schema_extra={"example": 98.5})
  lag_14: float = Field(..., json_schema_extra={"example": 94.2})
  lag_28: float = Field(..., json_schema_extra={"example": 89.0})
  rolling_mean_7: float = Field(..., json_schema_extra={"example": 96.1})
  rolling_std_7: float = Field(..., json_schema_extra={"example": 3.4})


@app.get("/health")
def health_check():
  return {"status": "healthy", "service": "SupplyChain-Pulse"}


@app.post("/predict")
def predict_demand(payload: ForecastPayload):
  try:
    input_data = pd.DataFrame([payload.model_dump()])
    result = forecaster.predict_with_explanations(input_data)
    return result
  except Exception as err:
    raise HTTPException(status_code=500, detail=str(err))


@app.post("/parse-incident", response_model=ParsedIncidentReport)
def parse_incident(log: IncidentLog):
  try:
    return RuleBasedIncidentExtractor.extract(log.raw_text)
  except Exception as err:
    raise HTTPException(status_code=500, detail=str(err))