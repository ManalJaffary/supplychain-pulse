import re
from typing import List
from pydantic import BaseModel, Field


class IncidentLog(BaseModel):
  raw_text: str = Field(
      ...,
      json_schema_extra={
          "example": "Port strikes at Karachi delayed shipments by 12 days."
      },
  )


class ParsedIncidentReport(BaseModel):
  supplier: str = Field(..., description="Identified vendor or location")
  severity: str = Field(..., description="Risk tier: Low, Medium, High")
  delay_days: int = Field(..., description="Estimated delivery delay in days")
  affected_materials: List[str] = Field(default_factory=list)
  is_force_majeure: bool = Field(...)


class RuleBasedIncidentExtractor:
  """Production parsing service with deterministic validation."""

  @staticmethod
  def extract(text: str) -> ParsedIncidentReport:
    text_lower = text.lower()

    # Risk level classification
    if any(
        k in text_lower for k in ["strike", "fire", "closure", "embargo", "war"]
    ):
      severity = "High"
    elif any(k in text_lower for k in ["delay", "congestion", "storm"]):
      severity = "Medium"
    else:
      severity = "Low"

    # Delay duration extraction via regex
    delay_match = re.search(r"(\d+)\s*(?:day|days)", text_lower)
    delay_days = int(delay_match.group(1)) if delay_match else (
        7 if severity == "High" else 2
    )

    # Material classification
    catalog = ["steel", "semiconductors", "aluminum", "copper", "electronics"]
    materials = [mat.title() for mat in catalog if mat in text_lower]
    if not materials:
      materials = ["General Cargo"]

    # Force majeure legal clause detection
    is_force_majeure = any(
        k in text_lower
        for k in ["flood", "hurricane", "earthquake", "act of god", "war"]
    )

    return ParsedIncidentReport(
        supplier="Global Logistics Hub",
        severity=severity,
        delay_days=delay_days,
        affected_materials=materials,
        is_force_majeure=is_force_majeure,
    )