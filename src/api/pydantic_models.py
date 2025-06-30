from pydantic import BaseModel
from typing import List, Dict, Any

class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[float]
