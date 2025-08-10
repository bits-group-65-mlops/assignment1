from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import numpy as np

class PredictionFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")
    
    @field_validator('*')
    def check_positive(cls, v, info):
        if v <= 0:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be positive")
        return v

class PredictionRequest(BaseModel):
    data: List[PredictionFeatures]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "data": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2
                    }
                ]
            }
        }
    }
