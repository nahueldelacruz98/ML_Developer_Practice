from fastapi import APIRouter
from pydantic import BaseModel, Field

from helpers.database import insert_prediction, get_predictions as get_db_predictions
from helpers.model import predict as model_predict


router = APIRouter()


# Pydantic model for validating the input data
class Sample(BaseModel):
    gender: str = Field(..., pattern="^(Female|Male|Other)$")
    age: int = Field(..., ge=0)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    smoking_history: str = Field(
        ..., pattern="^(never|current|former|ever|not current)$"
    )
    bmi: float = Field(..., ge=0)
    HbA1c_level: float = Field(..., ge=0)
    blood_glucose_level: int = Field(..., ge=0)


@router.post("/predict")
async def predict(sample: Sample):

    # Run model and get prediction
    prediction = model_predict(**sample.model_dump())

    # Insert prediction into the database
    insert_prediction(**sample.model_dump(), prediction=prediction)

    return {"prediction": prediction}


@router.get("/predictions")
async def get_predictions():
    return get_db_predictions()
