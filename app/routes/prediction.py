import pandas as pd

from fastapi import APIRouter, HTTPException

from app.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse
)

from src.utils import load_object


router = APIRouter(
    prefix="/v1",
    tags=["Prediction"]
)


MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"


model = load_object(MODEL_PATH)
preprocessor = load_object(PREPROCESSOR_PATH)


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    try:

        input_data = pd.DataFrame([{
            "gender": request.gender,
            "race_ethnicity": request.race_ethnicity,
            "parental_level_of_education":
                request.parental_level_of_education,
            "lunch": request.lunch,
            "test_preparation_course":
                request.test_preparation_course,
            "reading_score": request.reading_score,
            "writing_score": request.writing_score
        }])

        transformed_data = preprocessor.transform(input_data)

        prediction = model.predict(transformed_data)

        return PredictionResponse(
            math_score=round(float(prediction[0]), 2)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )