from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):

    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str

    reading_score: float = Field(..., ge=0, le=100)
    writing_score: float = Field(..., ge=0, le=100)


class PredictionResponse(BaseModel):

    math_score: float