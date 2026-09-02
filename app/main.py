from fastapi import FastAPI

from app.routes.prediction import router as prediction_router


app = FastAPI(
    title="Math Score Prediction API",
    description="API for predicting student math scores",
    version="1.0.0"
)


app.include_router(prediction_router)


@app.get("/")
def root():

    return {
        "message": "Math Score Prediction API is running"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }