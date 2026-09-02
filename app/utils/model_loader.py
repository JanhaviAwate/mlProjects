import joblib
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "artifacts" / "model.pkl"


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)