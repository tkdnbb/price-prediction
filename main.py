from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class HouseFeatures(BaseModel):
    square_footage: int
    bedrooms: int
    bathrooms: float
    year_built: int
    lot_size: int
    distance_to_city_center: float
    school_rating: float


class PredictRequest(BaseModel):
    houses: list[HouseFeatures]


class PredictResponse(BaseModel):
    predictions: list[float]


class ModelInfoResponse(BaseModel):
    feature_names: list[str]
    coefficients: list[float]
    intercept: float
    r2_score: float
    rmse: float


# ---------------------------------------------------------------------------
# Global model state (populated during startup)
# ---------------------------------------------------------------------------

_model: LinearRegression | None = None
_feature_names: list[str] = []
_train_r2: float = 0.0
_train_rmse: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _feature_names, _train_r2, _train_rmse

    data = pd.read_csv("data/House_Price_Dataset.csv")

    X = data.drop(columns=["id", "price"]).copy()
    y = data["price"]

    # Same preprocessing as notebook
    X["bedrooms"] = np.log(X["bedrooms"] + 1)
    X["bathrooms"] = np.log(X["bathrooms"] + 1)

    _feature_names = list(X.columns)

    _model = LinearRegression()
    _model.fit(X, y)

    y_pred = _model.predict(X)
    _train_r2 = float(r2_score(y, y_pred))
    _train_rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="House Price Prediction", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess(houses: list[HouseFeatures]) -> np.ndarray:
    """Apply the same log transforms used during training and return a numpy array."""
    rows = [
        {
            "square_footage": h.square_footage,
            "bedrooms": np.log(h.bedrooms + 1),
            "bathrooms": np.log(h.bathrooms + 1),
            "year_built": h.year_built,
            "lot_size": h.lot_size,
            "distance_to_city_center": h.distance_to_city_center,
            "school_rating": h.school_rating,
        }
        for h in houses
    ]
    return pd.DataFrame(rows)[_feature_names].to_numpy()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Accept one or more sets of housing features and return price predictions.

    Pass a single-element list for a single prediction, or multiple elements
    for a batch prediction.
    """
    X = _preprocess(request.houses)
    predictions: list[float] = _model.predict(X).tolist()
    return PredictResponse(predictions=predictions)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return the model's coefficients, intercept, and training-set metrics."""
    return ModelInfoResponse(
        feature_names=_feature_names,
        coefficients=_model.coef_.tolist(),
        intercept=float(_model.intercept_),
        r2_score=_train_r2,
        rmse=_train_rmse,
    )


@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}
