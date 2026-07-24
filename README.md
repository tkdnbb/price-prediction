# price-prediction

This project uses `Python 3.13.5`.

# Dependencies

See `pyproject.toml`. Install with:

```bash
uv sync
```

# Running the server

```bash
uv run fastapi dev main.py
```

Or with uvicorn directly:

```bash
.venv/bin/uvicorn main:app --reload
```

Interactive API docs available at `http://localhost:8000/docs`.

# APIs

## `POST /predict`

Accepts one or more sets of housing features and returns price predictions.  
Pass a single-element list for a single prediction, or multiple elements for a batch prediction.

**Request body:**
```json
{
  "houses": [
    {
      "square_footage": 2000,
      "bedrooms": 3,
      "bathrooms": 2.0,
      "year_built": 2005,
      "lot_size": 5000,
      "distance_to_city_center": 5.2,
      "school_rating": 8.5
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [245000.0]
}
```

## `GET /model-info`

Returns the linear regression model's feature names, coefficients, intercept, and training-set performance metrics (R² and RMSE).

**Response:**
```json
{
  "feature_names": ["square_footage", "bedrooms", "bathrooms", "year_built", "lot_size", "distance_to_city_center", "school_rating"],
  "coefficients": [...],
  "intercept": ...,
  "r2_score": 0.95,
  "rmse": 12000.0
}
```

## `GET /health`

Simple liveness check.

**Response:**
```json
{
  "status": "ok"
}
```

# Model

A `LinearRegression` model trained on `data/House_Price_Dataset.csv` at startup.  
`bedrooms` and `bathrooms` are log-transformed (`log(x + 1)`) before training and inference.
