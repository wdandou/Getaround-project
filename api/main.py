"""
Getaround - Pricing Prediction API
===================================

API REST exposant le modèle de pricing optimisé (voir
`ml_pricing/train_model.py`) sous la forme d'un endpoint /predict.

Endpoints :
    GET  /            -> informations générales sur l'API
    POST /predict     -> prédiction du prix de location journalier
    GET  /docs         -> documentation interactive (Swagger UI, générée
                          automatiquement par FastAPI)

Lancement en local :
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Exemple d'appel :
    curl -i -H "Content-Type: application/json" -X POST \\
      -d '{"input": [[140000, 100, true, true, false, false, true, true, true]]}' \\
      http://localhost:8000/predict
"""

from pathlib import Path
from typing import List, Union

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "pricing_model.joblib"

# Ordre des colonnes attendu dans chaque ligne de "input"
# (doit correspondre à l'ordre utilisé lors de l'entraînement, voir
# ml_pricing/train_model.py : ALL_FEATURES)
FEATURE_ORDER = [
    "mileage",
    "engine_power",
    "private_parking_available",
    "has_gps",
    "has_air_conditioning",
    "automatic_car",
    "has_getaround_connect",
    "has_speed_regulator",
    "winter_tires",
    "model_key",
    "fuel",
    "paint_color",
    "car_type",
]

app = FastAPI(
    title="Getaround Pricing Prediction API",
    description=(
        "API de prédiction du prix de location journalier optimal d'un "
        "véhicule Getaround, à partir de ses caractéristiques."
    ),
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)


class PredictionInput(BaseModel):
    input: List[List[Union[float, int, bool, str]]] = Field(
        ...,
        description=(
            "Liste de véhicules. Chaque véhicule est une liste de "
            f"{len(FEATURE_ORDER)} valeurs, dans l'ordre suivant : "
            f"{FEATURE_ORDER}"
        ),
        examples=[
            [
                [
                    140411,
                    100,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    True,
                    "Citroën",
                    "diesel",
                    "black",
                    "convertible",
                ]
            ]
        ],
    )


class PredictionOutput(BaseModel):
    prediction: List[float]


@app.get("/", tags=["Général"])
def root():
    """Message de bienvenue et lien vers la documentation."""
    return {
        "message": "Getaround Pricing Prediction API",
        "docs": "/docs",
        "predict_endpoint": "/predict (POST)",
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prédiction"])
def predict(payload: PredictionInput):
    """
    Prédit le prix de location journalier (en €) pour une ou plusieurs
    voitures.

    - **input** : liste de véhicules, chaque véhicule étant une liste de
      valeurs ordonnées selon `FEATURE_ORDER`.

    Retourne un objet JSON avec une clé `prediction` contenant la liste
    des prix prédits, dans le même ordre que les véhicules fournis.
    """
    X = pd.DataFrame(payload.input, columns=FEATURE_ORDER)
    preds = model.predict(X)
    return {"prediction": [round(float(p), 2) for p in preds]}
