"""
Getaround - Pricing Optimization
=================================

Entraîne un modèle de Machine Learning qui prédit le prix de location
journalier optimal d'un véhicule (rental_price_per_day) à partir de ses
caractéristiques (modèle, kilométrage, puissance, équipements...).

Le script :
    - charge et prépare les données,
    - compare plusieurs algorithmes de régression via validation croisée,
    - trace les expériences avec MLflow (paramètres, métriques, modèle),
    - réentraîne le meilleur modèle sur l'ensemble des données,
    - sérialise le pipeline final (préprocessing + modèle) pour l'API.

Données source :
https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv

Usage :
    python train_model.py
    mlflow ui   # pour visualiser les runs (http://localhost:5000)
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "get_around_pricing_project.csv"
MODEL_DIR = BASE_DIR.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "rental_price_per_day"
NUM_FEATURES = ["mileage", "engine_power"]
BOOL_FEATURES = [
    "private_parking_available",
    "has_gps",
    "has_air_conditioning",
    "automatic_car",
    "has_getaround_connect",
    "has_speed_regulator",
    "winter_tires",
]
CAT_FEATURES = ["model_key", "fuel", "paint_color", "car_type"]
ALL_FEATURES = NUM_FEATURES + BOOL_FEATURES + CAT_FEATURES


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    for c in BOOL_FEATURES:
        df[c] = df[c].astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ],
        remainder="passthrough",  # boolean features, already numeric (0/1)
    )


def get_candidate_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }


def main():
    mlflow.set_experiment("getaround-pricing-optimization")

    df = load_data()
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_name, best_pipe, best_cv_r2 = None, None, -np.inf

    for name, model in get_candidate_models().items():
        pipe = Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])

        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = r2_score(y_test, preds)
            cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()

            mlflow.log_param("model_type", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("mae_test", mae)
            mlflow.log_metric("rmse_test", rmse)
            mlflow.log_metric("r2_test", r2)
            mlflow.log_metric("r2_cv5", cv_r2)
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            print(f"{name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2_test={r2:.3f}  R2_cv5={cv_r2:.3f}")

            if cv_r2 > best_cv_r2:
                best_cv_r2, best_name, best_pipe = cv_r2, name, pipe

    print(f"\nMeilleur modèle : {best_name} (R2 cross-val = {best_cv_r2:.3f})")

    # Réentraînement du meilleur modèle sur l'intégralité des données,
    # pour maximiser la qualité du modèle mis en production.
    final_model = get_candidate_models()[best_name]
    final_pipe = Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", final_model)])
    final_pipe.fit(X, y)

    with mlflow.start_run(run_name=f"{best_name}-final-full-data"):
        mlflow.log_param("model_type", best_name)
        mlflow.log_param("trained_on", "full_dataset")
        mlflow.sklearn.log_model(final_pipe, artifact_path="model", registered_model_name="getaround-pricing-model")

    model_path = MODEL_DIR / "pricing_model.joblib"
    joblib.dump(final_pipe, model_path)
    print(f"Modèle final sauvegardé : {model_path}")


if __name__ == "__main__":
    main()
