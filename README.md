# Getaround Analysis — Deployment

Projet de déploiement (Bloc 5 — CDSD) : analyse des retards de checkout et
API de pricing pour Getaround.

## Structure

```
02_Code_Python/
├── data/                       # jeux de données source (xlsx + csv)
├── eda/
│   └── delay_analysis_eda.py   # analyse des retards, simulation seuil/scope
├── dashboard/
│   └── app_streamlit.py        # dashboard web pour le Product Management
├── ml_pricing/
│   └── train_model.py          # entraînement + tracking MLflow
├── api/
│   └── main.py                 # API FastAPI (/predict, /docs)
├── model/
│   └── pricing_model.joblib    # modèle final sérialisé (généré par train_model.py)
├── requirements.txt
└── Dockerfile
```

## Installation

```bash
pip install -r requirements.txt
```

## 1. Analyse des retards (EDA)

```bash
python eda/delay_analysis_eda.py
```

Génère les indicateurs clés et les graphiques dans `eda/outputs/`.

## 2. Dashboard web (Streamlit)

```bash
streamlit run dashboard/app_streamlit.py
```

## 3. Entraînement du modèle de pricing (MLflow)

```bash
python ml_pricing/train_model.py
mlflow ui   # http://localhost:5000 pour visualiser les runs
```

Produit `model/pricing_model.joblib`, utilisé par l'API, ainsi qu'un
dossier local `ml_pricing/mlruns/` (non fourni dans ce livrable pour
rester léger) contenant l'historique des runs consultable via `mlflow ui`.

## 4. API de prédiction (FastAPI)

```bash
uvicorn api.main:app --reload
```

- Documentation interactive : http://localhost:8000/docs
- Endpoint de prédiction : `POST /predict`

```bash
curl -i -H "Content-Type: application/json" -X POST \
  -d '{"input": [[140411, 100, true, true, false, false, true, true, true, "Citroën", "diesel", "black", "convertible"]]}' \
  http://localhost:8000/predict
```

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"input": [[140411, 100, True, True, False, False, True, True, True,
                      "Citroën", "diesel", "black", "convertible"]]},
)
print(response.json())
```

## 5. Déploiement

```bash
docker build -t getaround-api .
docker run -p 7860:7860 getaround-api
```

Prêt à être poussé sur **Hugging Face Spaces** (SDK: Docker).

## Sources de données

- [Delay Analysis](https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx)
- [Pricing Optimization](https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv)
