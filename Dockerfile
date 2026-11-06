# Getaround Pricing API - image Docker pour déploiement (Hugging Face Spaces / autre hébergeur)
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY model/ ./model/

EXPOSE 7860

# Hugging Face Spaces (SDK Docker) route le trafic vers le port 7860 par défaut
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
