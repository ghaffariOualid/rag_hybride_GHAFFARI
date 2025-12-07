# Dockerfile

# Utiliser l'image de base Python 3.12 (fidèle à RAGio)
FROM python:3.12-slim-bookworm

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers d'exigence et installer les dépendances
COPY requirements.txt .

# Installer les dépendances
# Torch est nécessaire pour les modèles Sentence-Transformers/Cross-Encoder
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt 

# Copier tout le code source
COPY . .

# Exposer le port API
EXPOSE 8001

# Définir le point d'entrée pour exécuter l'API via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]