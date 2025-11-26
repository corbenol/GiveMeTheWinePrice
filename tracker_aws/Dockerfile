FROM python:3.10-slim

WORKDIR /home/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

#client amazon AWS CLI v2
# ---------------------------------------------------------------------
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws/

#dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
ENV PORT=5000

# sécurisation user
RUN groupadd -r unprivileged
RUN useradd -r -g unprivileged mlflowuser
RUN chsh -s /usr/sbin/nologin root

USER mlflowuser

#Lancement du serveur MLflow


CMD ["sh", "-c", "mlflow server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${ARTIFACT_ROOT}"]
