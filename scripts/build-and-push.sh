#!/usr/bin/env bash
set -euo pipefail

DOCKERHUB_USER="villanidev"
IMAGE_NAME="rinha-de-backend-2026"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
RELEASE_TAG="$DOCKERHUB_USER/$IMAGE_NAME:$TIMESTAMP"
LATEST_TAG="$DOCKERHUB_USER/$IMAGE_NAME:latest"
DATA_BIN="data-output/data.bin"

echo "==> [0/4] Logging in to Docker Hub..."
echo "$DOCKERHUB_TOKEN" | docker login --username "$DOCKERHUB_USER" --password-stdin

# Passo 1: Condicional inteligente para o pré-processamento pesado
if [[ -f "$DATA_BIN" ]]; then
  echo "==> [1/4] data.bin encontrado em $DATA_BIN, pulando bloco de 1 hora."
else
  echo "==> [1/4] data.bin NÃO encontrado. Executando pré-processamento pesado (aguarde ~1 hora)..."

  # Compila e roda o pré-processador isolado dentro do Docker
  docker build --target preprocessor -t fraud-preprocessor-temp .

  # Extrai o arquivo binário gerado para a máquina hospedeira para servir de cache
  mkdir -p data-output
  docker run --rm --entrypoint cat fraud-preprocessor-temp /build/target/data.bin > "$DATA_BIN"
  docker rmi fraud-preprocessor-temp
fi

# Passo 2: Build final e ultra rápido usando o arquivo do cache local
echo "==> [2/4] Building release image (data.bin baked in)..."
docker build -t "$RELEASE_TAG" -t "$LATEST_TAG" .

# Passo 3: Envio das tags para o registro remoto
echo "==> [3/4] Pushing to Docker Hub..."
docker push "$RELEASE_TAG"
docker push "$LATEST_TAG"

echo ""
echo " ==> [4/4] Publicado com sucesso: ✓ $RELEASE_TAG"
echo "Para testar a imagem de produção localmente:"
echo " IMAGE=$RELEASE_TAG docker-compose up"