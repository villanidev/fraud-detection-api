#!/usr/bin/env bash
set -euo pipefail

DOCKERHUB_USER="villanidev"
IMAGE_NAME="rinha-de-backend-2026"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
BUILD_TAG="fraud-api-builder:latest"
RELEASE_TAG="$DOCKERHUB_USER/$IMAGE_NAME:$TIMESTAMP"
LATEST_TAG="$DOCKERHUB_USER/$IMAGE_NAME:latest"
DATA_BIN="data-output/data.bin"

echo "==> [0/4] Logging in to Docker Hub..."
echo "$DOCKERHUB_TOKEN" | docker login --username "$DOCKERHUB_USER" --password-stdin

echo "==> [1/4] Building native image..."
docker build -f Dockerfile.native -t "$BUILD_TAG" .

if [[ -f "$DATA_BIN" ]]; then
    echo "==> [2/4] data.bin encontrado em $DATA_BIN, pulando preprocessing."
else
    echo "==> [2/4] Running preprocessing (aguarde alguns minutos)..."
    mkdir -p data-output
    docker run --rm \
      -v "$(pwd)/data-output:/data" \
      -e REFERENCES_PATH=/app/refs/references.json.gz \
      -e NORMALIZATION_PATH=/app/refs/normalization.json \
      -e MCC_RISK_PATH=/app/refs/mcc_risk.json \
      -e DATA_BIN_PATH=/data/data.bin \
      "$BUILD_TAG" --preprocess
fi

echo "==> [3/4] Building release image (data.bin baked in)..."
docker build -f Dockerfile.release \
  --build-arg BUILD_IMAGE="$BUILD_TAG" \
  -t "$RELEASE_TAG" \
  -t "$LATEST_TAG" \
  .

echo "==> [4/4] Pushing to Docker Hub..."
docker push "$RELEASE_TAG"
docker push "$LATEST_TAG"

echo ""
echo "✓ Publicado: $RELEASE_TAG"
echo ""
echo "Para testar localmente:"
echo "  IMAGE=$RELEASE_TAG docker-compose up"
