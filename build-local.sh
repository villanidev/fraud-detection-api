#!/usr/bin/env bash
set -euo pipefail

BUILD_TAG="fraud-api-builder:latest"
RELEASE_TAG="villanidev/rinha-de-backend-2026:latest"
DATA_BIN="data-output/data.bin"

echo "==> [1/3] Building native image..."
docker build -f Dockerfile.native -t "$BUILD_TAG" .

if [[ -f "$DATA_BIN" ]]; then
    echo "==> [2/3] data.bin encontrado em $DATA_BIN, pulando preprocessing."
else
    echo "==> [2/3] Running preprocessing (aguarde ~6 minutos)..."
    mkdir -p data-output
    docker run --rm \
      -v "$(pwd)/data-output:/data" \
      -e REFERENCES_PATH=/app/refs/references.json.gz \
      -e NORMALIZATION_PATH=/app/refs/normalization.json \
      -e MCC_RISK_PATH=/app/refs/mcc_risk.json \
      -e DATA_BIN_PATH=/data/data.bin \
      "$BUILD_TAG" --preprocess
fi

echo "==> [3/3] Building release image (data.bin baked in)..."
docker build -f Dockerfile.release \
  --build-arg BUILD_IMAGE="$BUILD_TAG" \
  -t "$RELEASE_TAG" \
  .

echo ""
echo "✓ Imagem local pronta: $RELEASE_TAG"
echo ""
echo "Para subir com unix socket:"
echo "  docker compose -f docker-compose.unix.yaml up"
echo ""
echo "Para subir com TCP (haproxy padrão):"
echo "  docker compose up"
echo ""
echo "Para forçar novo preprocessing (apagar data.bin):"
echo "  rm $DATA_BIN && ./build-local.sh"
