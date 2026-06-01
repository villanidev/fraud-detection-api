#!/usr/bin/env bash
set -euo pipefail

RELEASE_TAG="villanidev/rinha-de-backend-2026:latest"
DATA_BIN="data-output/data.bin"

# Passo 1: Se o data.bin não existe, extraí ele rodando apenas o estágio de preprocessamento
if [[ -f "$DATA_BIN" ]]; then
  echo "==> [1/2] data.bin encontrado em $DATA_BIN, pulando bloco de 1 hora."
else
  echo "==> [1/2] data.bin NÃO encontrado. Executando pré-processamento pesado (aguarde ~1 hora)..."

  # Força o Docker a rodar especificamente até o estágio que gera o arquivo
  docker build --target preprocessor -t fraud-preprocessor-temp .

  # Cria o diretório local e extrai o arquivo gerado lá de dentro para a sua máquina hospedeira
  mkdir -p data-output
  docker run --rm --entrypoint cat fraud-preprocessor-temp /build/target/data.bin > "$DATA_BIN"
  docker rmi fraud-preprocessor-temp
fi

# Passo 2: Faz o build final ultra rápido injetando o data.bin que já está no cache local
echo "==> [2/2] Building release image rápida (data.bin baked in)..."
docker build -t "$RELEASE_TAG" .

echo " Imagem local pronta: ✓ $RELEASE_TAG"
echo ""
echo "Para subir:"
echo " docker compose -f docker-compose.unix.yaml up"
echo ""
echo "Para forçar novo preprocessing (apagar data.bin):"
echo " rm $DATA_BIN && ./build-local.sh"