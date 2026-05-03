#!/usr/bin/env bash
# run-local-test.sh — Simulates the Rinha de Backend 2026 evaluation locally.
#
# What this does:
#   1. (Optional) Builds the native Docker image via the 3-step pipeline in README
#   2. Starts docker-compose (nginx + api1 + api2) with the same resource limits as production
#   3. Waits for both instances to be ready (/ready endpoint)
#   4. Runs k6 with the same incremental load profile
#   5. Prints scoring estimate
#
# Usage:
#   ./test/run-local-test.sh                        # usa imagem já existente (padrão)
#   ./test/run-local-test.sh --build                # reconstrói a imagem nativa (~5 min)
#   ./test/run-local-test.sh --vus 100 --duration 120s
#
# Dependencies: docker, k6

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Defaults ────────────────────────────────────────────────────────────────
VUS=50
DURATION=90s
BUILD=false  # imagem nativa já existe localmente por padrão
IMAGE="${IMAGE:-villanidev/rinha-de-backend-2026:latest}"

# ── Arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vus)      VUS="$2";      shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --build)    BUILD=true;    shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Rinha de Backend 2026 — Local Test"
echo "  VUs: $VUS | Duration: $DURATION"
echo "══════════════════════════════════════════════════════"

# ── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[cleanup] Stopping docker-compose..."
    docker compose down --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

# ── 1. Build (opcional — imagem nativa leva ~5 min) ──────────────────────────
if [[ "$BUILD" == "true" ]]; then
    echo ""
    echo "[build] Step 1/3: Compilando binário nativo via Dockerfile.native..."
    docker build -f Dockerfile.native -t fraud-api-builder:latest .

    echo ""
    echo "[build] Step 2/3: Gerando data.bin..."
    mkdir -p data-output
    docker run --rm \
    -v "$(pwd)/data-output:/data" \
    -e REFERENCES_PATH=/app/refs/references.json.gz \
    -e NORMALIZATION_PATH=/app/refs/normalization.json \
    -e MCC_RISK_PATH=/app/refs/mcc_risk.json \
    -e DATA_BIN_PATH=/data/data.bin \
    fraud-api-builder:latest --preprocess

    echo ""
    echo "[build] Step 3/3: Baking data.bin na imagem final: $IMAGE"
    docker build -f Dockerfile.release \
        --build-arg BUILD_IMAGE=fraud-api-builder:latest \
        -t "$IMAGE" .
    echo "[build] Done."
else
    echo ""
    echo "[build] Usando imagem existente: $IMAGE (passe --build para reconstruir)"
fi

# ── 2. Start stack ───────────────────────────────────────────────────────────
echo ""
echo "[compose] Starting nginx + api1 + api2 with resource limits..."
IMAGE="$IMAGE" docker compose up -d --remove-orphans

# ── 3. Wait for readiness ────────────────────────────────────────────────────
echo ""
echo "[ready] Waiting for /ready on both instances..."

wait_ready() {
    local url="$1"
    local name="$2"
    local max_attempts=60
    local attempt=0
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -sf "$url/ready" > /dev/null 2>&1; then
            echo "[ready] $name is up ✓"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    echo "[ready] ERROR: $name failed to start after 120s"
    docker compose logs
    exit 1
}

# Wait for nginx (which proxies to both backends)
wait_ready "http://localhost:9999" "nginx→api1/api2"
# Binário nativo: startup < 20ms, sem JVM warm-up necessário

# ── 4. Quick smoke test ──────────────────────────────────────────────────────
echo ""
echo "[smoke] Sending a test request..."
SMOKE_BODY='{"id":"tx-smoke","transaction":{"amount":100.0,"installments":1,"requested_at":"2026-03-11T10:00:00Z"},"customer":{"avg_amount":200.0,"tx_count_24h":2,"known_merchants":["MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5411","avg_amount":150.0},"terminal":{"is_online":false,"card_present":true,"km_from_home":5.0},"last_transaction":null}'
SMOKE_RESULT=$(curl -sf -X POST http://localhost:9999/fraud-score \
    -H 'Content-Type: application/json' \
    -d "$SMOKE_BODY" 2>&1 || echo "FAILED")

if [[ "$SMOKE_RESULT" == *"approved"* ]]; then
    echo "[smoke] OK → $SMOKE_RESULT"
else
    echo "[smoke] ERROR: $SMOKE_RESULT"
    docker compose logs
    exit 1
fi

# ── 5. Run k6 ────────────────────────────────────────────────────────────────
echo ""
echo "[k6] Starting load test..."
echo ""

k6 run \
    --env VUS="$VUS" \
    --env DURATION="$DURATION" \
    --env HOST="http://localhost:9999" \
    test/k6-local.js

echo ""
echo "[done] Results saved to test/results-local.json"
