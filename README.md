# fraud-detection-api

API de detecção de fraude em transações para a Rinha de Backend 2026.

---

## Arquitetura

Cada instância da API executa o mesmo pipeline a cada requisição:

```
TransactionRequest
      │
      ▼
  embed()          → transforma campos numéricos em vetor
      │
      ▼
  search()         → IVF-PQ: busca aproximada nos 3M vetores de referência
      │              (K=2176 centroids, nprobe=6, candidates=10, rerank=true)
      ▼
  score()          → regras determinísticas sobre os vizinhos encontrados
      │
      ▼
  DecisionResponse { approved, score }
```

O índice (`data.bin`, ~220 MB) é gerado **uma vez** na sua máquina e baked na imagem Docker antes de publicar. Em produção a API nunca escreve em disco.

---

## Stack

| Camada        | Tecnologia                                                  |
| ------------- |-------------------------------------------------------------|
| Linguagem     | Java 21                                                     |
| Framework     | Helidon 4 SE (declarative DI, sem reflection em runtime)    |
| Runtime       | GraalVM CE 25 -> Native Image → binário nativo de ~89 MB    |
| Imagem base   | `gcr.io/distroless/base-debian12` (sem shell, sem OS extra) |
| Load balancer | haproxy:alpine                                              |
| Algoritmo     | IVF-PQ (Inverted File + Product Quantization)               |

### Por que GraalVM Native Image?

O compilador GraalVM analisa todo o código em tempo de compilação (closed-world assumption) e gera um binário nativo:

- **Startup em < 20ms** (vs. ~800ms na JVM)
- **Menor footprint de memória** — sem JIT, sem metaspace, sem warmup
- **Sem JVM em runtime** — o binário roda diretamente sobre o kernel
- Trade-off: build demora ~2 a 5 minutos e requer `mvn package -Pnative-image`

---

## Rodando localmente

### Pré-requisitos

- Java 21+ na `PATH` (`java -version`)
- Maven 3.9+ (`mvn -version`)
- O arquivo `src/main/resources/references.json.gz` presente

### 1. Gerar o índice (`data.bin`) — obrigatório antes de subir a API

Isso leva ~1/2 horas na primeira vez. Só precisa rodar uma vez (ou quando o dataset mudar).

```bash
mvn package -DskipTests
java -jar target/fraud-detection-api.jar --preprocess
# data.bin será gerado na raiz do projeto
```

### 2a. Rodar via JVM

```bash
mvn package -DskipTests
java -jar target/fraud-detection-api.jar
```

Saída esperada:

```
2026.05.03 01:28:16.230 INFO Started all channels in 15 milliseconds. 768 milliseconds since JVM startup. Java 25.0.2+10-jvmci-b01
Loading references (data.bin) ...
Fraud detection API started on port: 8080
```

### 2b. Rodar via binário nativo (requer GraalVM instalado localmente)

```bash
mvn package -Pnative-image -DskipTests
./target/fraud-detection-api
```

Saída esperada:

```
2026.05.02 23:31:59.350 INFO Started all channels in 1 milliseconds. 20 milliseconds since JVM startup. Java 25.0.2+10-jvmci-b01
Loading references (data.bin) ...
Fraud detection API started on port: 8080
```

> Para instalar o GraalVM localmente: `sdk install java 25.0.x-graal` (SDKMAN) ou baixe em https://www.graalvm.org/downloads/

---

## Imagens Docker

### Build local para teste (sem publicar)

```bash
# 1. Compila e gera o binário nativo dentro do Docker
docker build -f Dockerfile.native -t fraud-api-builder:latest .

# 2. Gera o data.bin localmente via container
mkdir -p data-output
docker run --rm \
  -v "$(pwd)/data-output:/data" \
  -e REFERENCES_PATH=/app/refs/references.json.gz \
  -e NORMALIZATION_PATH=/app/refs/normalization.json \
  -e MCC_RISK_PATH=/app/refs/mcc_risk.json \
  -e DATA_BIN_PATH=/data/data.bin \
  fraud-api-builder:latest --preprocess

# 3. Bake do data.bin na imagem final
docker build -f Dockerfile.release \
  --build-arg BUILD_IMAGE=fraud-api-builder:latest \
  -t fraud-api-local:latest \
  .

# 4. Sobe o stack completo com a imagem local
IMAGE=fraud-api-local:latest docker-compose up
```

### Build + push para Docker Hub (CI/submissão)

```bash
export DOCKERHUB_TOKEN=seu_token_aqui
./build-and-push.sh
```

O script executa as etapas 1-4 acima e depois publica `villanidev/rinha-de-backend-2026:latest` e uma tag com timestamp.

---

## Endpoints

### `POST /fraud-score`

Avalia se uma transação é fraude.

```bash
curl -s -X POST http://localhost:9999/fraud-score \
  -H 'Content-Type: application/json' \
  -d '{
    "transaction_id": "txn-001",
    "amount": 9500.00,
    "merchant_category_code": 5912,
    "transaction_hour": 3,
    "distance_from_home_km": 450.0,
    "is_foreign_transaction": false,
    "is_new_merchant": true,
    "transactions_last_hour": 5,
    "avg_transaction_value": 120.0
  }'
```

Resposta:

```json
{
  "approved": false,
  "score": 0.8
}
```

Para testar direto na API (sem nginx), use porta `8080` no lugar de `9999`. Útil para debug de instância específica.

### `GET /ready`

Health check — retorna 204 quando a API está pronta (índice carregado).

```bash
curl -s http://localhost:9999/ready
```

### Teste em lote com o arquivo de exemplo

```bash
jq -c '.[]' src/main/resources/example-payloads.json | while read p; do
  curl -s -X POST http://localhost:9999/fraud-score \
    -H 'Content-Type: application/json' \
    -d "$p" | jq -c '.'
done
```

---

## Limites de recursos (competição)

| Serviço   | CPU     | Memória    |
|-----------|---------|------------|
| haproxy   | 0.15    | 50 MB      |
| api1      | 0.425   | 150 MB     |
| api2      | 0.425   | 150 MB     |
| **Total** | **1.0** | **350 MB** |

---

## Variáveis de ambiente

| Variável                     | Padrão                                  | Descrição                                      |
| ---------------------------- | --------------------------------------- | ---------------------------------------------- |
| `REFERENCES_PATH`            | `src/main/resources/references.json.gz` | Dataset de referência                          |
| `NORMALIZATION_PATH`         | `src/main/resources/normalization.json` | Parâmetros de normalização                     |
| `MCC_RISK_PATH`              | `src/main/resources/mcc_risk.json`      | Score de risco por MCC                         |
| `DATA_BIN_PATH`              | `data.bin`                              | Caminho do índice gerado                       |
| `APP_VECTOR_SEARCH_INDEX`    | `ivf_pq`                                | Algoritmo de busca (`brute_force` ou `ivf_pq`) |
| `APP_VECTOR_SEARCH_RERANK`   | `true`                                  | Rerank dos candidatos por distância exata      |
| `APP_VECTOR_SEARCH_NPROBE`   | `16`                                    | Número de clusters inspecionados por query     |
| `APP_VECTOR_SEARCH_CANDIDATES` | `50`                                    | Candidatos coarse antes do rerank              |
