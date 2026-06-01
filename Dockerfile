# =========================================================================
# ESTÁGIO 1: Base do Builder com Liberica NIK
# =========================================================================
FROM bellsoft/liberica-native-image-kit-container:jdk-25-nik-25.0-glibc AS builder-base
WORKDIR /build
RUN apk add --no-cache maven

COPY pom.xml .
RUN mvn dependency:go-offline

COPY src src
RUN mvn package -Pnative-image -DskipTests

# =========================================================================
# ESTÁGIO 2: Pré-processador Condicional (Só executado pelo script se necessário)
# =========================================================================
FROM builder-base AS preprocessor
ENV REFERENCES_PATH=/build/src/main/resources/references.json.gz
ENV NORMALIZATION_PATH=/build/src/main/resources/normalization.json
ENV MCC_RISK_PATH=/build/src/main/resources/mcc_risk.json
ENV DATA_BIN_PATH=/build/target/data.bin

# Executa a tarefa pesada de 1 hora para gerar o data.bin interno
RUN /build/target/fraud-detection-api --preprocess

# =========================================================================
# ESTÁGIO 3: Runtime de Produção Limpo
# =========================================================================
FROM gcr.io/distroless/base-debian12
WORKDIR /app

# Copia a biblioteca dinâmica zlib necessária para o runtime
COPY --from=builder-base /usr/lib/libz.so.1 /lib/x86_64-linux-gnu/libz.so.1

# Copia o binário nativo do Helidon gerado no primeiro estágio
COPY --from=builder-base /build/target/fraud-detection-api /app/server

# Como o script garante a existência dele antes do build final, não há risco de falha.
COPY data-output/data.bin /app/data.bin

ENV DATA_BIN_PATH=/app/data.bin
EXPOSE 8080

ENTRYPOINT ["/app/server", "-Xms125m", "-Xmx125m"]