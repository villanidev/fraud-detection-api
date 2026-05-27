package villani.dev.vectorsearch.index.strategies.ivfpq;

/**
 * Compresses each 14D vector into M=7 bytes using Product Quantization.
 * Each byte is the centroid index within a 2D subspace (256 centroids per subspace).
 *
 * Plain Java — no DI annotations; created by VectorIndexFactory.
 */
public class ProductQuantizer {

    public static final int M = 7;       // subquantizers (one per subspace)
    public static final int SUB_D = 2;   // dimensions per subspace (14/7)
    public static final int CODEBOOK_SIZE = 4096;

    private final KMeans kMeans;
    private float[][][] codebooks; // [M][CODEBOOK_SIZE][SUB_D] — kept for serialization + encode()
    private float[][] cbFlat;      // [M][CODEBOOK_SIZE * SUB_D] — interleaved, contiguous, cache-friendly

    public ProductQuantizer(KMeans kMeans) {
        this.kMeans = kMeans;
    }

    /** Converts the 3D codebook array into a flat 2D layout for cache-friendly access. */
    private static float[][] toFlat(float[][][] cb) {
        int M = cb.length;
        int C = cb[0].length;
        int D = cb[0][0].length;
        float[][] flat = new float[M][C * D];
        for (int m = 0; m < M; m++)
            for (int c = 0; c < C; c++)
                for (int d = 0; d < D; d++)
                    flat[m][c * D + d] = cb[m][c][d];
        return flat;
    }

    /**
     * Trains codebooks from full 14D vectors.
     * Extracts 2D sub-vectors per subspace and runs K-Means(K=256) on each.
     */
    public void train(float[][] vectors, long seed) {
        codebooks = new float[M][CODEBOOK_SIZE][SUB_D];
        float[][] subVectors = new float[vectors.length][SUB_D];

        for (int m = 0; m < M; m++) {
            int offset = m * SUB_D;
            for (int i = 0; i < vectors.length; i++) {
                subVectors[i][0] = vectors[i][offset];
                subVectors[i][1] = vectors[i][offset + 1];
            }
            codebooks[m] = kMeans.clusterSub(subVectors, CODEBOOK_SIZE, seed + m);
        }
        cbFlat = toFlat(codebooks);
    }

    /** Overload: train from flat vectors (row-major) with known N to avoid full matrix allocation externally. */
    public void train(float[] vectorsFlat, int N, long seed) {
        codebooks = new float[M][CODEBOOK_SIZE][SUB_D];
        for (int m = 0; m < M; m++) {
            int offset = m * SUB_D;
            float[][] subVectors = new float[N][SUB_D];
            for (int i = 0; i < N; i++) {
                int base = i * (M * SUB_D); // but M*SUB_D == 14
                // base should be i*14
                base = i * (M * SUB_D);
                subVectors[i][0] = vectorsFlat[base + offset];
                subVectors[i][1] = vectorsFlat[base + offset + 1];
            }
            codebooks[m] = kMeans.clusterSub(subVectors, CODEBOOK_SIZE, seed + m);
        }
        cbFlat = toFlat(codebooks);
    }

    /**
     * Encodes a 14D vector to 7 bytes (one centroid index per subspace).
     */
    public short[] encode(float[] vec) {
        short[] codes = new short[M];
        float[] sub = new float[SUB_D];
        for (int m = 0; m < M; m++) {
            sub[0] = vec[m * SUB_D];
            sub[1] = vec[m * SUB_D + 1];
            codes[m] = (short) kMeans.nearestSub(sub, codebooks[m], SUB_D);
        }
        return codes;
    }

    /** Encode using flat array without allocating small temporary float[14] per vector. */
    public short[] encodeFlat(float[] flat, int idx) {
        short[] codes = new short[M];
        float[] sub = new float[SUB_D];
        int base = idx * (M * SUB_D); // 14
        for (int m = 0; m < M; m++) {
            int off = base + m * SUB_D;
            sub[0] = flat[off];
            sub[1] = flat[off + 1];
            codes[m] = (short) kMeans.nearestSub(sub, codebooks[m], SUB_D);
        }
        return codes;
    }

    /**
     * Precomputes the ADC (Asymmetric Distance Computation) table for a query.
     * table[m][c] = squared distance from query subvector m to codebook[m][c].
     * This is computed once per query and reused for all candidate vectors.
     */
    public float[][] buildAdcTable(float[] query) {
        float[][] table = new float[M][CODEBOOK_SIZE];
        buildAdcTable(query, table);
        return table;
    }

    /**
     * In-place variant — writes into a pre-allocated table (avoids float[M][256] allocation per request).
     * Uses the flat codebook layout [M][C*SUB_D] for cache-friendly sequential access
     * (no pointer chasing through float[M][256][2]).
     * Use with a ThreadLocal-cached table for zero-allocation hot path.
     */
    public void buildAdcTable(float[] query, float[][] table) {
        for (int m = 0; m < M; m++) {
            float q0 = query[m * SUB_D];
            float q1 = query[m * SUB_D + 1];
            float[] cb = cbFlat[m];
            float[] row = table[m];
            for (int c = 0; c < CODEBOOK_SIZE; c++) {
                float d0 = q0 - cb[c * 2];
                float d1 = q1 - cb[c * 2 + 1];
                row[c] = d0 * d0 + d1 * d1;
            }
        }
    }

    /**
     * Approximate squared Euclidean distance between query and a PQ-encoded vector,
     * using the precomputed ADC table. Offset-based variant for flat byte[] storage
     * (avoids 3M small byte[7] object allocations and their JVM header overhead).
     */
    public float adcDistance(float[][] table, short[] codes, int offset) {
        return table[0][codes[offset]     & 0xFFFF]
             + table[1][codes[offset + 1] & 0xFFFF]
             + table[2][codes[offset + 2] & 0xFFFF]
             + table[3][codes[offset + 3] & 0xFFFF]
             + table[4][codes[offset + 4] & 0xFFFF]
             + table[5][codes[offset + 5] & 0xFFFF]
             + table[6][codes[offset + 6] & 0xFFFF];
    }

    public float[][][] getCodebooks() {
        return codebooks;
    }

    public void setCodebooks(float[][][] codebooks) {
        this.codebooks = codebooks;
        this.cbFlat = toFlat(codebooks);
    }

    /** Bytes needed to serialize codebooks: M * 256 * SUB_D * 4 bytes/float */
    public int serializedSize() {
        return M * CODEBOOK_SIZE * SUB_D * Float.BYTES;
    }
}
