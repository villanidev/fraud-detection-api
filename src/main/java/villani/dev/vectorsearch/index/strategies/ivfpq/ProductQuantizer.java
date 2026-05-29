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
    public static final int CODEBOOK_SIZE = 2048;

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
     * Returns a flat table of length M * CODEBOOK_SIZE where entry for (m,c)
     * is at index (m * CODEBOOK_SIZE + c). This layout improves cache locality.
     */
    public float[] buildAdcTableFlat(float[] query) {
        float[] table = new float[M * CODEBOOK_SIZE];
        buildAdcTableFlat(query, table);
        return table;
    }

    /**
     * In-place variant — writes into a pre-allocated flat table (avoids allocation per request).
     * Table length must be at least M * CODEBOOK_SIZE.
     */
    public void buildAdcTableFlat(float[] query, float[] table) {
        for (int m = 0; m < M; m++) {
            float q0 = query[m * SUB_D];
            float q1 = query[m * SUB_D + 1];
            float[] cb = cbFlat[m];
            int base = m * CODEBOOK_SIZE;
            for (int c = 0; c < CODEBOOK_SIZE; c++) {
                float d0 = q0 - cb[c * 2];
                float d1 = q1 - cb[c * 2 + 1];
                table[base + c] = d0 * d0 + d1 * d1;
            }
        }
    }

    /**
     * Approximate squared Euclidean distance between query and a PQ-encoded vector,
     * using the precomputed ADC table. Offset-based variant for flat byte[] storage
     * (avoids 3M small byte[7] object allocations and their JVM header overhead).
     */
    public float adcDistanceFlat(float[] tableFlat, short[] codes, int offset) {
        // tableFlat is M * CODEBOOK_SIZE; access at m*CODEBOOK_SIZE + code
        return tableFlat[0 * CODEBOOK_SIZE + (codes[offset]     & 0xFFFF)]
             + tableFlat[1 * CODEBOOK_SIZE + (codes[offset + 1] & 0xFFFF)]
             + tableFlat[2 * CODEBOOK_SIZE + (codes[offset + 2] & 0xFFFF)]
             + tableFlat[3 * CODEBOOK_SIZE + (codes[offset + 3] & 0xFFFF)]
             + tableFlat[4 * CODEBOOK_SIZE + (codes[offset + 4] & 0xFFFF)]
             + tableFlat[5 * CODEBOOK_SIZE + (codes[offset + 5] & 0xFFFF)]
             + tableFlat[6 * CODEBOOK_SIZE + (codes[offset + 6] & 0xFFFF)];
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
