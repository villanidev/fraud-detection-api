package villani.dev.vectorsearch.index.strategies.ivfpq;

/**
 * Compresses each 14D vector into M=7 bytes using Product Quantization.
 * Each byte is the centroid index within a 2D subspace (256 centroids per subspace).
 *
 * Plain Java — no DI annotations; created by VectorIndexFactory.
 */
public class ProductQuantizer {

    private static final int CARD_PRESENT_DIM = 10;
    private static final int UNKNOWN_MERCHANT_DIM = 11;

    public static final int M = 7;       // subquantizers (one per subspace)
    public static final int SUB_D = 2;   // dimensions per subspace (14/7)
    public static final int CODEBOOK_SIZE = 2048;

    private final KMeans kMeans;
    // Flattened codebooks: layout [ (m * CODEBOOK_SIZE + c) * SUB_D + d ]
    private float[] codebooksFlat; // length = M * CODEBOOK_SIZE * SUB_D

    public ProductQuantizer(KMeans kMeans) {
        this.kMeans = kMeans;
    }

    // helper: none (we store directly into flattened layout)

    /** Overload: train from flat vectors (row-major) with known N to avoid full matrix allocation externally. */
    public void train(float[] vectorsFlat, int N, long seed) {
        this.codebooksFlat = new float[M * CODEBOOK_SIZE * SUB_D];
        for (int m = 0; m < M; m++) {
            float[][] codebook = shouldUseExactDiscreteCodebook(m)
                    ? buildExactDiscreteCodebook(m)
                    : kMeans.clusterSubFlat(vectorsFlat, N, m * SUB_D, CODEBOOK_SIZE, seed + m);
            for (int c = 0; c < CODEBOOK_SIZE; c++) {
                int base = (m * CODEBOOK_SIZE + c) * SUB_D;
                this.codebooksFlat[base] = codebook[c][0];
                this.codebooksFlat[base + 1] = codebook[c][1];
            }
        }
    }

    private boolean shouldUseExactDiscreteCodebook(int subspace) {
        int dim0 = subspace * SUB_D;
        int dim1 = dim0 + 1;
        return dim0 == CARD_PRESENT_DIM && dim1 == UNKNOWN_MERCHANT_DIM;
    }

    private float[][] buildExactDiscreteCodebook(int subspace) {
        System.out.printf("[ProductQuantizer] subspace=%d uses exact discrete codebook; skipping KMeans.%n", subspace);
        float[][] codebook = new float[CODEBOOK_SIZE][SUB_D];
        codebook[0][0] = 0f;
        codebook[0][1] = 0f;
        codebook[1][0] = 0f;
        codebook[1][1] = 1f;
        codebook[2][0] = 1f;
        codebook[2][1] = 0f;
        codebook[3][0] = 1f;
        codebook[3][1] = 1f;
        for (int c = 4; c < CODEBOOK_SIZE; c++) {
            codebook[c][0] = codebook[c & 3][0];
            codebook[c][1] = codebook[c & 3][1];
        }
        return codebook;
    }

    /** Encode using flat array without allocating small temporary float[14] per vector. */
    public short[] encodeFlat(float[] flat, int idx) {
        short[] codes = new short[M];
        encodeFlatInto(flat, idx, codes, 0);
        return codes;
    }

    public void encodeFlatInto(float[] flat, int idx, short[] target, int targetOffset) {
        int base = idx * (M * SUB_D); // 14
        for (int m = 0; m < M; m++) {
            int off = base + m * SUB_D;
            target[targetOffset + m] = (short) nearestSubFlat(flat[off], flat[off + 1], m);
        }
    }

    /**
     * In-place variant — writes into a pre-allocated flat table (avoids allocation per request).
     * Table length must be at least M * CODEBOOK_SIZE.
     */
    public void buildAdcTableFlat(float[] query, float[] table) {
        for (int m = 0; m < M; m++) {
            float q0 = query[m * SUB_D];
            float q1 = query[m * SUB_D + 1];
            int base = (m * CODEBOOK_SIZE) * SUB_D;
            int outBase = m * CODEBOOK_SIZE;
            for (int c = 0; c < CODEBOOK_SIZE; c++) {
                int cbIdx = base + c * SUB_D;
                float d0 = q0 - codebooksFlat[cbIdx];
                float d1 = q1 - codebooksFlat[cbIdx + 1];
                table[outBase + c] = d0 * d0 + d1 * d1;
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
        return tableFlat[0 * CODEBOOK_SIZE + codes[offset]]
             + tableFlat[1 * CODEBOOK_SIZE + codes[offset + 1]]
             + tableFlat[2 * CODEBOOK_SIZE + codes[offset + 2]]
             + tableFlat[3 * CODEBOOK_SIZE + codes[offset + 3]]
             + tableFlat[4 * CODEBOOK_SIZE + codes[offset + 4]]
             + tableFlat[5 * CODEBOOK_SIZE + codes[offset + 5]]
             + tableFlat[6 * CODEBOOK_SIZE + codes[offset + 6]];
    }

    /** Return the flattened codebooks for serialization or inspection. */
    public float[] getCodebooksFlat() {
        return codebooksFlat;
    }

    public void setCodebooksFlat(float[] codebooksFlat) {
        this.codebooksFlat = codebooksFlat;
    }

    /** Bytes needed to serialize codebooks: M * 256 * SUB_D * 4 bytes/float */
    public int serializedSize() {
        return M * CODEBOOK_SIZE * SUB_D * Float.BYTES;
    }

    /** Find nearest centroid index for subvector in subspace m using flat codebooks. */
    private int nearestSubFlat(float v0, float v1, int m) {
        int best = 0;
        double bestDist = Double.MAX_VALUE;
        int base = (m * CODEBOOK_SIZE) * SUB_D;
        for (int c = 0; c < CODEBOOK_SIZE; c++) {
            int idx = base + c * SUB_D;
            double d0 = v0 - codebooksFlat[idx];
            double d1 = v1 - codebooksFlat[idx + 1];
            double dist = d0 * d0 + d1 * d1;
            if (dist < bestDist) {
                bestDist = dist;
                best = c;
            }
        }
        return best;
    }
}
