package villani.dev.vectorsearch.index.strategies.ivfpq;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Compresses each 14D vector into M=7 bytes using Product Quantization.
 * Each byte is the centroid index within a 2D subspace (256 centroids per subspace).
 *
 * Plain Java — no DI annotations; created by VectorIndexFactory.
 */
public class ProductQuantizer {

    public static final int M = 7;       // subquantizers (one per subspace)
    public static final int SUB_D = 2;   // dimensions per subspace (14/7)
    public static final int CODEBOOK_SIZE = 256;

    private final KMeans kMeans;
    private float[][][] codebooks; // [M][256][SUB_D]

    public ProductQuantizer(KMeans kMeans) {
        this.kMeans = kMeans;
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
    }

    /**
     * Encodes a 14D vector to 7 bytes (one centroid index per subspace).
     */
    public byte[] encode(float[] vec) {
        byte[] codes = new byte[M];
        float[] sub = new float[SUB_D];
        for (int m = 0; m < M; m++) {
            sub[0] = vec[m * SUB_D];
            sub[1] = vec[m * SUB_D + 1];
            codes[m] = (byte) kMeans.nearestSub(sub, codebooks[m], SUB_D);
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
     * Use with a ThreadLocal-cached table for zero-allocation hot path.
     */
    public void buildAdcTable(float[] query, float[][] table) {
        for (int m = 0; m < M; m++) {
            float q0 = query[m * SUB_D];
            float q1 = query[m * SUB_D + 1];
            for (int c = 0; c < CODEBOOK_SIZE; c++) {
                float d0 = q0 - codebooks[m][c][0];
                float d1 = q1 - codebooks[m][c][1];
                table[m][c] = d0 * d0 + d1 * d1;
            }
        }
    }

    /**
     * Approximate squared Euclidean distance between query and a PQ-encoded vector,
     * using the precomputed ADC table. Offset-based variant for flat byte[] storage
     * (avoids 3M small byte[7] object allocations and their JVM header overhead).
     */
    public float adcDistance(float[][] table, byte[] codes, int offset) {
        return table[0][codes[offset]     & 0xFF]
             + table[1][codes[offset + 1] & 0xFF]
             + table[2][codes[offset + 2] & 0xFF]
             + table[3][codes[offset + 3] & 0xFF]
             + table[4][codes[offset + 4] & 0xFF]
             + table[5][codes[offset + 5] & 0xFF]
             + table[6][codes[offset + 6] & 0xFF];
    }

    public float[][][] getCodebooks() {
        return codebooks;
    }

    public void setCodebooks(float[][][] codebooks) {
        this.codebooks = codebooks;
    }

    /** Bytes needed to serialize codebooks: M * 256 * SUB_D * 4 bytes/float */
    public int serializedSize() {
        return M * CODEBOOK_SIZE * SUB_D * Float.BYTES;
    }
}
