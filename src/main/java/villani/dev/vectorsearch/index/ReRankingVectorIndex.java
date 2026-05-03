package villani.dev.vectorsearch.index;

import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.util.Arrays;

/**
 * Decorator that adds exact-distance reranking on top of any VectorIndex.
 *
 * Two-stage search:
 *   Stage 1 — delegate to inner index for `candidates` approximate results.
 *   Stage 2 — rerank those candidates with exact squared Euclidean distance
 *              read directly from the memory-mapped vectors section of data.bin.
 *              Only ~50 × 56 bytes are paged in per query — zero heap cost.
 *
 * Pattern: Decorator (GoF) — wraps any VectorIndex, adds behaviour without
 *           modifying the wrapped class (OCP).
 */
public class ReRankingVectorIndex implements VectorIndex {

    private static final int DIMS = 14;
    private static final int VECTOR_BYTES = DIMS * Float.BYTES; // 56 bytes

    private final VectorIndex inner;
    private final MappedByteBuffer vectorsMmap;
    private final long vectorsOffset;   // byte offset of the first vector in data.bin
    private final int vectorCount;
    private final int candidates;       // coarse results from inner before rerank

    /**
     * @param inner         wrapped index (e.g. IVFPQIndex)
     * @param vectorsMmap   memory-mapped view of the full data.bin file
     * @param vectorsOffset byte position of the first original vector in data.bin
     * @param vectorCount   total number of reference vectors
     * @param candidates    number of approximate results to rerank (must be ≥ k)
     */
    public ReRankingVectorIndex(VectorIndex inner,
                                MappedByteBuffer vectorsMmap,
                                long vectorsOffset,
                                int vectorCount,
                                int candidates) {
        this.inner = inner;
        this.vectorsMmap = (MappedByteBuffer) vectorsMmap.duplicate().order(ByteOrder.BIG_ENDIAN);
        this.vectorsOffset = vectorsOffset;
        this.vectorCount = vectorCount;
        this.candidates = candidates;
    }

    @Override
    public int search(float[] query, int k, int[] neighbors, float[] distances) {
        // Stage 1: coarse ANN — get `candidates` approximate results
        int[] coarseNeighbors = new int[candidates];
        float[] coarseDists = new float[candidates];
        inner.search(query, candidates, coarseNeighbors, coarseDists);

        // Stage 2: exact rerank over the candidate set
        float[] exactDists = new float[candidates];
        float[] vec = new float[DIMS];
        for (int i = 0; i < candidates; i++) {
            int id = coarseNeighbors[i];
            if (id < 0) {
                exactDists[i] = Float.MAX_VALUE;
                continue;
            }
            readVector(id, vec);
            exactDists[i] = squaredDistance(query, vec);
        }

        // Partial sort to get true top-k
        Arrays.fill(distances, Float.MAX_VALUE);
        Arrays.fill(neighbors, -1);
        for (int i = 0; i < candidates; i++) {
            if (coarseNeighbors[i] >= 0 && exactDists[i] < distances[k - 1]) {
                insertSorted(neighbors, distances, k, coarseNeighbors[i], exactDists[i]);
            }
        }

        // Count fraud labels in exact top-k
        byte[] labels = inner.getLabels();
        int fraudCount = 0;
        for (int i = 0; i < k; i++) {
            if (neighbors[i] >= 0 && labels[neighbors[i]] == 1) fraudCount++;
        }
        return fraudCount;
    }

    /** Reads 14 floats for vector `id` from the mmap without heap allocation. */
    private void readVector(int id, float[] out) {
        long pos = vectorsOffset + (long) id * VECTOR_BYTES;
        FloatBuffer fb = ((MappedByteBuffer) vectorsMmap.position((int) pos))
                .asFloatBuffer();
        fb.get(out, 0, DIMS);
    }

    private static void insertSorted(int[] neighbors, float[] distances, int k, int id, float dist) {
        int pos = k - 1;
        while (pos > 0 && distances[pos - 1] > dist) {
            distances[pos] = distances[pos - 1];
            neighbors[pos] = neighbors[pos - 1];
            pos--;
        }
        distances[pos] = dist;
        neighbors[pos] = id;
    }

    private static float squaredDistance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < DIMS; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    @Override
    public byte[] getLabels() {
        return inner.getLabels();
    }

    @Override
    public void close() {
        inner.close();
    }
}
