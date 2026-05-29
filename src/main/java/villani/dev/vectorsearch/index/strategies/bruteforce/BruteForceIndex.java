package villani.dev.vectorsearch.index.strategies.bruteforce;

import villani.dev.vectorsearch.index.VectorIndex;

import java.util.Arrays;

/**
 * Exact K-NN via brute-force O(N*D) scan.
 * Use only for correctness validation — not viable for 3M vectors at runtime.
 *
 * Plain Java — no DI annotations; created by VectorIndexFactory.
 */
public class BruteForceIndex implements VectorIndex {

    private static final int DIMS = 14;
    private final float[] vectorsFlat;
    private final byte[] labels;

    public BruteForceIndex(float[] vectorsFlat, byte[] labels) {
        this.vectorsFlat = vectorsFlat;
        this.labels = labels;
    }

    @Override
    public int search(float[] query, int topK, int candidates, int[] neighbors, float[] distances) {
        Arrays.fill(distances, Float.MAX_VALUE);
        Arrays.fill(neighbors, -1);

        int N = labels.length;
        for (int i = 0; i < N; i++) {
            float d = squaredDistance(query, vectorsFlat, i);
            if (d < distances[topK - 1]) {
                insertSorted(neighbors, distances, topK, i, d);
            }
        }

        int fraudCount = 0;
        for (int i = 0; i < topK; i++) {
            if (neighbors[i] >= 0 && labels[neighbors[i]] == 1) fraudCount++;
        }
        return fraudCount;
    }

    /**
     * Inserts (id, dist) into the sorted top-k arrays, maintaining ascending distance order.
     * k is small (≤50), so a linear shift is optimal — no heap overhead.
     */
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

    private static float squaredDistance(float[] a, float[] flat, int idx) {
        float sum = 0.0f;
        int base = idx * DIMS;
        for (int i = 0; i < DIMS; i++) {
            float diff = a[i] - flat[base + i];
            sum += diff * diff;
        }
        return sum;
    }

    @Override
    public byte[] getLabels() {
        return labels;
    }
}
