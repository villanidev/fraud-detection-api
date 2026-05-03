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

    private final float[][] vectors;
    private final byte[] labels;

    public BruteForceIndex(float[][] vectors, byte[] labels) {
        this.vectors = vectors;
        this.labels = labels;
    }

    @Override
    public int search(float[] query, int k, int[] neighbors, float[] distances) {
        System.out.println("BruteForceIndex: Performing exact K-NN search (O(N*D))");
        Arrays.fill(distances, Float.MAX_VALUE);
        Arrays.fill(neighbors, -1);

        for (int i = 0; i < vectors.length; i++) {
            float d = squaredDistance(query, vectors[i]);
            if (d < distances[k - 1]) {
                insertSorted(neighbors, distances, k, i, d);
            }
        }

        int fraudCount = 0;
        for (int i = 0; i < k; i++) {
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

    private static float squaredDistance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < 14; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    @Override
    public byte[] getLabels() {
        return labels;
    }
}
