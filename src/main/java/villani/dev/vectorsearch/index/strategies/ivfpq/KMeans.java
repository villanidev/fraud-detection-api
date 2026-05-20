package villani.dev.vectorsearch.index.strategies.ivfpq;

import java.util.Random;
import java.util.stream.IntStream;

/**
 * K-Means clustering with K-Means++ initialization.
 * Plain Java — no DI annotations; created by ProductQuantizer and VectorIndexFactory.
 *
 * Complexity:
 *   initPlusPlus  — O(N×K×D): incremental distance update per new centroid (not O(N×K²×D))
 *   cluster loop  — O(iter × N×K×D): assignment parallelized across CPU cores
 *   assign        — O(N×K×D): single pass stores assignments, second pass fills lists
 */
public class KMeans {

    private static final int ITERATIONS = 25;

    /**
     * Clusters vectors into K centroids using K-Means++ initialization.
     *
     * @param vectors input vectors (N x D)
     * @param K       number of clusters
     * @param seed    random seed for reproducibility
     * @return K centroids (K x D)
     */
    public float[][] cluster(float[][] vectors, int K, long seed) {
        int N = vectors.length;
        int D = vectors[0].length;
        Random rng = new Random(seed);

        float[][] centroids = initPlusPlus(vectors, K, D, rng);
        int[] assignments = new int[N];

        for (int iter = 0; iter < ITERATIONS; iter++) {
            assignParallel(vectors, centroids, assignments, N, D);
            updateCentroids(vectors, centroids, assignments, N, K, D);
        }

        return centroids;
    }

    /**
     * Clusters vectors into K centroids for sub-vectors of dimension subD.
     * Used by ProductQuantizer per subspace.
     */
    public float[][] clusterSub(float[][] subVectors, int K, long seed) {
        int N = subVectors.length;
        int D = subVectors[0].length;
        Random rng = new Random(seed);

        float[][] centroids = initPlusPlus(subVectors, K, D, rng);
        int[] assignments = new int[N];

        for (int iter = 0; iter < ITERATIONS; iter++) {
            assignParallel(subVectors, centroids, assignments, N, D);
            updateCentroids(subVectors, centroids, assignments, N, K, D);
        }

        return centroids;
    }

    /**
     * Assigns each vector to its nearest centroid, returning inverted lists.
     *
     * @return inverted lists: idsByCluster[c] = array of vector IDs in cluster c
     */
    public int[][] assign(float[][] vectors, float[][] centroids) {
        int N = vectors.length;
        int K = centroids.length;
        int D = centroids[0].length;

        // Single pass: assign and count simultaneously
        int[] allAssignments = new int[N];
        int[] sizes = new int[K];
        for (int i = 0; i < N; i++) {
            int c = nearestIdx(vectors[i], centroids, D);
            allAssignments[i] = c;
            sizes[c]++;
        }

        int[][] invertedLists = new int[K][];
        for (int c = 0; c < K; c++) invertedLists[c] = new int[sizes[c]];

        int[] pos = new int[K];
        for (int i = 0; i < N; i++) {
            int c = allAssignments[i];
            invertedLists[c][pos[c]++] = i;
        }

        return invertedLists;
    }

    /** Returns index of the nearest centroid to vec (full D dimensions). */
    public int nearest(float[] vec, float[][] centroids) {
        return nearestIdx(vec, centroids, vec.length);
    }

    /** Returns index of the nearest centroid to a sub-vector of given dimension. */
    public int nearestSub(float[] sub, float[][] centroids, int D) {
        return nearestIdx(sub, centroids, D);
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * K-Means++ init — O(N×K×D).
     *
     * Maintains distances[i] = min dist from vector[i] to any chosen centroid.
     * After choosing centroid[k], only compares each vector against centroid[k]
     * (not all k previous centroids), updating distances[i] if improved.
     * This reduces complexity from O(N×K²×D) to O(N×K×D).
     */
    private static float[][] initPlusPlus(float[][] vectors, int K, int D, Random rng) {
        int N = vectors.length;
        float[][] centroids = new float[K][D];
        float[] distances = new float[N];

        // First centroid: uniformly random
        int first = rng.nextInt(N);
        System.arraycopy(vectors[first], 0, centroids[0], 0, D);

        // Initialize distances against the first centroid — O(N×D)
        for (int i = 0; i < N; i++) {
            distances[i] = squaredDist(vectors[i], centroids[0], D);
        }

        for (int k = 1; k < K; k++) {
            // Sample next centroid proportional to D² distance
            double total = 0;
            for (float d : distances) total += d;

            double threshold = rng.nextDouble() * total;
            double cumulative = 0;
            int chosen = N - 1;
            for (int i = 0; i < N; i++) {
                cumulative += distances[i];
                if (cumulative >= threshold) { chosen = i; break; }
            }
            System.arraycopy(vectors[chosen], 0, centroids[k], 0, D);

            // Incremental update: only compare against the newly added centroid — O(N×D)
            for (int i = 0; i < N; i++) {
                float d = squaredDist(vectors[i], centroids[k], D);
                if (d < distances[i]) distances[i] = d;
            }
        }

        return centroids;
    }

    /**
     * Parallel assignment — each vector is independent, scales with CPU cores.
     * Uses parallel IntStream; assignments[] is written at disjoint indices.
     */
    private static void assignParallel(float[][] vectors, float[][] centroids,
                                       int[] assignments, int N, int D) {
        IntStream.range(0, N).parallel()
                .forEach(i -> assignments[i] = nearestIdx(vectors[i], centroids, D));
    }

    /**
     * Centroid update — O(N×D + K×D).
     * Uses multiply-by-inverse instead of per-element division.
     * Empty clusters keep their previous centroid (no reinit needed for preprocessing).
     */
    private static void updateCentroids(float[][] vectors, float[][] centroids,
                                        int[] assignments, int N, int K, int D) {
        float[][] acc = new float[K][D];
        int[] counts = new int[K];

        for (int i = 0; i < N; i++) {
            int c = assignments[i];
            counts[c]++;
            float[] v = vectors[i];
            float[] a = acc[c];
            for (int d = 0; d < D; d++) a[d] += v[d];
        }

        for (int c = 0; c < K; c++) {
            if (counts[c] > 0) {
                float inv = 1f / counts[c];
                float[] a = acc[c];
                float[] cen = centroids[c];
                for (int d = 0; d < D; d++) cen[d] = a[d] * inv;
            }
            // Empty cluster: keep previous centroid unchanged
        }
    }

    /** O(K×D) nearest centroid search with early-exit potential via accumulating partial dist. */
    private static int nearestIdx(float[] vec, float[][] centroids, int D) {
        int best = 0;
        float bestDist = Float.MAX_VALUE;
        for (int c = 0; c < centroids.length; c++) {
            float dist = squaredDist(vec, centroids[c], D);
            if (dist < bestDist) { bestDist = dist; best = c; }
        }
        return best;
    }

    private static float squaredDist(float[] a, float[] b, int D) {
        float sum = 0;
        for (int i = 0; i < D; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}
