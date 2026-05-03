package villani.dev.vectorsearch.index.strategies.ivfpq;

import villani.dev.vectorsearch.index.VectorIndex;

import java.util.Arrays;

/**
 * Approximate Nearest Neighbor search via IVF (Inverted File Index) + PQ (Product Quantization).
 *
 * Search pipeline per query:
 *   1. Find the nprobe nearest IVF centroids to the query (coarse quantization).
 *   2. Precompute the ADC table: table[m][c] = dist²(query_sub_m, codebook[m][c]).
 *   3. Scan the first nprobeGray clusters. If fraudCount ∈ {0,1,4,5} → early exit (clear decision).
 *   4. If gray zone (fraudCount ∈ {2,3}) → continue scanning up to nprobe clusters.
 *   5. Return fraud count in top-k (caller decides k ≤ candidates).
 *
 * Plain Java — no DI annotations; created by VectorIndexFactory.
 */
public class IVFPQIndex implements VectorIndex {

    private final float[][] centroids;     // [K][14]
    private final int[][] idsByCluster;    // [K][count]
    private final byte[][] codesByCluster; // [K][count*M] flat — avoids 3M byte[7] object headers
    private final byte[] labels;           // [N] — 0=legit, 1=fraud
    private final ProductQuantizer pq;
    private final int nprobe;
    private final int nprobeGray;  // fast-path probe count; early exit if decision is clear
    private final int candidates;

    // ThreadLocal scratch arrays — eliminates ~11KB of heap allocation per request.
    // centroidDist/centroidOrder are K=512 floats/ints; adcTable is M×256 floats.
    private final ThreadLocal<float[]>   tlCentroidDist;
    private final ThreadLocal<int[]>     tlCentroidOrder;
    private final ThreadLocal<float[][]> tlAdcTable;

    public IVFPQIndex(float[][] centroids,
                      int[][] idsByCluster,
                      byte[][] codesByCluster,
                      byte[] labels,
                      ProductQuantizer pq,
                      int nprobe,
                      int nprobeGray,
                      int candidates) {
        this.centroids = centroids;
        this.idsByCluster = idsByCluster;
        this.codesByCluster = codesByCluster;
        this.labels = labels;
        this.pq = pq;
        this.nprobe = nprobe;
        this.nprobeGray = nprobeGray;
        this.candidates = candidates;

        int K = centroids.length;
        this.tlCentroidDist  = ThreadLocal.withInitial(() -> new float[K]);
        this.tlCentroidOrder = ThreadLocal.withInitial(() -> new int[K]);
        this.tlAdcTable      = ThreadLocal.withInitial(() -> new float[ProductQuantizer.M][ProductQuantizer.CODEBOOK_SIZE]);
    }

    @Override
    public int search(float[] query, int k, int[] neighbors, float[] distances) {
        int K = centroids.length;
        int actualProbes = Math.min(nprobe, K);
        int actualCandidates = Math.min(candidates, k);

        // --- Step 1: Find nprobe nearest centroids (reuse ThreadLocal scratch arrays) ---
        float[]   centroidDist  = tlCentroidDist.get();
        int[]     centroidOrder = tlCentroidOrder.get();
        for (int c = 0; c < K; c++) {
            centroidDist[c]  = squaredDistance(query, centroids[c]);
            centroidOrder[c] = c;
        }
        partialSort(centroidOrder, centroidDist, actualProbes);

        // --- Step 2: Precompute ADC table (reuse ThreadLocal scratch array) ---
        float[][] adcTable = tlAdcTable.get();
        pq.buildAdcTable(query, adcTable);

        // --- Step 3+4: Scan clusters — early exit after nprobeGray if decision is clear ---
        Arrays.fill(distances, Float.MAX_VALUE);
        Arrays.fill(neighbors, -1);

        int actualGray = Math.min(nprobeGray, actualProbes);
        for (int p = 0; p < actualProbes; p++) {
            int clusterIdx = centroidOrder[p];
            int[] ids = idsByCluster[clusterIdx];
            byte[] codes = codesByCluster[clusterIdx]; // flat: codes for vector i at offset i*M

            for (int i = 0; i < ids.length; i++) {
                float approxDist = pq.adcDistance(adcTable, codes, i * ProductQuantizer.M);
                if (approxDist < distances[actualCandidates - 1]) {
                    insertSorted(neighbors, distances, actualCandidates, ids[i], approxDist);
                }
            }

            // After fast-path probes: count fraud in top-k and exit if decision is unambiguous
            if (p == actualGray - 1) {
                int fc = 0;
                for (int i = 0; i < k; i++) {
                    if (neighbors[i] >= 0 && labels[neighbors[i]] == 1) fc++;
                }
                if (fc <= 1 || fc >= k - 1) return fc;  // clear: 0/1 legit or 4/5 fraud
                // gray zone (fc==2 or fc==3) — continue full scan
            }
        }

        // --- Step 5: Count fraud labels in top-k ---
        int fraudCount = 0;
        for (int i = 0; i < k; i++) {
            if (neighbors[i] >= 0 && labels[neighbors[i]] == 1) fraudCount++;
        }
        return fraudCount;
    }

    /**
     * Inserts (id, dist) into sorted top-k arrays, maintaining ascending order.
     * Linear shift — optimal for small k (≤50).
     */
    static void insertSorted(int[] neighbors, float[] distances, int k, int id, float dist) {
        int pos = k - 1;
        while (pos > 0 && distances[pos - 1] > dist) {
            distances[pos] = distances[pos - 1];
            neighbors[pos] = neighbors[pos - 1];
            pos--;
        }
        distances[pos] = dist;
        neighbors[pos] = id;
    }

    /**
     * Partial sort: rearranges centroidOrder so the first 'top' elements are the
     * indices of the 'top' smallest values in centroidDist. Uses quickselect O(K).
     */
    private static void partialSort(int[] order, float[] dist, int top) {
        quickselect(order, dist, 0, order.length - 1, top);
        // Sort just the top slice for deterministic probe order
        for (int i = 1; i < top; i++) {
            int oi = order[i];
            float di = dist[oi];
            int j = i - 1;
            while (j >= 0 && dist[order[j]] > di) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = oi;
        }
    }

    private static void quickselect(int[] order, float[] dist, int left, int right, int k) {
        if (left >= right) return;
        int pivotIdx = partition(order, dist, left, right);
        int rank = pivotIdx - left + 1;
        if (rank == k) return;
        if (k < rank) quickselect(order, dist, left, pivotIdx - 1, k);
        else quickselect(order, dist, pivotIdx + 1, right, k - rank);
    }

    private static int partition(int[] order, float[] dist, int left, int right) {
        float pivotDist = dist[order[right]];
        int i = left - 1;
        for (int j = left; j < right; j++) {
            if (dist[order[j]] <= pivotDist) {
                i++;
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
        }
        int tmp = order[i + 1]; order[i + 1] = order[right]; order[right] = tmp;
        return i + 1;
    }

    private static float squaredDistance(float[] a, float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < 14; i++) {
            float diff = a[i] - b[i];
            sum = Math.fma(diff, diff, sum);
        }
        return sum;
    }

    @Override
    public byte[] getLabels() {
        return labels;
    }
}
