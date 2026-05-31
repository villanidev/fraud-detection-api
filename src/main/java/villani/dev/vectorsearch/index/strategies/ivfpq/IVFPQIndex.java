package villani.dev.vectorsearch.index.strategies.ivfpq;

import villani.dev.vectorsearch.index.VectorIndex;

import java.util.Arrays;

/**
 * Approximate Nearest Neighbor search via IVF (Inverted File Index) + PQ (Product Quantization).
 * Search pipeline per query:
 *   1. Find the nprobe nearest IVF centroids to the query (coarse quantization).
 *   2. Precompute the ADC table: table[m][c] = dist²(query_sub_m, codebook[m][c]).
 *   3. Scan the first nprobeGray clusters. If fraudCount ∈ {0,1,4,5} → early exit (clear decision).
 *   4. If gray zone (fraudCount ∈ {2,3}) → continue scanning up to nprobe clusters.
 *   5. Return fraud count in top-k (caller decides k ≤ candidates).
 * Plain Java — no DI annotations; created by VectorIndexFactory.
 */
public class IVFPQIndex implements VectorIndex {

    private static final int DIMS = 14;

    private final float[] centroidsFlat;   // [K * DIMS] — flat for sequential access, no pointer chasing
    private final int K;
    private final int[][] idsByCluster;    // [K][count]
    private final short[][] codesByCluster; // [K][count*M] flat — avoids 3M short[7] object headers
    private final byte[] labels;           // [N] — 0=legit, 1=fraud
    private final ProductQuantizer pq;
    private final int nprobe;
    private final int candidates;

    // ThreadLocal scratch arrays — eliminates ~11KB of heap allocation per request.
    // centroidDist/centroidOrder are K=512 floats/ints; adcTable is M×256 floats.
    private final ThreadLocal<float[]>   tlCentroidDist;
    private final ThreadLocal<int[]>     tlCentroidOrder;
    private final ThreadLocal<float[]> tlAdcTable;

    public IVFPQIndex(float[][] centroids,
                      int[][] idsByCluster,
                      short[][] codesByCluster,
                      byte[] labels,
                      ProductQuantizer pq,
                      int nprobe,
                      int candidates) {
        System.out.println("Initializing IVFPQIndex with " + centroids.length + " centroids. And nprobe=" + nprobe + ", " +
                "candidates=" + candidates + ".");
        this.K = centroids.length;
        // Flatten [K][14] → float[K*14] to eliminate pointer chasing in the hot centroid scan loop
        this.centroidsFlat = new float[K * DIMS];
        for (int c = 0; c < K; c++)
            System.arraycopy(centroids[c], 0, centroidsFlat, c * DIMS, DIMS);
        this.idsByCluster = idsByCluster;
        this.codesByCluster = codesByCluster;
        this.labels = labels;
        this.pq = pq;
        this.nprobe = nprobe;
        this.candidates = candidates;

        this.tlCentroidDist  = ThreadLocal.withInitial(() -> new float[K]);
        this.tlCentroidOrder = ThreadLocal.withInitial(() -> new int[K]);
        this.tlAdcTable      = ThreadLocal.withInitial(() -> new float[ProductQuantizer.M * ProductQuantizer.CODEBOOK_SIZE]);
    }

    @Override
    public int search(float[] query, int topK, int[] neighbors, float[] distances) {
        return searchWithParams(query, topK, this.nprobe, this.candidates, neighbors, distances);
    }

    /**
     * Search variant that allows caller to override `nprobe` and `candidates` per-call.
     * This avoids creating new index instances when tuning search parameters at runtime.
     */
    public int searchWithParams(float[] query, int topK, int nprobeParam, int candidatesParam, int[] neighbors, float[] distances) {
        // --- Step 1: Find nprobe nearest centroids (reuse ThreadLocal scratch arrays) ---
        float[] centroidDist  = tlCentroidDist.get();
        int[] centroidOrder = tlCentroidOrder.get();
        for (int c = 0; c < K; c++) {
            centroidDist[c]  = squaredDistance(query, centroidsFlat, c * DIMS);
            centroidOrder[c] = c;
        }
        partialSort(centroidOrder, centroidDist, Math.min(nprobeParam, K));

        // --- Step 2: Precompute ADC table (reuse ThreadLocal scratch array) ---
        float[] adcTable = tlAdcTable.get();
        pq.buildAdcTableFlat(query, adcTable);

        // --- Step 3: Scan every cluster ---
        Arrays.fill(distances, Float.MAX_VALUE);
        Arrays.fill(neighbors, -1);

        int actualProbes = Math.min(nprobeParam, K);
        for (int p = 0; p < actualProbes; p++) {
            int clusterIdx = centroidOrder[p];
            int[] ids = idsByCluster[clusterIdx];
            short[] codes = codesByCluster[clusterIdx]; // flat: codes for vector i at offset i*M

            for (int i = 0; i < ids.length; i++) {
                float approxDist = pq.adcDistanceFlat(adcTable, codes, i * ProductQuantizer.M);
                if (approxDist < distances[Math.max(0, candidatesParam - 1)]) {
                    insertSorted(neighbors, distances, candidatesParam, ids[i], approxDist);
                }
            }
        }

        // --- Step 4: Count fraud labels in top-k ---
        int fraudCount = 0;
        for (int i = 0; i < topK; i++) {
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
        // Ordena os top elementos para manter o determinismo
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
        while (left < right) {
            int pivotIdx = medianOfThreePartition(order, dist, left, right);
            int rank = pivotIdx - left + 1;
            if (rank == k) return;
            if (k < rank) {
                right = pivotIdx - 1;
            } else {
                left = pivotIdx + 1;
                k -= rank;
            }
        }
    }

    private static int medianOfThreePartition(int[] order, float[] dist, int left, int right) {
        int mid = (left + right) >>> 1;
        // Coloca a mediana dos três (left, mid, right) na posição right
        if (dist[order[left]] > dist[order[mid]]) swap(order, left, mid);
        if (dist[order[left]] > dist[order[right]]) swap(order, left, right);
        if (dist[order[mid]] > dist[order[right]]) swap(order, mid, right);
        // O pivô agora está em order[right] (o maior dos três medianos)
        float pivotDist = dist[order[right]];
        int i = left - 1;
        for (int j = left; j < right; j++) {
            if (dist[order[j]] <= pivotDist) {
                i++;
                swap(order, i, j);
            }
        }
        swap(order, i + 1, right);
        return i + 1;
    }

    private static void swap(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    private static float squaredDistance(float[] query, float[] flat, int offset) {
        float d0 = query[0] - flat[offset];
        float d1 = query[1] - flat[offset + 1];
        float d2 = query[2] - flat[offset + 2];
        float d3 = query[3] - flat[offset + 3];
        float d4 = query[4] - flat[offset + 4];
        float d5 = query[5] - flat[offset + 5];
        float d6 = query[6] - flat[offset + 6];
        float d7 = query[7] - flat[offset + 7];
        float d8 = query[8] - flat[offset + 8];
        float d9 = query[9] - flat[offset + 9];
        float d10 = query[10] - flat[offset + 10];
        float d11 = query[11] - flat[offset + 11];
        float d12 = query[12] - flat[offset + 12];
        float d13 = query[13] - flat[offset + 13];
        return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
                d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13;
    }

    @Override
    public byte[] getLabels() {
        return labels;
    }
}
