package villani.dev.vectorsearch.index.strategies.ivfpq;

import villani.dev.vectorsearch.index.VectorIndex;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
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
    private static final int SCALAR_LUT_WIDTH = 256;
    private static final int FLAG_HAS_SENTINEL_DIM5 = 1;
    private static final int FLAG_HAS_CONTINUOUS_DIM5 = 1 << 1;
    private static final int FLAG_HAS_SENTINEL_DIM6 = 1 << 2;
    private static final int FLAG_HAS_CONTINUOUS_DIM6 = 1 << 3;

    private final float[] centroidsFlat;   // [K * DIMS] — flat for sequential access, no pointer chasing
    private final int K;
    private final int[] clusterSizes;
    private final int[] idsStartIndex;
    private final IntBuffer idsPayload;
    private final int[] codesStartIndex;
    private final ShortBuffer codesPayload;
    private final byte[] labels;           // [N] — 0=legit, 1=fraud
    private final ProductQuantizer pq;
    private final float[] clusterMinValues;
    private final float[] clusterMaxValues;
    private final int[] clusterFlags;
    private final ByteBuffer scalarQuantizedPayload;
    private final int scalarQuantizedStrideBytes;
    private final boolean scalarScanEnabled;
    private final boolean pruneEnabled;
    private final float pruneEpsilon;
    private final float pruneSentinelPenalty;
    private final int[] scalarPayloadStartByCluster;
    private final int nprobe;
    private final int candidates;

    // ThreadLocal scratch arrays — eliminates ~11KB of heap allocation per request.
    // centroidDist/centroidOrder are K=512 floats/ints; adcTable is M×256 floats.
    private final ThreadLocal<float[]>   tlCentroidDist;
    private final ThreadLocal<int[]>     tlCentroidOrder;
    private final ThreadLocal<float[]> tlAdcTable;
    private final ThreadLocal<float[]> tlScalarLut;

    public IVFPQIndex(float[][] centroids,
                      int[] clusterSizes,
                      int[] idsStartIndex,
                      IntBuffer idsPayload,
                      int[] codesStartIndex,
                      ShortBuffer codesPayload,
                      byte[] labels,
                      ProductQuantizer pq,
                      float[] clusterMinValues,
                      float[] clusterMaxValues,
                      int[] clusterFlags,
                      ByteBuffer scalarQuantizedPayload,
                      int scalarQuantizedStrideBytes,
                      boolean scalarScanEnabled,
                      boolean pruneEnabled,
                      float pruneEpsilon,
                      float pruneSentinelPenalty,
                      int nprobe,
                      int candidates) {
        System.out.println("Initializing IVFPQIndex with " + centroids.length + " centroids. And nprobe=" + nprobe + ", " +
                "candidates=" + candidates + ".");
        this.K = centroids.length;
        // Flatten [K][14] → float[K*14] to eliminate pointer chasing in the hot centroid scan loop
        this.centroidsFlat = new float[K * DIMS];
        for (int c = 0; c < K; c++)
            System.arraycopy(centroids[c], 0, centroidsFlat, c * DIMS, DIMS);
        this.clusterSizes = clusterSizes;
        this.idsStartIndex = idsStartIndex;
        this.idsPayload = idsPayload;
        this.codesStartIndex = codesStartIndex;
        this.codesPayload = codesPayload;
        this.labels = labels;
        this.pq = pq;
        this.clusterMinValues = clusterMinValues;
        this.clusterMaxValues = clusterMaxValues;
        this.clusterFlags = clusterFlags;
        this.scalarQuantizedPayload = scalarQuantizedPayload;
        this.scalarQuantizedStrideBytes = scalarQuantizedStrideBytes;
        this.scalarScanEnabled = scalarScanEnabled && scalarQuantizedPayload != null && scalarQuantizedStrideBytes > 0;
        this.pruneEnabled = pruneEnabled && clusterMinValues != null && clusterMaxValues != null && clusterFlags != null;
        this.pruneEpsilon = pruneEpsilon;
        this.pruneSentinelPenalty = pruneSentinelPenalty;
        this.scalarPayloadStartByCluster = new int[K];
        this.nprobe = nprobe;
        this.candidates = candidates;

        int scalarOffset = 0;
        for (int c = 0; c < K; c++) {
            this.scalarPayloadStartByCluster[c] = scalarOffset;
            scalarOffset += clusterSizes[c] * scalarQuantizedStrideBytes;
        }

        this.tlCentroidDist  = ThreadLocal.withInitial(() -> new float[K]);
        this.tlCentroidOrder = ThreadLocal.withInitial(() -> new int[K]);
        this.tlAdcTable      = ThreadLocal.withInitial(() -> new float[ProductQuantizer.M * ProductQuantizer.CODEBOOK_SIZE]);
        this.tlScalarLut     = ThreadLocal.withInitial(() -> new float[DIMS * SCALAR_LUT_WIDTH]);
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

        // --- Step 3: Scan every cluster ---
        Arrays.fill(distances, Float.MAX_VALUE);
        Arrays.fill(neighbors, -1);

        float[] adcTable = null;
        float[] scalarLut = null;
        if (scalarScanEnabled) {
            scalarLut = tlScalarLut.get();
            buildScalarLut(query, scalarLut);
        } else {
            adcTable = tlAdcTable.get();
            pq.buildAdcTableFlat(query, adcTable);
        }

        int actualProbes = Math.min(nprobeParam, K);
        for (int p = 0; p < actualProbes; p++) {
            int clusterIdx = centroidOrder[p];
            if (shouldPruneCluster(clusterIdx, query, distances, candidatesParam)) {
                continue;
            }
            int count = clusterSizes[clusterIdx];
            int idsBase = idsStartIndex[clusterIdx];
            int codesBase = codesStartIndex[clusterIdx];
            for (int i = 0; i < count; i++) {
                float approxDist = scalarScanEnabled
                        ? scalarDistance(clusterIdx, i, scalarLut)
                        : adcDistanceMapped(adcTable, codesBase + i * ProductQuantizer.M);
                if (approxDist < distances[Math.max(0, candidatesParam - 1)]) {
                    insertSorted(neighbors, distances, candidatesParam, idsPayload.get(idsBase + i), approxDist);
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

    private void buildScalarLut(float[] query, float[] lut) {
        for (int dim = 0; dim < DIMS; dim++) {
            int base = dim * SCALAR_LUT_WIDTH;
            if (dim == 9 || dim == 10 || dim == 11) {
                float distFalse = query[dim] * query[dim];
                float distTrue = (query[dim] - 1f) * (query[dim] - 1f);
                Arrays.fill(lut, base, base + SCALAR_LUT_WIDTH, distFalse);
                lut[base + 255] = distTrue;
                continue;
            }

            if (dim == 5 || dim == 6) {
                for (int code = 0; code < 255; code++) {
                    float decoded = code / 254.0f;
                    float diff = query[dim] - decoded;
                    lut[base + code] = diff * diff;
                }
                lut[base + 255] = query[dim] == -1f ? 0f : 4f;
                continue;
            }

            for (int code = 0; code < SCALAR_LUT_WIDTH; code++) {
                float decoded = code / 255.0f;
                float diff = query[dim] - decoded;
                lut[base + code] = diff * diff;
            }
        }
    }

    private float scalarDistance(int clusterIdx, int indexInCluster, float[] lut) {
        int offset = scalarPayloadStartByCluster[clusterIdx] + indexInCluster * scalarQuantizedStrideBytes;
        float distance = 0f;
        for (int dim = 0; dim < DIMS; dim++) {
            int code = scalarQuantizedPayload.get(offset + dim) & 0xFF;
            distance += lut[dim * SCALAR_LUT_WIDTH + code];
        }
        return distance;
    }

    private float adcDistanceMapped(float[] tableFlat, int codeOffset) {
        return tableFlat[0 * ProductQuantizer.CODEBOOK_SIZE + codesPayload.get(codeOffset)]
                + tableFlat[1 * ProductQuantizer.CODEBOOK_SIZE + codesPayload.get(codeOffset + 1)]
                + tableFlat[2 * ProductQuantizer.CODEBOOK_SIZE + codesPayload.get(codeOffset + 2)]
                + tableFlat[3 * ProductQuantizer.CODEBOOK_SIZE + codesPayload.get(codeOffset + 3)]
                + tableFlat[4 * ProductQuantizer.CODEBOOK_SIZE + codesPayload.get(codeOffset + 4)]
                + tableFlat[5 * ProductQuantizer.CODEBOOK_SIZE + codesPayload.get(codeOffset + 5)]
                + tableFlat[6 * ProductQuantizer.CODEBOOK_SIZE + codesPayload.get(codeOffset + 6)];
    }

    private boolean shouldPruneCluster(int clusterIdx, float[] query, float[] distances, int candidatesParam) {
        if (!pruneEnabled || candidatesParam <= 0) {
            return false;
        }

        float threshold = distances[candidatesParam - 1];
        if (!Float.isFinite(threshold)) {
            return false;
        }

        float lowerBound = clusterLowerBound(query, clusterIdx);
        return lowerBound > threshold + pruneEpsilon;
    }

    private float clusterLowerBound(float[] query, int clusterIdx) {
        int base = clusterIdx * DIMS;
        float lowerBound = 0f;
        for (int dim = 0; dim < DIMS; dim++) {
            if (dim == 5 || dim == 6) {
                lowerBound += sentinelAwareLowerBound(query[dim], clusterIdx, dim, base + dim);
                continue;
            }

            float min = clusterMinValues[base + dim];
            float max = clusterMaxValues[base + dim];
            if (query[dim] < min) {
                float diff = min - query[dim];
                lowerBound += diff * diff;
            } else if (query[dim] > max) {
                float diff = query[dim] - max;
                lowerBound += diff * diff;
            }
        }
        return lowerBound;
    }

    private float sentinelAwareLowerBound(float queryValue, int clusterIdx, int dim, int valueIndex) {
        int flags = clusterFlags[clusterIdx];
        boolean hasSentinel = dim == 5
                ? (flags & FLAG_HAS_SENTINEL_DIM5) != 0
                : (flags & FLAG_HAS_SENTINEL_DIM6) != 0;
        boolean hasContinuous = dim == 5
                ? (flags & FLAG_HAS_CONTINUOUS_DIM5) != 0
                : (flags & FLAG_HAS_CONTINUOUS_DIM6) != 0;

        float best = Float.POSITIVE_INFINITY;
        if (hasContinuous) {
            float min = clusterMinValues[valueIndex];
            float max = clusterMaxValues[valueIndex];
            if (queryValue < min) {
                float diff = min - queryValue;
                best = diff * diff;
            } else if (queryValue > max) {
                float diff = queryValue - max;
                best = diff * diff;
            } else {
                best = 0f;
            }
        }

        if (hasSentinel) {
            float sentinelDiff = queryValue + 1f;
            float sentinelBound = sentinelDiff * sentinelDiff;
            if (sentinelBound > pruneSentinelPenalty) {
                sentinelBound = pruneSentinelPenalty;
            }
            if (sentinelBound < best) {
                best = sentinelBound;
            }
        }

        return best == Float.POSITIVE_INFINITY ? 0f : best;
    }

    @Override
    public byte[] getLabels() {
        return labels;
    }
}
