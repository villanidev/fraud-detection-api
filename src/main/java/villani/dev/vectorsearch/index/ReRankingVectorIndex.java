package villani.dev.vectorsearch.index;

import java.nio.ByteOrder;
import java.util.Arrays;
import villani.dev.vectorsearch.index.strategies.ivfpq.IVFPQIndex;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Decorator that adds exact-distance reranking on top of any VectorIndex.
 *
 * Two-stage search:
 *   Stage 1 — delegate to inner index for `candidates` approximate results.
 *   Stage 2 — rerank those candidates with exact squared Euclidean distance
 *              read directly from the memory-mapped vectors section of data.bin.
 *              Only ~50 × 56 bytes are paged in per query — zero heap cost.
 *
 * Thread-safety: MappedByteBuffer.position() is not thread-safe. Each thread
 * gets its own duplicate via ThreadLocal to avoid race conditions under concurrency.
 *
 * Pattern: Decorator (GoF) — wraps any VectorIndex, adds behaviour without
 *           modifying the wrapped class (OCP).
 */
public class ReRankingVectorIndex implements VectorIndex {

    private static final int DIMS = 14;
    private static final int VECTOR_BYTES = DIMS * Float.BYTES; // 56 bytes

    private final VectorIndex inner;
    private final FileChannel vectorsChannel;  // mantido aberto, thread‑safe para read(pos)
    private final long vectorsOffset;
    private final int vectorCount;
    private final int candidates;
    private int rerankNprobe;
    private final int rerankCandidates;

    // Scratch buffers usados no hot path da busca (apenas cálculo, sem I/O)
    private final ThreadLocal<int[]>   tlCoarseNeighbors;
    private final ThreadLocal<float[]> tlCoarseDists;
    private final ThreadLocal<float[]> tlExactDists;
    private final ThreadLocal<float[]> tlVec; // buffer para vetor lido
    //buffer off-heap reutilizável por thread (56 bytes)
    private final ThreadLocal<ByteBuffer> tlReadBuffer = ThreadLocal.withInitial(() -> {
        ByteBuffer buf = ByteBuffer.allocateDirect(VECTOR_BYTES);
        buf.order(ByteOrder.BIG_ENDIAN);
        return buf;
    });

    public ReRankingVectorIndex(VectorIndex inner,
                                FileChannel vectorsChannel,
                                long vectorsOffset,
                                int vectorCount,
                                int rerankNprobe,
                                int rerankCandidates) {
        System.out.println("Initializing ReRankingVectorIndex with rerankNprobe=" + rerankNprobe + " and rerankCandidates=" + rerankCandidates);
        this.inner = inner;
        this.vectorsChannel = vectorsChannel;
        this.vectorsOffset = vectorsOffset;
        this.vectorCount = vectorCount;
        this.candidates = rerankCandidates; // keep per-instance candidate scratch sized to rerank window
        this.rerankNprobe = rerankNprobe;
        this.rerankCandidates = rerankCandidates;
        this.tlCoarseNeighbors = ThreadLocal.withInitial(() -> new int[candidates]);
        this.tlCoarseDists     = ThreadLocal.withInitial(() -> new float[candidates]);
        this.tlExactDists      = ThreadLocal.withInitial(() -> new float[candidates]);
        this.tlVec             = ThreadLocal.withInitial(() -> new float[DIMS]);
    }

    @Override
    public int search(float[] query, int topK, int[] neighbors, float[] distances) {
        int[] coarseNeighbors = tlCoarseNeighbors.get();
        float[] coarseDists   = tlCoarseDists.get();

        int firstCheck = inner.search(query, candidates, coarseNeighbors, coarseDists);

        if (firstCheck == 0) return 0;

        float[] exactDists = tlExactDists.get();
        computeExactDistancesForCandidates(query, coarseNeighbors, candidates, exactDists);
        mergeExactIntoTopK(coarseNeighbors, candidates, exactDists, topK, neighbors, distances);

        int fraudCount = computeFraudCount(coarseNeighbors, topK);

        if (fraudCount == 2 || fraudCount == 3) {
            //System.out.println("Grey zone, count: " + fraudCount);
            // Re-run coarse search with larger probe/candidate settings
            int[] newCoarseNeighbors = new int[rerankCandidates];
            float[] newCoarseDists = new float[rerankCandidates];

            if (inner instanceof IVFPQIndex ivf) {
                ivf.searchWithParams(query, rerankCandidates, rerankNprobe, rerankCandidates, newCoarseNeighbors, newCoarseDists);
            } else {
                inner.search(query, rerankCandidates, newCoarseNeighbors, newCoarseDists);
            }

            // Recompute exact distances for the expanded candidate set and merge
            float[] exactDists2 = new float[newCoarseNeighbors.length];
            computeExactDistancesForCandidates(query, newCoarseNeighbors, newCoarseNeighbors.length, exactDists2);
            mergeExactIntoTopK(newCoarseNeighbors, newCoarseNeighbors.length, exactDists2, topK, neighbors, distances);

            // Recompute fraud count after expanded rerank
            fraudCount = computeFraudCount(neighbors, topK);
        }

        return fraudCount;
    }

    private void computeExactDistancesForCandidates(float[] query, int[] candidateIds, int candidateCount, float[] outExactDists) {
        float[] vec = tlVec.get();
        for (int i = 0; i < candidateCount; i++) {
            int id = candidateIds[i];
            if (id < 0) {
                outExactDists[i] = Float.MAX_VALUE;
                continue;
            }
            try {
                float[] readVec = readVector(id);
                System.arraycopy(readVec, 0, vec, 0, DIMS);
                outExactDists[i] = squaredDistance(query, vec);
            } catch (Exception e) {
                outExactDists[i] = Float.MAX_VALUE;
            }
        }
    }

    private void mergeExactIntoTopK(int[] candidateIds, int candidateCount, float[] exactDists, int topK, int[] neighbors, float[] distances) {
        Arrays.fill(distances, Float.MAX_VALUE);
        Arrays.fill(neighbors, -1);
        for (int i = 0; i < candidateCount; i++) {
            if (candidateIds[i] >= 0 && exactDists[i] < distances[topK - 1]) {
                insertSorted(neighbors, distances, topK, candidateIds[i], exactDists[i]);
            }
        }
    }

    private int computeFraudCount(int[] neighbors, int topK) {
        byte[] labels = inner.getLabels();
        int fraud = 0;
        for (int i = 0; i < topK; i++) {
            if (neighbors[i] >= 0 && labels[neighbors[i]] == 1) fraud++;
        }
        return fraud;
    }

    private float[] readVector(int id) throws Exception {
        long offset = vectorsOffset + (long) id * VECTOR_BYTES;
        ByteBuffer buf = tlReadBuffer.get();  // reutiliza o buffer da thread
        buf.order(ByteOrder.BIG_ENDIAN);
        int bytesRead = 0;
        while (buf.hasRemaining()) {
            int n = vectorsChannel.read(buf, offset + bytesRead);
            if (n == -1) throw new IOException("EOF");
            bytesRead += n;
        }
        buf.flip();
        float[] result = new float[DIMS];
        buf.asFloatBuffer().get(result);
        return result;
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
        float d0 = a[0] - b[0];
        float d1 = a[1] - b[1];
        float d2 = a[2] - b[2];
        float d3 = a[3] - b[3];
        float d4 = a[4] - b[4];
        float d5 = a[5] - b[5];
        float d6 = a[6] - b[6];
        float d7 = a[7] - b[7];
        float d8 = a[8] - b[8];
        float d9 = a[9] - b[9];
        float d10 = a[10] - b[10];
        float d11 = a[11] - b[11];
        float d12 = a[12] - b[12];
        float d13 = a[13] - b[13];
        return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
                d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13;
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
