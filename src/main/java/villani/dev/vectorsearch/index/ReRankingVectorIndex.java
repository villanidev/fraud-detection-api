package villani.dev.vectorsearch.index;

import java.nio.ByteOrder;
import java.util.Arrays;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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

    // Thread pool dedicado para leitura de arquivos (evita pinning)
    /*private static final ExecutorService ioPool = Executors.newFixedThreadPool(4,
            r -> {
                Thread t = new Thread(r, "vector-io");
                t.setDaemon(true);
                return t;
            });*/

    // Scratch buffers usados no hot path da busca (apenas cálculo, sem I/O)
    private final ThreadLocal<int[]>   tlCoarseNeighbors;
    private final ThreadLocal<float[]> tlCoarseDists;
    private final ThreadLocal<float[]> tlExactDists;
    private final ThreadLocal<float[]> tlVec; // buffer para vetor lido

    public ReRankingVectorIndex(VectorIndex inner,
                                FileChannel vectorsChannel,
                                long vectorsOffset,
                                int vectorCount,
                                int candidates) {
        this.inner = inner;
        this.vectorsChannel = vectorsChannel;
        this.vectorsOffset = vectorsOffset;
        this.vectorCount = vectorCount;
        this.candidates = candidates;
        this.tlCoarseNeighbors = ThreadLocal.withInitial(() -> new int[candidates]);
        this.tlCoarseDists     = ThreadLocal.withInitial(() -> new float[candidates]);
        this.tlExactDists      = ThreadLocal.withInitial(() -> new float[candidates]);
        this.tlVec             = ThreadLocal.withInitial(() -> new float[DIMS]);
    }

    @Override
    public int search(float[] query, int k, int[] neighbors, float[] distances) {
        int[] coarseNeighbors = tlCoarseNeighbors.get();
        float[] coarseDists   = tlCoarseDists.get();
        inner.search(query, candidates, coarseNeighbors, coarseDists);

        float[] exactDists = tlExactDists.get();
        float[] vec = tlVec.get();  // será reutilizado para cada vetor lido

        // Para cada candidato, lê o vetor via I/O assíncrono (descarregado em pool)
        // Isso pode ser feito em paralelo, mas para simplicidade e baixo paralelismo (10 a 50 candidatos)
        // faremos sequencial com Future, mantendo a semântica original.
        for (int i = 0; i < candidates; i++) {
            int id = coarseNeighbors[i];
            if (id < 0) {
                exactDists[i] = Float.MAX_VALUE;
                continue;
            }
            try {
                //readVectorAsync(id, vec);  // bloqueia a virtual thread sem pinning
                float[] readVec = readVector(id); // chamada direta, sem pool
                System.arraycopy(readVec, 0, vec, 0, DIMS);
            } catch (Exception e) {
                exactDists[i] = Float.MAX_VALUE;
                continue;
            }
            exactDists[i] = squaredDistance(query, vec);
        }

        // Merge top‑k
        Arrays.fill(distances, Float.MAX_VALUE);
        Arrays.fill(neighbors, -1);
        for (int i = 0; i < candidates; i++) {
            if (coarseNeighbors[i] >= 0 && exactDists[i] < distances[k - 1]) {
                insertSorted(neighbors, distances, k, coarseNeighbors[i], exactDists[i]);
            }
        }

        byte[] labels = inner.getLabels();
        int fraudCount = 0;
        for (int i = 0; i < k; i++) {
            if (neighbors[i] >= 0 && labels[neighbors[i]] == 1) fraudCount++;
        }
        return fraudCount;
    }

    /** Lê o vetor `id` do disco usando o pool de I/O, bloqueando a thread virtual sem pinning. */
    /*private void readVectorAsync(int id, float[] out) throws Exception {
        long offset = vectorsOffset + (long) id * VECTOR_BYTES;
        Future<float[]> future = ioPool.submit(() -> {
            ByteBuffer buf = ByteBuffer.allocateDirect(VECTOR_BYTES);
            buf.order(ByteOrder.BIG_ENDIAN);
            // read(pos) é thread‑safe e não altera a posição do canal
            int bytesRead = 0;
            while (buf.hasRemaining()) {
                int n = vectorsChannel.read(buf, offset + bytesRead);
                if (n == -1) throw new IOException("EOF reached before reading full vector");
                bytesRead += n;
            }
            buf.flip();
            float[] result = new float[DIMS];
            buf.asFloatBuffer().get(result);
            return result;
        });

        // future.get() suspende a virtual thread sem prender a carrier
        float[] read = future.get();
        System.arraycopy(read, 0, out, 0, DIMS);
    }*/

    private float[] readVector(int id) throws Exception {
        long offset = vectorsOffset + (long) id * VECTOR_BYTES;
        ByteBuffer buf = ByteBuffer.allocateDirect(VECTOR_BYTES);
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
        float sum = 0.0f;
        for (int i = 0; i < DIMS; i++) {
            float diff = a[i] - b[i];
            sum = Math.fma(diff, diff, sum);
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
