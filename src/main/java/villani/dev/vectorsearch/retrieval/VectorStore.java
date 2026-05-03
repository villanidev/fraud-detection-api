package villani.dev.vectorsearch.retrieval;

import io.helidon.service.registry.Service;
import villani.dev.vectorsearch.index.VectorIndex;
import villani.dev.vectorsearch.index.VectorIndexFactory;
import villani.dev.vectorsearch.index.strategies.ivfpq.KMeans;
import villani.dev.vectorsearch.index.strategies.ivfpq.ProductQuantizer;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Facade for all vector data and search.
 * Loads data.bin at startup, exposes normalization/MCC constants for embedding,
 * and delegates search to the configured VectorIndex strategy.
 *
 * data.bin binary layout (all multi-byte values in native byte order):
 *
 *   [ Header — 36 bytes ]
 *     int magic           = 0x52494E44
 *     int version         = 1
 *     int K               — number of IVF clusters
 *     int N               — number of vectors
 *     long vectorsOffset  — byte offset of the vectors section
 *
 *   [ Normalization — 7 floats = 28 bytes ]
 *   [ MCC table — 10000 floats = 40000 bytes ]
 *   [ PQ codebooks — M*256*SUB_D floats = 14336 bytes ]
 *   [ IVF centroids — K*14 floats ]
 *   [ Original vectors — N*14 floats ]  ← vectorsOffset points here
 *   [ Labels — N bytes ]
 *   [ Inverted lists — per cluster: int count, then count*(int id + 7 bytes PQ code) ]
 */
@Service.Singleton
public class VectorStore {

    public static final int MAGIC = 0x52494E44;
    public static final int VERSION = 1;
    static final int DIMS = 14;
    public static final int MCC_TABLE_SIZE = 10_000;
    public static final int NORM_COUNT = 7;

    private final VectorIndexFactory factory;

    // Data populated by load() — volatile for safe publication after single-writer startup
    private volatile VectorIndex index;
    private volatile float[] norms;
    private volatile float[] mccRisk;

    @Service.Inject
    public VectorStore(VectorIndexFactory factory) {
        this.factory = factory;
    }

    /**
     * Reads data.bin, deserializes all sections, and instantiates the configured VectorIndex.
     * Must be called from Main before the server starts accepting requests.
     */
    public void load(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            MappedByteBuffer buf = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            buf.order(ByteOrder.BIG_ENDIAN);

            // --- Header ---
            int magic = buf.getInt();
            if (magic != MAGIC) throw new IOException("Invalid data.bin: wrong magic 0x" + Integer.toHexString(magic));
            int version = buf.getInt();
            if (version != VERSION) throw new IOException("Unsupported data.bin version: " + version);
            int K = buf.getInt();
            int N = buf.getInt();
            long vectorsOffset = buf.getLong();

            // --- Normalization (7 floats) ---
            float[] loadedNorms = new float[NORM_COUNT];
            for (int i = 0; i < NORM_COUNT; i++) loadedNorms[i] = buf.getFloat();

            // --- MCC table (10000 floats) ---
            float[] loadedMcc = new float[MCC_TABLE_SIZE];
            for (int i = 0; i < MCC_TABLE_SIZE; i++) loadedMcc[i] = buf.getFloat();

            // --- PQ codebooks (M * 256 * SUB_D floats) ---
            float[][][] codebooks = new float[ProductQuantizer.M][ProductQuantizer.CODEBOOK_SIZE][ProductQuantizer.SUB_D];
            for (int m = 0; m < ProductQuantizer.M; m++) {
                for (int c = 0; c < ProductQuantizer.CODEBOOK_SIZE; c++) {
                    codebooks[m][c][0] = buf.getFloat();
                    codebooks[m][c][1] = buf.getFloat();
                }
            }

            // --- IVF centroids (K * 14 floats) ---
            float[][] centroids = new float[K][DIMS];
            for (int c = 0; c < K; c++) {
                for (int d = 0; d < DIMS; d++) centroids[c][d] = buf.getFloat();
            }

            // --- Original vectors (N * 14 floats) --- mmap, don't read into heap
            // vectorsOffset used by ReRankingVectorIndex; create a duplicate view for it
            MappedByteBuffer vectorsMmap = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            vectorsMmap.order(ByteOrder.BIG_ENDIAN);

            // Advance buf past the vectors section — track position naturally
            // (do NOT use vectorsOffset here; vectorsOffset is passed to ReRankingVectorIndex)
            buf.position(buf.position() + N * DIMS * Float.BYTES);

            // --- Labels (N bytes) ---
            byte[] labels = new byte[N];
            buf.get(labels);

            // --- Inverted lists ---
            int[][] idsByCluster = new int[K][];
            byte[][][] codesByCluster = new byte[K][][];
            for (int c = 0; c < K; c++) {
                int count = buf.getInt();
                idsByCluster[c] = new int[count];
                codesByCluster[c] = new byte[count][ProductQuantizer.M];
                for (int i = 0; i < count; i++) {
                    idsByCluster[c][i] = buf.getInt();
                    buf.get(codesByCluster[c][i]);
                }
            }

            // --- Assemble ProductQuantizer from loaded codebooks (no training at runtime) ---
            ProductQuantizer pq = new ProductQuantizer(new KMeans());
            pq.setCodebooks(codebooks);

            // --- Create index via factory ---
            // Original vectors array not loaded into heap (passed null — BruteForce not viable for 3M)
            this.index = factory.create(centroids, idsByCluster, codesByCluster,
                                        null, labels, pq,
                                        vectorsMmap, vectorsOffset, N);
            this.norms = loadedNorms;
            this.mccRisk = loadedMcc;
        }
    }

    public int search(float[] query, int k, int[] neighbors, float[] distances) {
        return index.search(query, k, neighbors, distances);
    }

    public byte[] getIndexLabels() {
        return index.getLabels();
    }

    /**
     * Normalization constants in order:
     * [0] max_amount, [1] max_installments, [2] amount_vs_avg_ratio,
     * [3] max_minutes, [4] max_km, [5] max_tx_count_24h, [6] max_merchant_avg_amount
     */
    public float[] getNormalization() {
        return norms;
    }

    /**
     * MCC risk table indexed by Integer.parseInt(mcc), default 0.5 for unknown MCCs.
     * Size = 10000.
     */
    public float[] getMccRisk() {
        return mccRisk;
    }
}
