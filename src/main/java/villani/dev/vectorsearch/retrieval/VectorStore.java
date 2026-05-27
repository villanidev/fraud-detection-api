package villani.dev.vectorsearch.retrieval;

import io.helidon.service.registry.Service;
import villani.dev.vectorsearch.index.VectorIndex;
import villani.dev.vectorsearch.index.VectorIndexFactory;
import villani.dev.vectorsearch.index.strategies.ivfpq.KMeans;
import villani.dev.vectorsearch.index.strategies.ivfpq.ProductQuantizer;

import java.io.IOException;
import java.nio.ByteBuffer;
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
    private FileChannel vectorsChannel;
    private long vectorsOffset;

    private volatile float[][] centroids;
    private volatile int[][] idsByCluster;
    private volatile short[][] codesByCluster;
    private volatile float[] vectors;
    private volatile byte[] labels;
    private volatile ProductQuantizer pq;
    private volatile int vectorCount;

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
            // Buffer de leitura de apenas 8 KB – nao aloca mais o arquivo inteiro.
            ByteBuffer readBuf = ByteBuffer.allocateDirect(8192);
            readBuf.order(ByteOrder.BIG_ENDIAN);
            readBuf.flip(); // inicialmente vazio, para forçar leitura

            // --- Header ---
            this.vectorsChannel = FileChannel.open(path, StandardOpenOption.READ);
            int magic = readInt(channel, readBuf);
            if (magic != MAGIC) throw new IOException("Invalid data.bin: wrong magic 0x" + Integer.toHexString(magic));
            int version = readInt(channel, readBuf);
            if (version != VERSION) throw new IOException("Unsupported data.bin version: " + version);
            int K = readInt(channel, readBuf);
            int N = readInt(channel, readBuf);
            this.vectorsOffset = readLong(channel, readBuf);

            // --- Normalization (7 floats) ---
            float[] loadedNorms = new float[NORM_COUNT];
            for (int i = 0; i < NORM_COUNT; i++) loadedNorms[i] = readFloat(channel, readBuf);

            // --- MCC table (10000 floats) ---
            float[] loadedMcc = new float[MCC_TABLE_SIZE];
            for (int i = 0; i < MCC_TABLE_SIZE; i++) loadedMcc[i] = readFloat(channel, readBuf);

            // --- PQ codebooks ---
            float[][][] codebooks = new float[ProductQuantizer.M][ProductQuantizer.CODEBOOK_SIZE][ProductQuantizer.SUB_D];
            for (int m = 0; m < ProductQuantizer.M; m++) {
                for (int c = 0; c < ProductQuantizer.CODEBOOK_SIZE; c++) {
                    codebooks[m][c][0] = readFloat(channel, readBuf);
                    codebooks[m][c][1] = readFloat(channel, readBuf);
                }
            }

            // --- IVF centroids (K * 14 floats) ---
            float[][] centroids = new float[K][DIMS];
            for (int c = 0; c < K; c++) {
                for (int d = 0; d < DIMS; d++) centroids[c][d] = readFloat(channel, readBuf);
            }
            this.centroids = centroids;

            // posiciona o canal exatamente no início da seção de vetores
            channel.position(vectorsOffset);

            // --- Original vectors (opcional, só se brute-force) ---
            float[] vectorsFlat = null;
            if (factory.isBruteForce()) {
                vectorsFlat = new float[N * DIMS];
                for (int i = 0; i < N; i++) {
                    int base = i * DIMS;
                    for (int d = 0; d < DIMS; d++) {
                        vectorsFlat[base + d] = readFloat(channel, readBuf);
                    }
                }
                this.vectors = vectorsFlat;
            } else {
                // Pula a seção de vetores (não usada no ivf_pq)
                long vectorSectionBytes = (long) N * DIMS * Float.BYTES;
                skipFully(channel, vectorSectionBytes);  // ou use skipFully
                readBuf.clear();
                readBuf.flip(); // invalida buffer, pois a posição saltou
            }

            // --- Labels (N bytes) ---
            byte[] labels = new byte[N];
            readFully(channel, readBuf, labels);
            this.labels = labels;

            // --- Inverted lists ---
            int[][] idsByCluster = new int[K][];
            short[][] codesByCluster = new short[K][];
            for (int c = 0; c < K; c++) {
                int count = readInt(channel, readBuf);
                idsByCluster[c] = new int[count];
                codesByCluster[c] = new short[count * ProductQuantizer.M];
                for (int i = 0; i < count; i++) {
                    idsByCluster[c][i] = readInt(channel, readBuf);
                    // Lê exatamente M shorts do código
                    for (int m = 0; m < ProductQuantizer.M; m++) {
                        codesByCluster[c][i * ProductQuantizer.M + m] = readShort(channel, readBuf);
                    }
                }
            }
            this.idsByCluster = idsByCluster;
            this.codesByCluster = codesByCluster;

            // --- Monta ProductQuantizer e cria o índice ---
            ProductQuantizer pq = new ProductQuantizer(new KMeans());
            pq.setCodebooks(codebooks);
            this.pq = pq;
            this.vectorCount = N;

            // Cria o índice padrão com os parâmetros do config
                this.index = factory.create(centroids, idsByCluster, codesByCluster,
                    vectors, labels, pq, this.vectorsChannel, this.vectorsOffset, N);
            this.norms = loadedNorms;
            this.mccRisk = loadedMcc;

        }
    }

    private int readInt(FileChannel ch, ByteBuffer buf) throws IOException {
        ensureBuffer(ch, buf, 4);
        return buf.getInt();
    }

    private long readLong(FileChannel ch, ByteBuffer buf) throws IOException {
        ensureBuffer(ch, buf, 8);
        return buf.getLong();
    }

    private float readFloat(FileChannel ch, ByteBuffer buf) throws IOException {
        ensureBuffer(ch, buf, 4);
        return buf.getFloat();
    }

    private short readShort(FileChannel ch, ByteBuffer buf) throws IOException {
        ensureBuffer(ch, buf, 2);
        return buf.getShort();
    }

    private void readFully(FileChannel ch, ByteBuffer buf, byte[] dst) throws IOException {
        int offset = 0;
        while (offset < dst.length) {
            if (!buf.hasRemaining()) {
                buf.clear();
                int read = ch.read(buf);
                if (read == -1) throw new IOException("EOF reached unexpectedly");
                buf.flip();
            }
            int toCopy = Math.min(buf.remaining(), dst.length - offset);
            buf.get(dst, offset, toCopy);
            offset += toCopy;
        }
    }

    private void skipFully(FileChannel ch, long bytes) throws IOException {
        long remaining = bytes;
        while (remaining > 0) {
            long skipped = ch.position() + remaining; // move para o final
            ch.position(skipped); // a própria chamada já posiciona
            remaining = 0; // encerra
        }
    }

    private void ensureBuffer(FileChannel ch, ByteBuffer buf, int needed) throws IOException {
        // Se já temos dados suficientes, não faz nada
        if (buf.remaining() >= needed) return;

        // Move os dados não lidos para o início do buffer
        buf.compact();                // pos = bytes restantes, limit = capacidade
        buf.flip();                   // prepara para leitura (pos=0, limit=bytes restantes)

        // Se mesmo após compactar ainda não temos suficientes, preenchemos
        while (buf.remaining() < needed) {
            // Coloca o buffer em modo escrita após os dados já existentes
            int pos = buf.limit();    // bytes já presentes
            buf.limit(buf.capacity());
            buf.position(pos);

            int read = ch.read(buf);
            if (read == -1) {
                // EOF: não conseguimos completar a leitura
                throw new IOException("EOF reached before reading " + needed + " bytes (available: " + buf.position() + ")");
            }

            buf.flip();  // prepara para leitura novamente
        }
    }

    public int search(float[] query, int topK, int[] neighbors, float[] distances) {
        return index.search(query, topK, neighbors, distances);
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

    public VectorIndex createIndexForBenchmark(int nprobe, int candidates) {
        return factory.create(
                centroids, idsByCluster, codesByCluster,
                vectors, labels, pq,
                vectorsChannel, vectorsOffset, vectorCount,
                nprobe, candidates
        );
    }
}
