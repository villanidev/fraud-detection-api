package villani.dev.vectorsearch.retrieval;

import io.helidon.service.registry.Service;
import villani.dev.vectorsearch.index.VectorIndex;
import villani.dev.vectorsearch.index.VectorIndexFactory;
import villani.dev.vectorsearch.index.strategies.ivfpq.KMeans;
import villani.dev.vectorsearch.index.strategies.ivfpq.ProductQuantizer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Facade for all vector data and search.
 * Loads data.bin at startup, exposes normalization/MCC constants for embedding,
 * and delegates search to the configured VectorIndex strategy.
 *
 * data.bin V2 binary layout (all multi-byte values in BIG_ENDIAN):
 *
 *   [ Header — 128 bytes ]
 *     int magic
 *     int version         = 2
 *     int dims            = 14
 *     int flags
 *     int K               — number of IVF clusters
 *     int N               — number of vectors
 *     int pqSubQuantizers = ProductQuantizer.M
 *     int codebookSize    = ProductQuantizer.CODEBOOK_SIZE
 *     long normsOffset
 *     long mccOffset
 *     long codebooksOffset
 *     long centroidOffset
 *     long vectorsOffset
 *     long labelsOffset
 *     long clusterDirectoryOffset
 *     long idsPayloadOffset
 *     long codesPayloadOffset
 *     long quantizedPayloadOffset
 *     int  quantizedStrideBytes
 *     int  reserved
 *     long fileSize
 *
 *   [ Normalization — 7 floats = 28 bytes ]
 *   [ MCC table — 10000 floats = 40000 bytes ]
 *   [ PQ codebooks — M*CODEBOOK_SIZE*SUB_D floats ]
 *   [ IVF centroids — K*14 floats ]
 *   [ Original vectors — N*14 floats ]
 *   [ Labels — N bytes ]
 *   [ Cluster directory — per cluster: count/fraudCount/offsets/flags/radius/min/max ]
 *   [ IDs payload — N ints ordered by cluster ]
 *   [ Codes payload — N*M shorts ordered by cluster ]
 *   [ Scalar quantized payload — N*16 bytes ordered by cluster ]
 */
@Service.Singleton
public class VectorStore {

    public static final int MAGIC = 0x52494E44;
    public static final int VERSION_V2 = 2;
     public static final int FLAG_HAS_SCALAR_QUANTIZED_PAYLOAD = 1;
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
    private volatile int[] clusterSizes;
    private volatile int[] idsStartIndex;
    private volatile IntBuffer idsPayload;
    private volatile int[] codesStartIndex;
    private volatile ShortBuffer codesPayload;
    private volatile float[] vectors;
    private volatile byte[] labels;
    private volatile ProductQuantizer pq;
    private volatile int vectorCount;
    private volatile float[] clusterMinValues;
    private volatile float[] clusterMaxValues;
    private volatile int[] clusterFlags;
    private volatile ByteBuffer scalarQuantizedPayload;
    private volatile int scalarQuantizedStrideBytes;

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
            ByteBuffer readBuf = ByteBuffer.allocateDirect(8192);
            readBuf.order(ByteOrder.BIG_ENDIAN);
            readBuf.flip();

            this.vectorsChannel = FileChannel.open(path, StandardOpenOption.READ);
            int magic = readInt(channel, readBuf);
            if (magic != MAGIC) throw new IOException("Invalid data.bin: wrong magic 0x" + Integer.toHexString(magic));
            int version = readInt(channel, readBuf);
            if (version == VERSION_V2) {
                loadV2(channel, readBuf);
                return;
            }
            throw new IOException("Unsupported data.bin version: " + version);
        }
    }

    private void loadV2(FileChannel channel, ByteBuffer readBuf) throws IOException {
        int dims = readInt(channel, readBuf);
        if (dims != DIMS) throw new IOException("Unsupported vector dimension: " + dims);
        int flags = readInt(channel, readBuf);
        int K = readInt(channel, readBuf);
        int N = readInt(channel, readBuf);
        int pqSubQuantizers = readInt(channel, readBuf);
        int codebookSize = readInt(channel, readBuf);
        if (pqSubQuantizers != ProductQuantizer.M) {
            throw new IOException("Unsupported PQ subquantizers: " + pqSubQuantizers);
        }
        if (codebookSize != ProductQuantizer.CODEBOOK_SIZE) {
            throw new IOException("Unsupported PQ codebook size: " + codebookSize);
        }

        long normsOffset = readLong(channel, readBuf);
        long mccOffset = readLong(channel, readBuf);
        long codebooksOffset = readLong(channel, readBuf);
        long centroidOffset = readLong(channel, readBuf);
        long vectorsOffset = readLong(channel, readBuf);
        long labelsOffset = readLong(channel, readBuf);
        long clusterDirectoryOffset = readLong(channel, readBuf);
        long idsPayloadOffset = readLong(channel, readBuf);
        long codesPayloadOffset = readLong(channel, readBuf);
        long quantizedPayloadOffset = readLong(channel, readBuf);
        int quantizedStrideBytes = readInt(channel, readBuf);
        readInt(channel, readBuf);
        readLong(channel, readBuf);

        channel.position(normsOffset);
        invalidate(readBuf);
        float[] loadedNorms = new float[NORM_COUNT];
        for (int i = 0; i < NORM_COUNT; i++) loadedNorms[i] = readFloat(channel, readBuf);

        channel.position(mccOffset);
        invalidate(readBuf);
        float[] loadedMcc = new float[MCC_TABLE_SIZE];
        for (int i = 0; i < MCC_TABLE_SIZE; i++) loadedMcc[i] = readFloat(channel, readBuf);

        channel.position(codebooksOffset);
        invalidate(readBuf);
        float[] codebooksFlat = readCodebooks(channel, readBuf);

        channel.position(centroidOffset);
        invalidate(readBuf);
        float[][] loadedCentroids = readCentroids(channel, readBuf, K);

        this.vectorsOffset = vectorsOffset;
        float[] vectorsFlat = readVectorsSection(channel, readBuf, N, vectorsOffset);

        channel.position(labelsOffset);
        invalidate(readBuf);
        byte[] loadedLabels = new byte[N];
        readFully(channel, readBuf, loadedLabels);

        channel.position(clusterDirectoryOffset);
        invalidate(readBuf);
        int[] counts = new int[K];
        int[] idsStartIndex = new int[K];
        int[] codesStartIndex = new int[K];
        int[] loadedClusterFlags = new int[K];
        float[] loadedClusterMinValues = new float[K * DIMS];
        float[] loadedClusterMaxValues = new float[K * DIMS];
        for (int c = 0; c < K; c++) {
            counts[c] = readInt(channel, readBuf);
            readInt(channel, readBuf);
            idsStartIndex[c] = readInt(channel, readBuf);
            codesStartIndex[c] = readInt(channel, readBuf);
            loadedClusterFlags[c] = readInt(channel, readBuf);
            readInt(channel, readBuf);
            readFloat(channel, readBuf);
            readFloat(channel, readBuf);
            int base = c * DIMS;
            for (int d = 0; d < DIMS; d++) loadedClusterMinValues[base + d] = readFloat(channel, readBuf);
            for (int d = 0; d < DIMS; d++) loadedClusterMaxValues[base + d] = readFloat(channel, readBuf);
        }

        IntBuffer loadedIdsPayload = mapIntSection(channel, idsPayloadOffset, (long) N * Integer.BYTES);
        ShortBuffer loadedCodesPayload = mapShortSection(channel, codesPayloadOffset, (long) N * ProductQuantizer.M * Short.BYTES);

        ByteBuffer loadedScalarQuantizedPayload = null;
        if ((flags & FLAG_HAS_SCALAR_QUANTIZED_PAYLOAD) != 0) {
            loadedScalarQuantizedPayload = mapByteSection(channel, quantizedPayloadOffset, (long) N * quantizedStrideBytes);
        }

        publishLoadedIndex(N, loadedNorms, loadedMcc, codebooksFlat, loadedCentroids,
            vectorsFlat, loadedLabels, counts, idsStartIndex, loadedIdsPayload, codesStartIndex, loadedCodesPayload,
            loadedClusterMinValues, loadedClusterMaxValues,
            loadedClusterFlags,
            loadedScalarQuantizedPayload, quantizedStrideBytes);
    }

    private float[] readCodebooks(FileChannel channel, ByteBuffer readBuf) throws IOException {
        float[] codebooksFlat = new float[ProductQuantizer.M * ProductQuantizer.CODEBOOK_SIZE * ProductQuantizer.SUB_D];
        for (int m = 0; m < ProductQuantizer.M; m++) {
            for (int c = 0; c < ProductQuantizer.CODEBOOK_SIZE; c++) {
                int base = (m * ProductQuantizer.CODEBOOK_SIZE + c) * ProductQuantizer.SUB_D;
                codebooksFlat[base] = readFloat(channel, readBuf);
                codebooksFlat[base + 1] = readFloat(channel, readBuf);
            }
        }
        return codebooksFlat;
    }

    private float[][] readCentroids(FileChannel channel, ByteBuffer readBuf, int clusterCount) throws IOException {
        float[][] loadedCentroids = new float[clusterCount][DIMS];
        for (int c = 0; c < clusterCount; c++) {
            for (int d = 0; d < DIMS; d++) loadedCentroids[c][d] = readFloat(channel, readBuf);
        }
        return loadedCentroids;
    }

    private float[] readVectorsSection(FileChannel channel, ByteBuffer readBuf, int vectorCount, long sectionOffset) throws IOException {
        channel.position(sectionOffset);
        invalidate(readBuf);

        float[] vectorsFlat = null;
        if (factory.isBruteForce()) {
            vectorsFlat = new float[vectorCount * DIMS];
            for (int i = 0; i < vectorCount; i++) {
                int base = i * DIMS;
                for (int d = 0; d < DIMS; d++) {
                    vectorsFlat[base + d] = readFloat(channel, readBuf);
                }
            }
            return vectorsFlat;
        }

        skipFully(channel, (long) vectorCount * DIMS * Float.BYTES);
        invalidate(readBuf);
        return null;
    }

    private void publishLoadedIndex(int loadedVectorCount,
                                    float[] loadedNorms,
                                    float[] loadedMcc,
                                    float[] codebooksFlat,
                                    float[][] loadedCentroids,
                                    float[] vectorsFlat,
                                    byte[] loadedLabels,
                                    int[] loadedClusterSizes,
                                    int[] loadedIdsStartIndex,
                                    IntBuffer loadedIdsPayload,
                                    int[] loadedCodesStartIndex,
                                    ShortBuffer loadedCodesPayload,
                                    float[] loadedClusterMinValues,
                                    float[] loadedClusterMaxValues,
                                    int[] loadedClusterFlags,
                                    ByteBuffer loadedScalarQuantizedPayload,
                                    int loadedScalarQuantizedStrideBytes) {
        ProductQuantizer loadedPq = new ProductQuantizer(new KMeans());
        loadedPq.setCodebooksFlat(codebooksFlat);

        this.centroids = loadedCentroids;
        this.clusterSizes = loadedClusterSizes;
        this.idsStartIndex = loadedIdsStartIndex;
        this.idsPayload = loadedIdsPayload;
        this.codesStartIndex = loadedCodesStartIndex;
        this.codesPayload = loadedCodesPayload;
        this.vectors = vectorsFlat;
        this.labels = loadedLabels;
        this.pq = loadedPq;
        this.vectorCount = loadedVectorCount;
        this.clusterMinValues = loadedClusterMinValues;
        this.clusterMaxValues = loadedClusterMaxValues;
        this.clusterFlags = loadedClusterFlags;
        this.scalarQuantizedPayload = loadedScalarQuantizedPayload;
        this.scalarQuantizedStrideBytes = loadedScalarQuantizedStrideBytes;
        this.norms = loadedNorms;
        this.mccRisk = loadedMcc;
        this.index = factory.create(loadedCentroids, loadedClusterSizes, loadedIdsStartIndex, loadedIdsPayload, loadedCodesStartIndex, loadedCodesPayload,
            vectorsFlat, loadedLabels, loadedPq,
            loadedClusterMinValues, loadedClusterMaxValues, loadedClusterFlags,
            loadedScalarQuantizedPayload, loadedScalarQuantizedStrideBytes,
            this.vectorsChannel, this.vectorsOffset, loadedVectorCount);
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
        ch.position(ch.position() + bytes);
    }

    private void invalidate(ByteBuffer buf) {
        buf.clear();
        buf.flip();
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

    private IntBuffer mapIntSection(FileChannel channel, long offset, long lengthBytes) throws IOException {
        MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, offset, lengthBytes);
        mapped.order(ByteOrder.BIG_ENDIAN);
        return mapped.asIntBuffer();
    }

    private ShortBuffer mapShortSection(FileChannel channel, long offset, long lengthBytes) throws IOException {
        MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, offset, lengthBytes);
        mapped.order(ByteOrder.BIG_ENDIAN);
        return mapped.asShortBuffer();
    }

    private ByteBuffer mapByteSection(FileChannel channel, long offset, long lengthBytes) throws IOException {
        return channel.map(FileChannel.MapMode.READ_ONLY, offset, lengthBytes);
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
                centroids, clusterSizes, idsStartIndex, idsPayload, codesStartIndex, codesPayload,
            vectors, labels, pq,
            clusterMinValues, clusterMaxValues, clusterFlags,
            scalarQuantizedPayload, scalarQuantizedStrideBytes,
                vectorsChannel, vectorsOffset, vectorCount,
                nprobe, candidates
        );
    }

    public VectorIndex createIndexForBenchmark(int nprobe, int candidates, String mode, boolean pruneEnabled) {
        return factory.create(
                centroids, clusterSizes, idsStartIndex, idsPayload, codesStartIndex, codesPayload,
                vectors, labels, pq,
                clusterMinValues, clusterMaxValues, clusterFlags,
                scalarQuantizedPayload, scalarQuantizedStrideBytes,
                vectorsChannel, vectorsOffset, vectorCount,
                nprobe, candidates, mode, pruneEnabled
        );
    }

    public float[] getClusterMinValues() {
        return clusterMinValues;
    }

    public float[] getClusterMaxValues() {
        return clusterMaxValues;
    }

    public int[] getClusterFlags() {
        return clusterFlags;
    }

    public ByteBuffer getScalarQuantizedPayload() {
        return scalarQuantizedPayload;
    }

    public int getScalarQuantizedStrideBytes() {
        return scalarQuantizedStrideBytes;
    }
}
