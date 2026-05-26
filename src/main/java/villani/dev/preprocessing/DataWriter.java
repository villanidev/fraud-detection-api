package villani.dev.preprocessing;

import io.helidon.service.registry.Service;
import villani.dev.vectorsearch.index.strategies.ivfpq.ProductQuantizer;
import villani.dev.vectorsearch.retrieval.VectorStore;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Writes the preprocessed index to a binary data.bin file.
 *
 * Binary layout (all values in BIG-ENDIAN — DataOutputStream default):
 *
 *   [ Header — 20 bytes ]
 *     int  magic          = 0x52494E44
 *     int  version        = 1
 *     int  K              — IVF cluster count
 *     int  N              — total vector count
 *     long vectorsOffset  — byte offset of the original vectors section
 *
 *   [ Normalization — 7 × 4 = 28 bytes ]
 *   [ MCC table    — 10000 × 4 = 40000 bytes ]
 *   [ PQ codebooks — M × 256 × SUB_D × 4 = 14336 bytes ]
 *   [ IVF centroids — K × 14 × 4 bytes ]
 *   [ Original vectors — N × 14 × 4 bytes ]  ← vectorsOffset
 *   [ Labels           — N × 1 byte ]
 *   [ Inverted lists   — per cluster: int count, then count × (int id + 7 bytes PQ code) ]
 *
 * Note: VectorStore.load() uses nativeOrder() via MappedByteBuffer, so the file must be written
 * consistently. We use DataOutputStream (big-endian) and read with buf.order(BIG_ENDIAN) in
 * VectorStore — ensure both agree. Alternatively switch both to native order using NIO directly.
 * Current implementation uses DataOutputStream (big-endian); VectorStore must match.
 */
@Service.Singleton
public class DataWriter {

    private static final int DIMS = 14;
    private static final int MCC_TABLE_SIZE = VectorStore.MCC_TABLE_SIZE;

    /**
     * Writes the full index to the given output path.
     *
     * @param outputPath   destination file (will be created or overwritten)
     * @param norms        normalization constants float[7]
     * @param mccRisks     MCC risk table float[10000]
     * @param pq           trained ProductQuantizer (codebooks already set)
     * @param centroids    IVF centroids float[K][14]
     * @param vectorsFlat      original reference vectors float[N][14]
     * @param labels       labels byte[N]  0=legit, 1=fraud
     * @param idsByCluster inverted list IDs int[K][]
     * @param codesByCluster PQ codes per cluster byte[K][][7]
     */
    public void write(Path outputPath,
                      float[] norms,
                      float[] mccRisks,
                      ProductQuantizer pq,
                      float[][] centroids,
                      float[] vectorsFlat,
                      byte[] labels,
                      int[][] idsByCluster,
                      byte[][][] codesByCluster) throws IOException {

        int K = centroids.length;
        int N = vectorsFlat.length / DIMS;

        // Pre-compute vectorsOffset:
        // header(24) + norms(28) + mcc(40000) + codebooks(14336) + centroids(K*14*4)
        // header = 4 ints (4 bytes each) + 1 long (8 bytes) = 24 bytes
        long vectorsOffset = 24L
                + (long) VectorStore.NORM_COUNT * Float.BYTES
                + (long) MCC_TABLE_SIZE * Float.BYTES
                + pq.serializedSize()
                + (long) K * DIMS * Float.BYTES;

        try (OutputStream fos = Files.newOutputStream(outputPath);
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16))) {

            // --- Header ---
            out.writeInt(VectorStore.MAGIC);
            out.writeInt(VectorStore.VERSION);
            out.writeInt(K);
            out.writeInt(N);
            out.writeLong(vectorsOffset);

            // --- Normalization ---
            for (float f : norms) out.writeFloat(f);

            // --- MCC table ---
            for (int i = 0; i < MCC_TABLE_SIZE; i++) out.writeFloat(mccRisks[i]);

            // --- PQ codebooks ---
            float[][][] codebooks = pq.getCodebooks();
            for (int m = 0; m < ProductQuantizer.M; m++) {
                for (int c = 0; c < ProductQuantizer.CODEBOOK_SIZE; c++) {
                    out.writeFloat(codebooks[m][c][0]);
                    out.writeFloat(codebooks[m][c][1]);
                }
            }

            // --- IVF centroids ---
            for (int c = 0; c < K; c++) {
                for (int d = 0; d < DIMS; d++) out.writeFloat(centroids[c][d]);
            }

            // --- Original vectors (at vectorsOffset) ---
            for (int i = 0; i < N; i++) {
                int base = i * DIMS;
                for (int d = 0; d < DIMS; d++) out.writeFloat(vectorsFlat[base + d]);
            }

            // --- Labels ---
            out.write(labels);

            // --- Inverted lists ---
            for (int c = 0; c < K; c++) {
                int count = idsByCluster[c].length;
                out.writeInt(count);
                for (int i = 0; i < count; i++) {
                    out.writeInt(idsByCluster[c][i]);
                    out.write(codesByCluster[c][i]); // 7 bytes
                }
            }
        }
    }
}
