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
 * Binary layout V2 (all values in BIG_ENDIAN — DataOutputStream default):
 *
 *   [ Header — 128 bytes ]
 *     int  magic              = 0x52494E44
 *     int  version            = 2
 *     int  dims               = 14
 *     int  flags
 *     int  K                  — IVF cluster count
 *     int  N                  — total vector count
 *     int  pqSubQuantizers    = ProductQuantizer.M
 *     int  codebookSize       = ProductQuantizer.CODEBOOK_SIZE
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
 *   [ Normalization — 7 × 4 = 28 bytes ]
 *   [ MCC table — 10000 × 4 bytes ]
 *   [ PQ codebooks — M × CODEBOOK_SIZE × SUB_D × 4 bytes ]
 *   [ IVF centroids — K × 14 × 4 bytes ]
 *   [ Original vectors — N × 14 × 4 bytes ]
 *   [ Labels — N × 1 byte ]
 *   [ Cluster directory — per cluster: int count, int idsStartIndex, int codesStartIndex, int reserved ]
 *   [ IDs payload — N × int ]
 *   [ Codes payload — N × M × short ]
 *   [ Scalar quantized payload — N × 16 bytes ]
 *
 * Note: VectorStore reads the file with BIG_ENDIAN ByteBuffers, so the writer must keep the same order.
 */
@Service.Singleton
public class DataWriter {

    private static final int DIMS = 14;
    private static final int MCC_TABLE_SIZE = VectorStore.MCC_TABLE_SIZE;
    private static final int CLUSTER_DIRECTORY_ENTRY_BYTES = 144;
    private static final int HEADER_V2_BYTES = 128;
    private static final int ALIGNMENT = 64;
    private static final int SCALAR_QUANTIZED_STRIDE_BYTES = 16;
    private static final int FLAG_HAS_SCALAR_QUANTIZED_PAYLOAD = 1;

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
    * @param codesByCluster flat PQ codes per cluster short[K][count*M]
    * @param clusterMinValues flattened min values by cluster [K*14]
    * @param clusterMaxValues flattened max values by cluster [K*14]
    * @param clusterRadiusSquared max squared centroid distance by cluster [K]
    * @param clusterFlags per-cluster flags for sentinel presence [K]
    * @param clusterFraudCounts fraud label count per cluster [K]
     */
    public void write(Path outputPath,
                      float[] norms,
                      float[] mccRisks,
                      ProductQuantizer pq,
                      float[][] centroids,
                      float[] vectorsFlat,
                      byte[] labels,
                      int[][] idsByCluster,
                      short[][] codesByCluster,
                      float[] clusterMinValues,
                      float[] clusterMaxValues,
                      float[] clusterRadiusSquared,
                      int[] clusterFlags,
                      int[] clusterFraudCounts) throws IOException {

        int K = centroids.length;
        int N = vectorsFlat.length / DIMS;
        int totalCodeCount = N * ProductQuantizer.M;

        long normsOffset = HEADER_V2_BYTES;
        long mccOffset = align(normsOffset + (long) VectorStore.NORM_COUNT * Float.BYTES);
        long codebooksOffset = align(mccOffset + (long) MCC_TABLE_SIZE * Float.BYTES);
        long centroidOffset = align(codebooksOffset + pq.serializedSize());
        long vectorsOffset = align(centroidOffset + (long) K * DIMS * Float.BYTES);
        long labelsOffset = align(vectorsOffset + (long) N * DIMS * Float.BYTES);
        long clusterDirectoryOffset = align(labelsOffset + N);
        long idsPayloadOffset = align(clusterDirectoryOffset + (long) K * CLUSTER_DIRECTORY_ENTRY_BYTES);
        long codesPayloadOffset = align(idsPayloadOffset + (long) N * Integer.BYTES);
        long quantizedPayloadOffset = align(codesPayloadOffset + (long) totalCodeCount * Short.BYTES);
        long fileSize = align(quantizedPayloadOffset + (long) N * SCALAR_QUANTIZED_STRIDE_BYTES);

        try (OutputStream fos = Files.newOutputStream(outputPath);
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16))) {

            writeHeaderV2(out, K, N, normsOffset, mccOffset, codebooksOffset,
                    centroidOffset, vectorsOffset, labelsOffset,
                    clusterDirectoryOffset, idsPayloadOffset, codesPayloadOffset,
                    quantizedPayloadOffset, fileSize);

            // --- Normalization ---
            for (float f : norms) out.writeFloat(f);
            padToOffset(out, mccOffset);

            // --- MCC table ---
            for (int i = 0; i < MCC_TABLE_SIZE; i++) out.writeFloat(mccRisks[i]);
            padToOffset(out, codebooksOffset);

            // --- PQ codebooks (flattened) ---
            float[] codebooksFlat = pq.getCodebooksFlat();
            for (int m = 0; m < ProductQuantizer.M; m++) {
                int base = m * ProductQuantizer.CODEBOOK_SIZE * ProductQuantizer.SUB_D;
                for (int c = 0; c < ProductQuantizer.CODEBOOK_SIZE; c++) {
                    int idx = base + c * ProductQuantizer.SUB_D;
                    out.writeFloat(codebooksFlat[idx]);
                    out.writeFloat(codebooksFlat[idx + 1]);
                }
            }
            padToOffset(out, centroidOffset);

            // --- IVF centroids ---
            for (int c = 0; c < K; c++) {
                for (int d = 0; d < DIMS; d++) out.writeFloat(centroids[c][d]);
            }
            padToOffset(out, vectorsOffset);

            // --- Original vectors (at vectorsOffset) ---
            for (int i = 0; i < N; i++) {
                int base = i * DIMS;
                for (int d = 0; d < DIMS; d++) out.writeFloat(vectorsFlat[base + d]);
            }
            padToOffset(out, labelsOffset);

            // --- Labels ---
            out.write(labels);
            padToOffset(out, clusterDirectoryOffset);

            // --- Cluster directory ---
            int idsStartIndex = 0;
            int codesStartIndex = 0;
            for (int c = 0; c < K; c++) {
                int count = idsByCluster[c].length;
                out.writeInt(count);
                out.writeInt(clusterFraudCounts[c]);
                out.writeInt(idsStartIndex);
                out.writeInt(codesStartIndex);
                out.writeInt(clusterFlags[c]);
                out.writeInt(0);
                out.writeFloat(clusterRadiusSquared[c]);
                out.writeFloat(0f);
                int base = c * DIMS;
                for (int d = 0; d < DIMS; d++) out.writeFloat(clusterMinValues[base + d]);
                for (int d = 0; d < DIMS; d++) out.writeFloat(clusterMaxValues[base + d]);
                idsStartIndex += count;
                codesStartIndex += count * ProductQuantizer.M;
            }
            padToOffset(out, idsPayloadOffset);

            // --- IDs payload ---
            for (int c = 0; c < K; c++) {
                int[] ids = idsByCluster[c];
                for (int id : ids) out.writeInt(id);
            }
            padToOffset(out, codesPayloadOffset);

            // --- Codes payload ---
            for (int c = 0; c < K; c++) {
                short[] clusterCodes = codesByCluster[c];
                for (short code : clusterCodes) {
                    out.writeShort(code);
                }
            }
            padToOffset(out, quantizedPayloadOffset);

            // --- Scalar quantized payload ---
            for (int c = 0; c < K; c++) {
                int[] ids = idsByCluster[c];
                for (int id : ids) {
                    writeScalarQuantizedVector(out, vectorsFlat, id);
                }
            }
            padToOffset(out, fileSize);
        }
    }

    private static void writeHeaderV2(DataOutputStream out,
                                      int clusterCount,
                                      int vectorCount,
                                      long normsOffset,
                                      long mccOffset,
                                      long codebooksOffset,
                                      long centroidOffset,
                                      long vectorsOffset,
                                      long labelsOffset,
                                      long clusterDirectoryOffset,
                                      long idsPayloadOffset,
                                      long codesPayloadOffset,
                                      long quantizedPayloadOffset,
                                      long fileSize) throws IOException {
        out.writeInt(VectorStore.MAGIC);
        out.writeInt(VectorStore.VERSION_V2);
        out.writeInt(DIMS);
        out.writeInt(FLAG_HAS_SCALAR_QUANTIZED_PAYLOAD);
        out.writeInt(clusterCount);
        out.writeInt(vectorCount);
        out.writeInt(ProductQuantizer.M);
        out.writeInt(ProductQuantizer.CODEBOOK_SIZE);
        out.writeLong(normsOffset);
        out.writeLong(mccOffset);
        out.writeLong(codebooksOffset);
        out.writeLong(centroidOffset);
        out.writeLong(vectorsOffset);
        out.writeLong(labelsOffset);
        out.writeLong(clusterDirectoryOffset);
        out.writeLong(idsPayloadOffset);
        out.writeLong(codesPayloadOffset);
        out.writeLong(quantizedPayloadOffset);
        out.writeInt(SCALAR_QUANTIZED_STRIDE_BYTES);
        out.writeInt(0);
        out.writeLong(fileSize);
        padToOffset(out, HEADER_V2_BYTES);
    }

    private static void writeScalarQuantizedVector(DataOutputStream out, float[] vectorsFlat, int vectorId) throws IOException {
        int base = vectorId * DIMS;
        for (int d = 0; d < DIMS; d++) {
            out.writeByte(encodeScalarDimension(d, vectorsFlat[base + d]));
        }
        out.writeByte(0);
        out.writeByte(0);
    }

    private static int encodeScalarDimension(int dim, float value) {
        if (dim == 9 || dim == 10 || dim == 11) {
            return value >= 0.5f ? 255 : 0;
        }
        if (dim == 5 || dim == 6) {
            if (value == -1f) return 255;
            return quantizeToByte(value, 254f);
        }
        return quantizeToByte(value, 255f);
    }

    private static int quantizeToByte(float value, float scale) {
        float clamped = value < 0f ? 0f : Math.min(value, 1f);
        return Math.round(clamped * scale);
    }

    private static long align(long value) {
        long remainder = value % ALIGNMENT;
        return remainder == 0 ? value : value + (ALIGNMENT - remainder);
    }

    private static void padToOffset(DataOutputStream out, long targetOffset) throws IOException {
        long current = out.size();
        while (current < targetOffset) {
            out.writeByte(0);
            current++;
        }
    }
}
