package villani.dev.preprocessing;

import io.helidon.config.Config;
import io.helidon.service.registry.Service;
import villani.dev.vectorsearch.index.strategies.ivfpq.KMeans;
import villani.dev.vectorsearch.index.strategies.ivfpq.ProductQuantizer;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Build-time preprocessing pipeline.
 * Invoked via --preprocess CLI flag in Main; output is data.bin baked into the Docker image.
 *
 * Pipeline:
 *   1. Load references.json.gz → float[N][14] vectors + byte[N] labels
 *   2. Load normalization.json → float[7] norms
 *   3. Load mcc_risk.json     → float[10000] mccRisks
 *   4. Train IVF (K-Means, K clusters)
 *   5. Train PQ  (M=7 subspaces, 256 centroids each)
 *   6. Encode all vectors to PQ codes
 *   7. Build inverted lists (cluster → vector IDs + PQ codes)
 *   8. Write data.bin
 */
@Service.Singleton
public class PreProcessorService {

    private static final int DEFAULT_K = 1472;  // IVF cluster count
    private static final long SEED = 42L;

    private final DataReader dataReader;
    private final DataWriter dataWriter;
    private final int K;

    @Service.Inject
    public PreProcessorService(DataReader dataReader, DataWriter dataWriter, Config config) {
        this.dataReader = dataReader;
        this.dataWriter = dataWriter;
        this.K = config.get("app.vector-search.clusters").asInt().orElse(DEFAULT_K);
    }

    /**
     * Runs the full preprocessing pipeline.
     *
     * @param referencesGz   path to references.json.gz
     * @param normalizationJson path to normalization.json
     * @param mccRiskJson    path to mcc_risk.json
     * @param outputBin      destination path for data.bin
     */
    public void ingestReferences(Path referencesGz,
                                 Path normalizationJson,
                                 Path mccRiskJson,
                                 Path outputBin) throws IOException {

        System.out.println("[preprocess] Loading references...");
        DataReader.ReferenceData ref = dataReader.loadReferences(referencesGz);
        float[] vectorsFlat = ref.flat();
        byte[]   labels  = ref.labels();
        int N = ref.count();
        System.out.printf("[preprocess] Loaded %,d vectors%n", N);

        float[] norms    = dataReader.loadNormalization(normalizationJson);
        float[] mccRisks = dataReader.loadMccRisks(mccRiskJson);

        // Train IVF centroids
        System.out.println("[preprocess] Training IVF K-Means (K=" + K + ")...");
        KMeans kMeans = new KMeans();
        float[][] centroids = kMeans.cluster(vectorsFlat, N, K, SEED);

        // Assign vectors to clusters
        System.out.println("[preprocess] Assigning vectors to clusters...");
        int[][] idsByCluster = kMeans.assignFlat(vectorsFlat, N, centroids);

        // Train PQ codebooks
        System.out.println("[preprocess] Training Product Quantizer...");
        ProductQuantizer pq = new ProductQuantizer(kMeans);
        pq.train(vectorsFlat, N, SEED);

        // Encode all vectors
        System.out.println("[preprocess] Encoding vectors...");
        short[][] allCodes = new short[N][];
        for (int i = 0; i < N; i++) allCodes[i] = pq.encodeFlat(vectorsFlat, i);

        // Build per-cluster code arrays
        short[][][] codesByCluster = new short[K][][];
        for (int c = 0; c < K; c++) {
            int[] ids = idsByCluster[c];
            codesByCluster[c] = new short[ids.length][];
            for (int i = 0; i < ids.length; i++) {
                codesByCluster[c][i] = allCodes[ids[i]];
            }
        }

        // Build BBoxes per cluster (min/max per dimension)
        System.out.println("[preprocess] Computing cluster BBoxes...");
        final int DIMS = 14;
        float[] bboxMin = new float[K * DIMS];
        float[] bboxMax = new float[K * DIMS];
        // init
        for (int i = 0; i < K * DIMS; i++) {
            bboxMin[i] = Float.POSITIVE_INFINITY;
            bboxMax[i] = Float.NEGATIVE_INFINITY;
        }
        // populate
        for (int c = 0; c < K; c++) {
            int[] ids = idsByCluster[c];
            int baseC = c * DIMS;
            for (int j = 0; j < ids.length; j++) {
                int vid = ids[j];
                int vbase = vid * DIMS;
                for (int d = 0; d < DIMS; d++) {
                    float v = vectorsFlat[vbase + d];
                    int idx = baseC + d;
                    if (v < bboxMin[idx]) bboxMin[idx] = v;
                    if (v > bboxMax[idx]) bboxMax[idx] = v;
                }
            }
        }

        // Write data.bin
        System.out.println("[preprocess] Writing " + outputBin + "...");
        dataWriter.write(outputBin, norms, mccRisks, pq,
            centroids, ref.flat(), labels, idsByCluster, codesByCluster, bboxMin, bboxMax);
        System.out.println("[preprocess] Done.");
    }
}
