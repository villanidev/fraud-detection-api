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

    private static final int DEFAULT_K = 512;  // IVF cluster count
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
        float[][] vectors = ref.vectors();
        byte[]   labels  = ref.labels();
        int N = vectors.length;
        System.out.printf("[preprocess] Loaded %,d vectors%n", N);

        float[] norms    = dataReader.loadNormalization(normalizationJson);
        float[] mccRisks = dataReader.loadMccRisks(mccRiskJson);

        // Train IVF centroids
        System.out.println("[preprocess] Training IVF K-Means (K=" + K + ")...");
        KMeans kMeans = new KMeans();
        float[][] centroids = kMeans.cluster(vectors, K, SEED);

        // Assign vectors to clusters
        System.out.println("[preprocess] Assigning vectors to clusters...");
        int[][] idsByCluster = kMeans.assign(vectors, centroids);

        // Train PQ codebooks
        System.out.println("[preprocess] Training Product Quantizer...");
        ProductQuantizer pq = new ProductQuantizer(kMeans);
        pq.train(vectors, SEED);

        // Encode all vectors
        System.out.println("[preprocess] Encoding vectors...");
        byte[][] allCodes = new byte[N][];
        for (int i = 0; i < N; i++) allCodes[i] = pq.encode(vectors[i]);

        // Build per-cluster code arrays
        byte[][][] codesByCluster = new byte[K][][];
        for (int c = 0; c < K; c++) {
            int[] ids = idsByCluster[c];
            codesByCluster[c] = new byte[ids.length][];
            for (int i = 0; i < ids.length; i++) {
                codesByCluster[c][i] = allCodes[ids[i]];
            }
        }

        // Write data.bin
        System.out.println("[preprocess] Writing " + outputBin + "...");
        dataWriter.write(outputBin, norms, mccRisks, pq,
                centroids, vectors, labels, idsByCluster, codesByCluster);
        System.out.println("[preprocess] Done.");
    }
}
