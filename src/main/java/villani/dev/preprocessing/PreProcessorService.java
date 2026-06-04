package villani.dev.preprocessing;

import io.helidon.config.Config;
import io.helidon.service.registry.Service;
import villani.dev.vectorsearch.index.strategies.ivfpq.KMeans;
import villani.dev.vectorsearch.index.strategies.ivfpq.ProductQuantizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.stream.IntStream;

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
    private static final int DIMS = 14;
    private static final int FLAG_HAS_SENTINEL_DIM5 = 1;
    private static final int FLAG_HAS_CONTINUOUS_DIM5 = 1 << 1;
    private static final int FLAG_HAS_SENTINEL_DIM6 = 1 << 2;
    private static final int FLAG_HAS_CONTINUOUS_DIM6 = 1 << 3;

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

        // Encode directly into flat per-cluster arrays to avoid 3M small short[7] objects.
        System.out.println("[preprocess] Encoding vectors into cluster payloads...");
        short[][] codesByCluster = new short[K][];
        IntStream.range(0, K).parallel().forEach(c -> {
            int[] ids = idsByCluster[c];
            short[] clusterCodes = new short[ids.length * ProductQuantizer.M];
            for (int i = 0; i < ids.length; i++) {
                int offset = i * ProductQuantizer.M;
                pq.encodeFlatInto(vectorsFlat, ids[i], clusterCodes, offset);
            }
            codesByCluster[c] = clusterCodes;
        });
        System.out.println("[preprocess] Finished encoding cluster payloads.");

        System.out.println("[preprocess] Computing cluster metadata...");
        float[] clusterMinValues = new float[K * DIMS];
        float[] clusterMaxValues = new float[K * DIMS];
        float[] clusterRadiusSquared = new float[K];
        int[] clusterFlags = new int[K];
        int[] clusterFraudCounts = new int[K];
        computeClusterMetadata(vectorsFlat, labels, centroids, idsByCluster,
                clusterMinValues, clusterMaxValues, clusterRadiusSquared,
                clusterFlags, clusterFraudCounts);

        // Write data.bin
        System.out.println("[preprocess] Writing " + outputBin + "...");
        dataWriter.write(outputBin, norms, mccRisks, pq,
            centroids, ref.flat(), labels, idsByCluster, codesByCluster,
            clusterMinValues, clusterMaxValues, clusterRadiusSquared,
            clusterFlags, clusterFraudCounts);
        System.out.println("[preprocess] Done.");
    }

    private void computeClusterMetadata(float[] vectorsFlat,
                                        byte[] labels,
                                        float[][] centroids,
                                        int[][] idsByCluster,
                                        float[] clusterMinValues,
                                        float[] clusterMaxValues,
                                        float[] clusterRadiusSquared,
                                        int[] clusterFlags,
                                        int[] clusterFraudCounts) {
        IntStream.range(0, idsByCluster.length).parallel().forEach(c -> {
            int base = c * DIMS;
            for (int d = 0; d < DIMS; d++) {
                clusterMinValues[base + d] = Float.POSITIVE_INFINITY;
                clusterMaxValues[base + d] = Float.NEGATIVE_INFINITY;
            }

            int[] ids = idsByCluster[c];
            float radiusSquared = 0f;
            int flags = 0;
            int fraudCount = 0;
            for (int id : ids) {
                int vectorBase = id * DIMS;
                float centroidDistance = 0f;
                for (int d = 0; d < DIMS; d++) {
                    float value = vectorsFlat[vectorBase + d];
                    if (value < clusterMinValues[base + d]) clusterMinValues[base + d] = value;
                    if (value > clusterMaxValues[base + d]) clusterMaxValues[base + d] = value;

                    float diff = value - centroids[c][d];
                    centroidDistance += diff * diff;
                }
                if (centroidDistance > radiusSquared) radiusSquared = centroidDistance;
                if (labels[id] == 1) fraudCount++;

                flags = updateSentinelFlags(flags, vectorsFlat[vectorBase + 5], FLAG_HAS_SENTINEL_DIM5, FLAG_HAS_CONTINUOUS_DIM5);
                flags = updateSentinelFlags(flags, vectorsFlat[vectorBase + 6], FLAG_HAS_SENTINEL_DIM6, FLAG_HAS_CONTINUOUS_DIM6);
            }

            if (ids.length == 0) {
                for (int d = 0; d < DIMS; d++) {
                    clusterMinValues[base + d] = 0f;
                    clusterMaxValues[base + d] = 0f;
                }
            }

            clusterRadiusSquared[c] = radiusSquared;
            clusterFlags[c] = flags;
            clusterFraudCounts[c] = fraudCount;
        });
    }

    private static int updateSentinelFlags(int flags, float value, int sentinelBit, int continuousBit) {
        if (value == -1f) return flags | sentinelBit;
        return flags | continuousBit;
    }
}
