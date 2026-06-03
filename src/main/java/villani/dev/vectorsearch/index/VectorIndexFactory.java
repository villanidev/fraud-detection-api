package villani.dev.vectorsearch.index;

import io.helidon.config.Config;
import io.helidon.service.registry.Service;
import villani.dev.vectorsearch.index.strategies.bruteforce.BruteForceIndex;
import villani.dev.vectorsearch.index.strategies.hnsw.HNSWIndex;
import villani.dev.vectorsearch.index.strategies.ivfpq.IVFPQIndex;
import villani.dev.vectorsearch.index.strategies.ivfpq.ProductQuantizer;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;

/**
 * Factory that reads the index strategy from config and creates the appropriate VectorIndex.
 *
 * Config keys (all under app.vector-search):
 *   index      — "brute_force" | "ivf_pq" | "hnsw"  (default: "ivf_pq")
 *   nprobe     — clusters probed per query             (default: 16)
 *   candidates — coarse results before reranking       (default: 50)
 *   rerank     — wrap result in ReRankingVectorIndex   (default: true)
 *
 * Overridable via env vars, e.g.:
 *   APP_VECTOR__SEARCH_INDEX=brute_force
 *   APP_VECTOR__SEARCH_RERANK=false
 *
 * Pattern: Factory Method (GoF) — centralises creation, callers depend only on VectorIndex.
 *          Supports OCP: new strategies added here without touching VectorStore or search path.
 */
@Service.Singleton
public class VectorIndexFactory {

    private static final String DEFAULT_INDEX = "ivf_pq";
    private static final String DEFAULT_MODE = "scalar";
    private static final int DEFAULT_NPROBE = 4;
    private static final int DEFAULT_CANDIDATES = 10;
    private static final boolean DEFAULT_RERANK = true;
    private static final int DEFAULT_RERANK_NPROBE = 32;
    private static final int DEFAULT_RERANK_CANDIDATES = 10;
    private static final boolean DEFAULT_PRUNE_ENABLED = true;
    private static final double DEFAULT_PRUNE_EPSILON = 0.02d;
    private static final double DEFAULT_PRUNE_SENTINEL_PENALTY = 4.0d;

    private final String indexType;
    private final int nprobe;
    private final int candidates;
    private final boolean rerank;
    private final int rerankNprobe;
    private final int rerankCandidates;
    private final String mode;
    private final boolean scalarScanEnabled;
    private final boolean pruneEnabled;

    @Service.Inject
    public VectorIndexFactory(Config config) {
        Config vs = config.get("app.vector-search");
        this.indexType = vs.get("index").asString().orElse(DEFAULT_INDEX);
        System.out.println("Configured vector search index: " + indexType);
        this.nprobe = vs.get("nprobe").asInt().orElse(DEFAULT_NPROBE);
        System.out.println("Configured nprobe: " + nprobe);
        this.candidates = vs.get("candidates").asInt().orElse(DEFAULT_CANDIDATES);
        System.out.println("Configured candidates: " + candidates);
        this.mode = vs.get("mode").asString().orElse(DEFAULT_MODE);
        if (!"pq".equals(mode) && !"scalar".equals(mode)) {
            throw new IllegalArgumentException("Unknown vector-search mode: '" + mode + "'. Valid values: pq, scalar");
        }
        this.scalarScanEnabled = "scalar".equals(mode);
        System.out.println("Configured vector search mode: " + mode);
        Config pruneNode = vs.get("prune");
        this.pruneEnabled = pruneNode.get("enabled").asBoolean().orElse(DEFAULT_PRUNE_ENABLED);
        System.out.println("Configured cluster prune: " + pruneEnabled);

        Config rerankNode = vs.get("rerank");
        this.rerank = rerankNode.get("enabled").asBoolean().orElse(DEFAULT_RERANK);
        System.out.println("Configured rerank: " + rerank);
        this.rerankNprobe = rerankNode.get("nprobe").asInt().orElse(DEFAULT_RERANK_NPROBE);
        System.out.println("Configured rerank nprobe: " + rerankNprobe);
        this.rerankCandidates = rerankNode.get("candidates").asInt().orElse(DEFAULT_RERANK_CANDIDATES);
        System.out.println("Configured rerank candidates: " + rerankCandidates);
    }

    /**
     * Creates and returns the configured VectorIndex, optionally wrapping it
     * in a ReRankingVectorIndex decorator.
     *
     * @param centroids      IVF centroids [K][14]
     * @param idsByCluster   inverted lists [K][count]
    * @param codesByCluster PQ codes per cluster [K][count][M]
    * @param vectors        original vectors as flat array (row-major) (used by BruteForce only)
     * @param labels         reference labels [N] — 0=legit, 1=fraud
     * @param pq             trained ProductQuantizer (used by IVF-PQ)
     * @param vectorsChannel    memory-mapped data.bin (for reranking; may be null if rerank=false)
     * @param vectorsOffset  byte offset of vectors section in data.bin
     * @param vectorCount    total number of reference vectors
     */
    public VectorIndex create(float[][] centroids,
                              int[] clusterSizes,
                              int[] idsStartIndex,
                              IntBuffer idsPayload,
                              int[] codesStartIndex,
                              ShortBuffer codesPayload,
                              float[] vectors,
                              byte[] labels,
                              ProductQuantizer pq,
                              float[] clusterMinValues,
                              float[] clusterMaxValues,
                              int[] clusterFlags,
                              ByteBuffer scalarQuantizedPayload,
                              int scalarQuantizedStrideBytes,
                              FileChannel vectorsChannel,   // substitui MappedByteBuffer
                              long vectorsOffset,
                              int vectorCount) {

        VectorIndex base = switch (indexType) {
            case "brute_force" -> new BruteForceIndex(vectors, labels);
            case "ivf_pq" -> new IVFPQIndex(centroids, clusterSizes, idsStartIndex, idsPayload, codesStartIndex, codesPayload,
                                              labels, pq, clusterMinValues, clusterMaxValues,
                                              clusterFlags, scalarQuantizedPayload,
                                              scalarQuantizedStrideBytes, scalarScanEnabled,
                                              pruneEnabled, (float) DEFAULT_PRUNE_EPSILON, (float) DEFAULT_PRUNE_SENTINEL_PENALTY,
                                              nprobe, candidates);
            case "hnsw" -> new HNSWIndex(labels);
            default -> throw new IllegalArgumentException(
                    "Unknown vector-search index: '" + indexType + "'. Valid values: brute_force, ivf_pq, hnsw");
        };

        if (rerank && vectorsChannel != null && !(base instanceof BruteForceIndex)) {
            return new ReRankingVectorIndex(base, vectorsChannel, vectorsOffset, candidates, vectorCount, rerankNprobe, rerankCandidates);
        }

        return base;
    }

    /**
     * Cria o índice com parâmetros de busca customizados (usado no benchmark).
     */
    public VectorIndex create(float[][] centroids,
                      int[] clusterSizes,
                      int[] idsStartIndex,
                      IntBuffer idsPayload,
                      int[] codesStartIndex,
                      ShortBuffer codesPayload,
                              float[] vectors,
                              byte[] labels,
                              ProductQuantizer pq,
                              float[] clusterMinValues,
                              float[] clusterMaxValues,
                              int[] clusterFlags,
                      ByteBuffer scalarQuantizedPayload,
                              int scalarQuantizedStrideBytes,
                              FileChannel vectorsChannel,
                              long vectorsOffset,
                              int vectorCount,
                              int nprobe,
                              int candidates) {
        return create(centroids, clusterSizes, idsStartIndex, idsPayload, codesStartIndex, codesPayload,
                vectors, labels, pq,
                clusterMinValues, clusterMaxValues, clusterFlags,
                scalarQuantizedPayload, scalarQuantizedStrideBytes,
                vectorsChannel, vectorsOffset, vectorCount,
                nprobe, candidates, scalarScanEnabled, pruneEnabled);
    }

    public VectorIndex create(float[][] centroids,
                      int[] clusterSizes,
                      int[] idsStartIndex,
                      IntBuffer idsPayload,
                      int[] codesStartIndex,
                      ShortBuffer codesPayload,
                              float[] vectors,
                              byte[] labels,
                              ProductQuantizer pq,
                              float[] clusterMinValues,
                              float[] clusterMaxValues,
                              int[] clusterFlags,
                      ByteBuffer scalarQuantizedPayload,
                              int scalarQuantizedStrideBytes,
                              FileChannel vectorsChannel,
                              long vectorsOffset,
                              int vectorCount,
                              int nprobe,
                              int candidates,
                              boolean scalarScanEnabled,
                              boolean pruneEnabled) {

        VectorIndex base = switch (indexType) {
            case "brute_force" -> new BruteForceIndex(vectors, labels);
                case "ivf_pq" -> new IVFPQIndex(centroids, clusterSizes, idsStartIndex, idsPayload, codesStartIndex, codesPayload,
                    labels, pq, clusterMinValues, clusterMaxValues,
                    clusterFlags, scalarQuantizedPayload,
                    scalarQuantizedStrideBytes, scalarScanEnabled,
                    pruneEnabled, (float) DEFAULT_PRUNE_EPSILON, (float) DEFAULT_PRUNE_SENTINEL_PENALTY,
                    nprobe, candidates);
            case "hnsw" -> new HNSWIndex(labels);
            default -> throw new IllegalArgumentException("Unknown index type: " + indexType);
        };

        if (rerank && vectorsChannel != null && !(base instanceof BruteForceIndex)) {
            return new ReRankingVectorIndex(base, vectorsChannel, vectorsOffset, candidates, vectorCount, rerankNprobe, rerankCandidates);
        }

        return base;
    }

    public VectorIndex create(float[][] centroids,
                              int[] clusterSizes,
                              int[] idsStartIndex,
                              IntBuffer idsPayload,
                              int[] codesStartIndex,
                              ShortBuffer codesPayload,
                              float[] vectors,
                              byte[] labels,
                              ProductQuantizer pq,
                              float[] clusterMinValues,
                              float[] clusterMaxValues,
                              int[] clusterFlags,
                              ByteBuffer scalarQuantizedPayload,
                              int scalarQuantizedStrideBytes,
                              FileChannel vectorsChannel,
                              long vectorsOffset,
                              int vectorCount,
                              int nprobe,
                              int candidates,
                              String mode,
                              boolean pruneEnabled) {
        boolean scalarEnabled = switch (mode) {
            case "pq" -> false;
            case "scalar" -> true;
            default -> throw new IllegalArgumentException("Unknown vector-search mode: '" + mode + "'. Valid values: pq, scalar");
        };

        return create(centroids, clusterSizes, idsStartIndex, idsPayload, codesStartIndex, codesPayload,
                vectors, labels, pq,
                clusterMinValues, clusterMaxValues, clusterFlags,
                scalarQuantizedPayload, scalarQuantizedStrideBytes,
                vectorsChannel, vectorsOffset, vectorCount,
                nprobe, candidates, scalarEnabled, pruneEnabled);
    }

    public String getIndexType()  { return indexType; }
    public boolean isRerank()     { return rerank; }
    public boolean isBruteForce() { return "brute_force".equals(indexType); }
    public int getCandidates()    { return candidates; }
}
