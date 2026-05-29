package villani.dev.vectorsearch.index;

import io.helidon.config.Config;
import io.helidon.service.registry.Service;
import villani.dev.vectorsearch.index.strategies.bruteforce.BruteForceIndex;
import villani.dev.vectorsearch.index.strategies.hnsw.HNSWIndex;
import villani.dev.vectorsearch.index.strategies.ivfpq.IVFPQIndex;
import villani.dev.vectorsearch.index.strategies.ivfpq.ProductQuantizer;

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
    private static final int DEFAULT_NPROBE = 2;
    private static final int DEFAULT_CANDIDATES = 10;
    private static final boolean DEFAULT_RERANK = true;
    private static final int DEFAULT_RERANK_NPROBE = 8;
    private static final int DEFAULT_RERANK_CANDIDATES = 10;

    private final String indexType;
    private final int nprobe;
    private final int candidates;
    private final boolean rerank;
    private final int rerankNprobe;
    private final int rerankCandidates;

    @Service.Inject
    public VectorIndexFactory(Config config) {
        Config vs = config.get("app.vector-search");
        this.indexType  = vs.get("index").asString().orElse(DEFAULT_INDEX);
        System.out.println("Configured vector search index: " + indexType);
        this.nprobe     = vs.get("nprobe").asInt().orElse(DEFAULT_NPROBE);
        System.out.println("Configured nprobe: " + nprobe);
        this.candidates = vs.get("candidates").asInt().orElse(DEFAULT_CANDIDATES);
        System.out.println("Configured candidates: " + candidates);

        // Expect nested `rerank` config with keys: enabled, nprobe, candidates
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
                              int[][] idsByCluster,
                              short[][] codesByCluster,
                              float[] vectors,
                              byte[] labels,
                              ProductQuantizer pq,
                              FileChannel vectorsChannel,   // substitui MappedByteBuffer
                              long vectorsOffset,
                              int vectorCount,
                              float[] bboxMin,
                              float[] bboxMax,
                              int[] clusterOfId) {

        VectorIndex base = switch (indexType) {
            case "brute_force" -> new BruteForceIndex(vectors, labels);
            case "ivf_pq" -> new IVFPQIndex(centroids, idsByCluster, codesByCluster,
                                              labels, pq, nprobe, candidates);
            case "hnsw" -> new HNSWIndex(labels);
            default -> throw new IllegalArgumentException(
                    "Unknown vector-search index: '" + indexType + "'. Valid values: brute_force, ivf_pq, hnsw");
        };

        if (rerank && vectorsChannel != null && !(base instanceof BruteForceIndex)) {
            return new ReRankingVectorIndex(base, vectorsChannel, vectorsOffset, vectorCount, rerankNprobe, rerankCandidates, bboxMin, bboxMax, clusterOfId);
        }

        return base;
    }

    /**
     * Cria o índice com parâmetros de busca customizados (usado no benchmark).
     */
    public VectorIndex create(float[][] centroids,
                              int[][] idsByCluster,
                              short[][] codesByCluster,
                              float[] vectors,
                              byte[] labels,
                              ProductQuantizer pq,
                              FileChannel vectorsChannel,
                              long vectorsOffset,
                              int vectorCount,
                              int nprobe,
                              int candidates) {

        VectorIndex base = switch (indexType) {
            case "brute_force" -> new BruteForceIndex(vectors, labels);
                case "ivf_pq" -> new IVFPQIndex(centroids, idsByCluster, codesByCluster,
                    labels, pq, nprobe, candidates);
            case "hnsw" -> new HNSWIndex(labels);
            default -> throw new IllegalArgumentException("Unknown index type: " + indexType);
        };

        if (rerank && vectorsChannel != null && !(base instanceof BruteForceIndex)) {
            return new ReRankingVectorIndex(base, vectorsChannel, vectorsOffset, vectorCount, rerankNprobe, rerankCandidates, null, null, null);
        }

        return base;
    }

    public String getIndexType()  { return indexType; }
    public boolean isRerank()     { return rerank; }
    public boolean isBruteForce() { return "brute_force".equals(indexType); }
    public int getCandidates()    { return candidates; }
}
