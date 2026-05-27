package villani.dev.fraud;

import io.helidon.config.Config;
import io.helidon.service.registry.Service;
import villani.dev.health.PerformanceStats;
import villani.dev.vectorsearch.embedding.EmbeddingService;
import villani.dev.vectorsearch.retrieval.VectorStore;

import java.util.concurrent.atomic.AtomicLong;

@Service.Singleton
public class FraudCheckService {

    private static final double SLOW_SEARCH_THRESHOLD_MS = 50.0;

    private final EmbeddingService embeddingService;
    private final VectorStore vectorStore;
    private final PerformanceStats performanceStats;
    private final boolean debug;
    private final int rerankNprobe;
    private final int rerankCandidates;

    private static final String[] DIM_NAMES = {
            "amount", "installments", "amount_vs_avg", "hour_of_day", "day_of_week",
            "minutes_since_last_tx", "km_from_last_tx", "km_from_home",
            "tx_count_24h", "is_online", "card_present", "unknown_merchant",
            "mcc_risk", "merchant_avg_amount"
    };

    // Gray-zone counters — fraudCount ∈ {2,3} where nprobe matters most
    private final AtomicLong totalRequests = new AtomicLong();
    private final AtomicLong grayZoneRequests = new AtomicLong();

    @Service.Inject
    public FraudCheckService(EmbeddingService embeddingService, VectorStore vectorStore,
                             PerformanceStats performanceStats, Config config) {
        this.embeddingService = embeddingService;
        this.vectorStore = vectorStore;
        this.performanceStats = performanceStats;
        this.debug = config.get("app.fraud.debug").asBoolean().orElse(false);
        this.rerankNprobe = config.get("app.fraud.rerank.nprobe").asInt().orElse(32);
        this.rerankCandidates = config.get("app.fraud.rerank.candidates").asInt().orElse(10);
    }

    public long getTotalRequests()    { return totalRequests.get(); }
    public long getGrayZoneRequests() { return grayZoneRequests.get(); }

    public String checkScore(TransactionRequest tx) {
        return checkScore(tx, System.nanoTime());
    }

    public String checkScore(TransactionRequest tx, long requestStartNs) {
        long t0 = System.nanoTime();

        // 1 Embedding
        float[] emb = embeddingService.embed(tx, vectorStore.getNormalization(), vectorStore.getMccRisk());
        long t1 = System.nanoTime();

        // 2 Busca K=5
        int[] neighbors = new int[5];
        float[] distances = new float[5];
        int fraudCount = vectorStore.search(emb, 5, neighbors, distances);
        long t2 = System.nanoTime();

        // 3 Score
        boolean grayZone = fraudCount == 2 || fraudCount == 3;
        float score = fraudCount / 5.0f;
        boolean approved = score < 0.6f;
        long t3 = System.nanoTime();

        totalRequests.incrementAndGet();
        if (grayZone) grayZoneRequests.incrementAndGet();

        long parseNs  = t0 - requestStartNs;
        long embedNs  = t1 - t0;
        long searchNs = t2 - t1;
        long totalNs  = t3 - requestStartNs;

        performanceStats.recordAll(parseNs, embedNs, searchNs, totalNs);

        double searchMs = searchNs / 1_000_000.0;
        if (debug || searchMs > SLOW_SEARCH_THRESHOLD_MS) {
            double parseMs = parseNs / 1_000_000.0;
            double embedMs = embedNs / 1_000_000.0;
            double totalMs = totalNs / 1_000_000.0;
            System.out.printf("[FRAUD] ── tx=%s  (parse=%.3fms  embed=%.3fms  search=%.3fms  total=%.3fms)%s%n",
                tx.id(), parseMs, embedMs, searchMs, totalMs,
                searchMs > SLOW_SEARCH_THRESHOLD_MS ? "  *** SLOW ***" : "");
            if (debug) {
                byte[] labels = vectorStore.getIndexLabels();
                System.out.println("[FRAUD] Embedding:");
                for (int i = 0; i < emb.length; i++) {
                    System.out.printf("[FRAUD]   [%2d] %-22s = % .6f%s%n",
                        i, DIM_NAMES[i], emb[i],
                        (emb[i] == -1f) ? "  (sentinel: no last_tx)" : "");
                }
                System.out.println("[FRAUD] Nearest neighbors (k=5):");
                for (int i = 0; i < neighbors.length; i++) {
                    int id = neighbors[i];
                    String labelStr = (id < 0) ? "EMPTY" : (labels[id] == 1 ? "FRAUD" : "legit");
                    System.out.printf("[FRAUD]   [%d] id=%-8d dist=%.6f  → %s%n", i, id, distances[i], labelStr);
                }
                System.out.printf("[FRAUD] Result: %d/5 fraud  score=%.4f  approved=%b%n%n", fraudCount, score, approved);
            }
        }

        return String.format("{\"approved\":%b,\"fraud_score\":%.4f}", approved, score);
    }

    public String checkScore(float[] txArray) {
        // 1 Embedding
        float[] emb = embeddingService.embed(txArray, vectorStore.getNormalization(), vectorStore.getMccRisk());

        // 2 Busca K=5
        int[] neighbors = new int[5];
        float[] distances = new float[5];
        int fraudCount = vectorStore.search(emb, 5, neighbors, distances);

        // 3 Score
        float score = fraudCount / 5.0f;
        boolean approved = score < 0.6f;

        return String.format("{\"approved\":%b,\"fraud_score\":%.4f}", approved, score);
    }

    public byte[] checkScore2(float[] txArray) {
        // 1 Embedding
        float[] emb = embeddingService.embed(txArray, vectorStore.getNormalization(), vectorStore.getMccRisk());

        // 2 Busca K=5
        int[] neighbors = new int[5];
        float[] distances = new float[5];
        int fraudCount = vectorStore.search(emb, 5, neighbors, distances);
        //System.out.println("final: " +fraudCount);

        if (fraudCount == 3) {
            // Use factory-created index with expanded params (likely a ReRankingVectorIndex)
            villani.dev.vectorsearch.index.VectorIndex rerankIndex = vectorStore.createIndexForBenchmark(rerankNprobe, rerankCandidates);
            int[] neighborsExact = new int[5];
            float[] distancesExact = new float[5];
            int fraudCountExact = rerankIndex.search(emb, 5, neighborsExact, distancesExact);
            if (fraudCountExact != fraudCount) {
                fraudCount = fraudCountExact;
            }
        }

        // 3 Score ta cacheado
        return DecisionResponse.get(fraudCount);
    }
}

