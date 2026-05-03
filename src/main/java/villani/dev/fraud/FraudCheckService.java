package villani.dev.fraud;

import io.helidon.service.registry.Service;
import villani.dev.vectorsearch.embedding.EmbeddingService;
import villani.dev.vectorsearch.retrieval.VectorStore;

@Service.Singleton
public class FraudCheckService {

    private final EmbeddingService embeddingService;
    private final VectorStore vectorStore;

    @Service.Inject
    public FraudCheckService(EmbeddingService embeddingService, VectorStore vectorStore) {
        this.embeddingService = embeddingService;
        this.vectorStore = vectorStore;
    }

    private static final String[] DIM_NAMES = {
        "amount", "installments", "amount_vs_avg", "hour_of_day", "day_of_week",
        "minutes_since_last_tx", "km_from_last_tx", "km_from_home",
        "tx_count_24h", "is_online", "card_present", "unknown_merchant",
        "mcc_risk", "merchant_avg_amount"
    };

    private static final boolean DEBUG = false;

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
        float score = fraudCount / 5.0f;
        boolean approved = score < 0.6f;
        long t3 = System.nanoTime();

        double parseMs  = (t0 - requestStartNs) / 1_000_000.0;
        double embedMs  = (t1 - t0) / 1_000_000.0;
        double searchMs = (t2 - t1) / 1_000_000.0;
        double totalMs  = (t3 - requestStartNs) / 1_000_000.0;

        if (DEBUG) {
            byte[] labels = vectorStore.getIndexLabels();
            System.out.printf("[FRAUD] ── tx=%s  (parse=%.3fms  embed=%.3fms  search=%.3fms  total=%.3fms) ───%n",
                tx.id(), parseMs, embedMs, searchMs, totalMs);
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

        return String.format("{\"approved\":%b,\"fraud_score\":%.4f}", approved, score);
    }
}
