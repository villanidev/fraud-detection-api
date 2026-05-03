package villani.dev;

import org.junit.jupiter.api.Test;
import villani.dev.fraud.TransactionRequest;
import villani.dev.vectorsearch.embedding.EmbeddingService;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;

/**
 * Validates EmbeddingService math against the canonical example in DETECTION_RULES.md.
 *
 * Reference:
 *   payload  → tx-1329056812
 *   expected → [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
 *   result   → approved=true, score=0.0
 *
 * Normalization constants (from normalization.json):
 *   [0]=10000, [1]=12, [2]=10, [3]=1440, [4]=1000, [5]=20, [6]=10000
 *
 * MCC risk (from mcc_risk.json):
 *   5411 → 0.15
 */
class EmbeddingServiceTest {

    private static final float DELTA = 0.0002f;

    // normalization.json constants
    private static final float[] NORMS = {10000f, 12f, 10f, 1440f, 1000f, 20f, 10000f};

    // mcc_risk.json — sparse table, only entries present in mcc_risk.json
    private static final float[] MCC_RISK = buildMccRisk();

    private final EmbeddingService service = new EmbeddingService();

    // ──────────────────────────────────────────────────────────────────────────
    // Test 1: canonical example from DETECTION_RULES.md
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    void canonicalEmbedding_tx1329056812() {
        TransactionRequest tx = new TransactionRequest(
            "tx-1329056812",
            new TransactionRequest.TransactionData(
                41.12f, 2,
                18, 3,          // 2026-03-11T18:45:53Z → hour=18, dow=3 (Wednesday)
                1774454753L     // epoch seconds (unused: last_tx=null)
            ),
            new TransactionRequest.CustomerData(
                82.24f, 3,
                false          // MERC-016 is in known_merchants → unknownMerchant=false
            ),
            new TransactionRequest.MerchantData("MERC-016", 5411, 60.25f),
            new TransactionRequest.TerminalData(false, true, 29.2331036248f),
            null  // last_transaction = null
        );

        float[] emb = service.embed(tx, NORMS, MCC_RISK);

        System.out.println("=== EmbeddingServiceTest: canonical example ===");

        String[] names = {"amount","installments","amount_vs_avg","hour_of_day","day_of_week",
                          "minutes_last_tx","km_last_tx","km_from_home","tx_count_24h",
                          "is_online","card_present","unknown_merchant","mcc_risk","merchant_avg"};
        float[] expected = {0.0041f, 0.1667f, 0.05f, 0.7826f, 0.3333f, -1f, -1f,
                            0.0292f, 0.15f, 0f, 1f, 0f, 0.15f, 0.006f};
        for (int i = 0; i < emb.length; i++) {
            System.out.printf("  [%2d] %-20s got=% .6f  expected=% .4f  %s%n",
                i, names[i], emb[i], expected[i],
                Math.abs(emb[i] - expected[i]) <= DELTA ? "OK" : "MISMATCH !!!");
        }

        // [0] amount = 41.12 / 10000
        assertDim(emb, 0,  0.0041f, "amount");
        // [1] installments = 2 / 12
        assertDim(emb, 1,  0.1667f, "installments");
        // [2] amount_vs_avg = (41.12/82.24) / 10 = 0.5/10
        assertDim(emb, 2,  0.05f,   "amount_vs_avg");
        // [3] hour_of_day = 18 / 23
        assertDim(emb, 3,  0.7826f, "hour_of_day");
        // [4] day_of_week — 2026-03-11 is Wednesday (getValue=3), (3-1)/6 = 0.3333
        assertDim(emb, 4,  0.3333f, "day_of_week");
        // [5] sentinel -1 (no last_transaction)
        assertDim(emb, 5, -1f,      "minutes_since_last_tx (sentinel)");
        // [6] sentinel -1 (no last_transaction)
        assertDim(emb, 6, -1f,      "km_from_last_tx (sentinel)");
        // [7] km_from_home = 29.2331 / 1000
        assertDim(emb, 7,  0.0292f, "km_from_home");
        // [8] tx_count_24h = 3 / 20
        assertDim(emb, 8,  0.15f,   "tx_count_24h");
        // [9] is_online = false → 0
        assertDim(emb, 9,  0f,      "is_online");
        // [10] card_present = true → 1
        assertDim(emb, 10, 1f,      "card_present");
        // [11] unknown_merchant: MERC-016 IS in known_merchants → 0
        assertDim(emb, 11, 0f,      "unknown_merchant");
        // [12] mcc_risk: 5411 → 0.15
        assertDim(emb, 12, 0.15f,   "mcc_risk");
        // [13] merchant_avg_amount = 60.25 / 10000
        assertDim(emb, 13, 0.006f,  "merchant_avg_amount");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Test 2: last_transaction present
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    void embeddingWithLastTransaction() {
        // tx-3576980410 from example-payloads.json
        // requested_at=2026-03-11T20:23:35Z, last_tx=2026-03-11T14:58:35Z
        // minutes_between = 325 min → 325/1440 = 0.2257
        // km_from_current = 18.8626 → 18.8626/1000 = 0.0189
        TransactionRequest tx = new TransactionRequest(
            "tx-3576980410",
            new TransactionRequest.TransactionData(
                384.88f, 3,
                20, 3,          // 2026-03-11T20:23:35Z → hour=20, dow=3 (Wednesday)
                1774460615L     // epoch seconds
            ),
            new TransactionRequest.CustomerData(
                769.76f, 3,
                false          // MERC-001 is in known_merchants → unknownMerchant=false
            ),
            new TransactionRequest.MerchantData("MERC-001", 5912, 298.95f),
            new TransactionRequest.TerminalData(false, true, 13.7090520965f),
            new TransactionRequest.LastTransactionData(
                1774441115L,    // 2026-03-11T14:58:35Z → (1774460615-1774441115)/60 = 325 min
                18.8626479774f
            )
        );

        float[] emb = service.embed(tx, NORMS, MCC_RISK);

        System.out.println("=== EmbeddingServiceTest: with last_transaction ===");

        // minutes_since_last_tx = 325 min → 325/1440 ≈ 0.2257
        assertDim(emb, 5, 0.2257f, "minutes_since_last_tx");
        // km_from_last_tx = 18.8626 / 1000 ≈ 0.0189
        assertDim(emb, 6, 0.0189f, "km_from_last_tx");
        // mcc_risk: 5912 → 0.20
        assertDim(emb, 12, 0.20f, "mcc_risk[5912]");
        // amount_vs_avg: (384.88/769.76)/10 = 0.5/10 = 0.05
        assertDim(emb, 2, 0.05f, "amount_vs_avg");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Test 3: unknown merchant, unknown MCC, clamp
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    void unknownMerchantAndMcc_usesDefaults() {
        TransactionRequest tx = new TransactionRequest(
            "tx-test",
            new TransactionRequest.TransactionData(
                50000f, 15,    // amount > max → clamped to 1.0; installments > max → clamped
                0, 1,          // 2026-01-05T00:00:00Z → hour=0, dow=1 (Monday)
                1767571200L    // epoch seconds (unused: last_tx=null)
            ),
            new TransactionRequest.CustomerData(100f, 25, true),  // MERC-ZZZ not in [MERC-AAA] → unknown
            new TransactionRequest.MerchantData("MERC-ZZZ", 9999, 9999f),
            new TransactionRequest.TerminalData(true, false, 2000f),
            null
        );

        float[] emb = service.embed(tx, NORMS, MCC_RISK);

        // [0] amount clamped: 50000/10000 = 5.0 → 1.0
        assertDim(emb, 0, 1.0f, "amount clamped");
        // [1] installments clamped: 15/12 = 1.25 → 1.0
        assertDim(emb, 1, 1.0f, "installments clamped");
        // [7] km_from_home clamped: 2000/1000 = 2.0 → 1.0
        assertDim(emb, 7, 1.0f, "km_from_home clamped");
        // [8] tx_count_24h clamped: 25/20 = 1.25 → 1.0
        assertDim(emb, 8, 1.0f, "tx_count_24h clamped");
        // [9] is_online = true → 1
        assertDim(emb, 9, 1.0f, "is_online=true");
        // [10] card_present = false → 0
        assertDim(emb, 10, 0f, "card_present=false");
        // [11] unknown_merchant: MERC-ZZZ not in [MERC-AAA] → 1
        assertDim(emb, 11, 1.0f, "unknown_merchant=true");
        // [12] mcc_risk: 9999 not in table → default 0.5
        assertDim(emb, 12, 0.5f, "mcc_risk unknown → 0.5");
        // [3] midnight → 0/23 = 0
        assertDim(emb, 3, 0f, "hour_of_day midnight");
        // [4] 2026-01-05 is Monday (getValue=1) → (1-1)/6 = 0
        assertDim(emb, 4, 0f, "day_of_week monday");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────────

    private static void assertDim(float[] emb, int idx, float expected, String name) {
        assertThat("dim[" + idx + "] " + name,
            (double) emb[idx], closeTo(expected, DELTA));
    }

    private static float[] buildMccRisk() {
        // Default 0.5 everywhere
        float[] table = new float[10_000];
        java.util.Arrays.fill(table, 0.5f);
        // Values from mcc_risk.json
        table[5411] = 0.15f;
        table[5812] = 0.30f;
        table[5912] = 0.20f;
        table[5944] = 0.45f;
        table[7801] = 0.80f;
        table[7802] = 0.75f;
        table[7995] = 0.85f;
        table[4511] = 0.35f;
        table[5311] = 0.25f;
        table[5999] = 0.50f;
        return table;
    }
}
