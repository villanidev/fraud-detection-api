package villani.dev.vectorsearch.embedding;

import io.helidon.service.registry.Service;
import villani.dev.fraud.TransactionRequest;

/**
 * Converts a TransactionRequest into a 14-dimensional normalized float vector
 * suitable for vector similarity search.
 *
 * Normalization constants index:
 *   [0] max_amount, [1] max_installments, [2] amount_vs_avg_ratio,
 *   [3] max_minutes, [4] max_km, [5] max_tx_count_24h, [6] max_merchant_avg_amount
 */
@Service.Singleton
public class EmbeddingService {

    public float[] embed(TransactionRequest tx, float[] norms, float[] mccRisk) {
        float[] embeddings = new float[14];

        float amount    = tx.transaction().amount();
        float avgAmount = tx.customer().avg_amount();

        // [0] amount
        embeddings[0] = clamp(amount / norms[0]);

        // [1] installments
        embeddings[1] = clamp(tx.transaction().installments() / norms[1]);

        // [2] amount_vs_avg  (guard against zero avg)
        embeddings[2] = avgAmount > 0 ? clamp((amount / avgAmount) / norms[2]) : 0f;

        // [3] hour_of_day  (0–23 UTC)
        embeddings[3] = tx.transaction().hour() / 23.0f;

        // [4] day_of_week  (Mon=0 … Sun=6)
        embeddings[4] = tx.transaction().dayOfWeek() / 6.0f;

        // [5] minutes_since_last_tx, [6] km_from_last_tx — sentinel -1 when null
        TransactionRequest.LastTransactionData last = tx.last_transaction();
        if (last == null) {
            embeddings[5] = -1f;
            embeddings[6] = -1f;
        } else {
            long minutes = (tx.transaction().epochSeconds() - last.epochSeconds()) / 60L;
            embeddings[5] = clamp(minutes / norms[3]);        // norms[3] = max_minutes
            embeddings[6] = clamp(last.km_from_current() / norms[4]); // norms[4] = max_km
        }

        // [7] km_from_home
        embeddings[7] = clamp(tx.terminal().km_from_home() / norms[4]);

        // [8] tx_count_24h
        embeddings[8] = clamp(tx.customer().tx_count_24h() / norms[5]);

        // [9] is_online
        embeddings[9] = tx.terminal().is_online() ? 1f : 0f;

        // [10] card_present
        embeddings[10] = tx.terminal().card_present() ? 1f : 0f;

        // [11] unknown_merchant (1 = unknown, 0 = known) — pre-computed at parse time
        embeddings[11] = tx.customer().unknownMerchant() ? 1f : 0f;

        // [12] mcc_risk — direct int index, no parseInt needed
        embeddings[12] = mccRiskFor(mccRisk, tx.merchant().mccCode());

        // [13] merchant_avg_amount
        embeddings[13] = clamp(tx.merchant().avg_amount() / norms[6]);

        return embeddings;
    }

    public float[] embed(float[] txArray, float[] norms, float[] mccRisk) {
        // [0] amount
        txArray[0] = clamp(txArray[0] / norms[0]);

        // [1] installments
        txArray[1] = clamp(txArray[1] / norms[1]);

        // [2] amount_vs_avg
        txArray[2] = clamp(txArray[2] / norms[2]);

        // [3] hour_of_day
        txArray[3] = txArray[3] / 23.0f;

        // [4] day_of_week
        txArray[4] = txArray[4] / 6.0f;

        // [5] minutes_since_last_tx, [6] km_from_last_tx
        if (txArray[5] != -1f && txArray[6] != -1f) {
            txArray[5] = clamp(txArray[5] / norms[3]);
            txArray[6] = clamp(txArray[6] / norms[4]);
        }

        // [7] km_from_home
        txArray[7] = clamp(txArray[7] / norms[4]);

        // [8] tx_count_24h
        txArray[8] = clamp(txArray[8] / norms[5]);

        // [12] mcc_risk (índice direto, sem parse)
        txArray[12] = txArray[12] >= 0 && txArray[12] < mccRisk.length ? mccRisk[(int) txArray[12]] : 0.5f;

        // [13] merchant_avg_amount
        txArray[13] = clamp(txArray[13] / norms[6]);

        return txArray;
    }

    private static float mccRiskFor(float[] mccRisk, int mccCode) {
        if (mccCode >= 0 && mccCode < mccRisk.length) return mccRisk[mccCode];
        return 0.5f;
    }

    private static float clamp(float x) {
        return x < 0f ? 0f : Math.min(x, 1f);
    }
}
