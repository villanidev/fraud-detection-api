package villani.dev.fraud;

import java.util.List;

public record TransactionRequest(
        String id,
        TransactionData transaction,
        CustomerData customer,
        MerchantData merchant,
        TerminalData terminal,
        LastTransactionData last_transaction
) {
    /**
     * Timestamp fields are pre-parsed at the HTTP layer to avoid OffsetDateTime allocation.
     * epochSeconds is only needed when last_transaction is present (minutes_since_last_tx).
     */
    public record TransactionData(
            float amount,
            int installments,
            int hour,
            int dayOfWeek,
            long epochSeconds
    ) {}

    /**
     * known_merchants resolved to a single boolean at parse time — no list stored.
     */
    public record CustomerData(
            float avg_amount,
            int tx_count_24h,
            boolean unknownMerchant
    ) {}

    public record MerchantData(
            String id,
            int mccCode,
            float avg_amount
    ) {}

    public record TerminalData(
            boolean is_online,
            boolean card_present,
            float km_from_home
    ) {}

    public record LastTransactionData(
            long epochSeconds,
            float km_from_current
    ) {}
}
