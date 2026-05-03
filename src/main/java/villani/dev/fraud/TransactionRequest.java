package villani.dev.fraud;

import java.time.OffsetDateTime;
import java.util.List;

public record TransactionRequest(
        String id,
        TransactionData transaction,
        CustomerData customer,
        MerchantData merchant,
        TerminalData terminal,
        LastTransactionData last_transaction
) {
    public record TransactionData(
            float amount,
            int installments,
            OffsetDateTime requested_at
    ) {}

    public record CustomerData(
            float avg_amount,
            int tx_count_24h,
            List<String> known_merchants
    ) {}

    public record MerchantData(
            String id,
            String mcc,
            float avg_amount
    ) {}

    public record TerminalData(
            boolean is_online,
            boolean card_present,
            float km_from_home
    ) {}

    public record LastTransactionData(
            OffsetDateTime timestamp,
            float km_from_current
    ) {}
}
