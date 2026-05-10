package villani.dev.fraud;

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

    // ── public factory ───────────────────────────────────────────────────────

    public static TransactionRequest parse(String j) {
        String id = extractStr(j, 0, "id");

        int txStart          = sectionStart(j, 0, "transaction");
        float txAmount       = extractFloat(j, txStart, "amount");
        int txInstallments   = extractInt(j, txStart, "installments");
        String reqAtStr      = extractStr(j, txStart, "requested_at");
        int txHour           = tsHour(reqAtStr);
        int txDow            = tsDayOfWeek(reqAtStr);
        long txEpoch         = tsEpochSeconds(reqAtStr);

        int custStart        = sectionStart(j, 0, "customer");
        float custAvgAmount  = extractFloat(j, custStart, "avg_amount");
        int custTxCount      = extractInt(j, custStart, "tx_count_24h");

        int merchStart       = sectionStart(j, 0, "merchant");
        String merchId       = extractStr(j, merchStart, "id");
        int mccCode          = extractIntStr(j, merchStart, "mcc");
        float merchAvg       = extractFloat(j, merchStart, "avg_amount");

        boolean unknownMerchant = !merchantIsKnown(j, custStart, "known_merchants", merchId);

        int termStart        = sectionStart(j, 0, "terminal");
        boolean isOnline     = extractBool(j, termStart, "is_online");
        boolean cardPresent  = extractBool(j, termStart, "card_present");
        float kmFromHome     = extractFloat(j, termStart, "km_from_home");

        LastTransactionData lastTx = null;
        int ltIdx = j.indexOf("\"last_transaction\"");
        if (ltIdx >= 0) {
            int colon = j.indexOf(':', ltIdx) + 1;
            while (j.charAt(colon) == ' ' || j.charAt(colon) == '\n' || j.charAt(colon) == '\r') colon++;
            if (j.charAt(colon) == '{') {
                long lastEpoch      = tsEpochSeconds(extractStr(j, colon, "timestamp"));
                float kmFromCurrent = extractFloat(j, colon, "km_from_current");
                lastTx = new LastTransactionData(lastEpoch, kmFromCurrent);
            }
        }

        //TODO flatten this structure
        return new TransactionRequest(id,
                new TransactionData(txAmount, txInstallments, txHour, txDow, txEpoch),
                new CustomerData(custAvgAmount, custTxCount, unknownMerchant),
                new MerchantData(merchId, mccCode, merchAvg),
                new TerminalData(isOnline, cardPresent, kmFromHome),
                lastTx);
    }

    // ── substring helpers ────────────────────────────────────────────────────

    private static int sectionStart(String j, int from, String key) {
        int k = j.indexOf('"' + key + '"', from);
        int colon = j.indexOf(':', k);
        return j.indexOf('{', colon);
    }

    private static String extractStr(String j, int from, String key) {
        int k = j.indexOf('"' + key + '"', from);
        int colon = j.indexOf(':', k);
        int q1 = j.indexOf('"', colon + 1);
        int q2 = j.indexOf('"', q1 + 1);
        return j.substring(q1 + 1, q2);
    }

    private static float extractFloat(String j, int from, String key) {
        int k = j.indexOf('"' + key + '"', from);
        int start = j.indexOf(':', k) + 1;
        while (j.charAt(start) == ' ') start++;
        int end = start;
        char c;
        while (end < j.length() && (c = j.charAt(end)) != ',' && c != '}' && c != '\n' && c != '\r') end++;
        return Float.parseFloat(j.substring(start, end).trim());
    }

    private static int extractInt(String j, int from, String key) {
        return (int) extractFloat(j, from, key);
    }

    private static boolean extractBool(String j, int from, String key) {
        int k = j.indexOf('"' + key + '"', from);
        int start = j.indexOf(':', k) + 1;
        while (j.charAt(start) == ' ') start++;
        return j.charAt(start) == 't';
    }

    private static boolean merchantIsKnown(String j, int from, String key, String targetId) {
        int k = j.indexOf('"' + key + '"', from);
        int bracket = j.indexOf('[', k);
        int end = j.indexOf(']', bracket);
        int pos = bracket + 1;
        int tLen = targetId.length();
        while (pos < end) {
            int q1 = j.indexOf('"', pos);
            if (q1 < 0 || q1 >= end) break;
            int q2 = j.indexOf('"', q1 + 1);
            if (q2 - q1 - 1 == tLen && j.regionMatches(q1 + 1, targetId, 0, tLen)) return true;
            pos = q2 + 1;
        }
        return false;
    }

    private static int extractIntStr(String j, int from, String key) {
        int k = j.indexOf('"' + key + '"', from);
        int q1 = j.indexOf('"', j.indexOf(':', k) + 1) + 1;
        int v = 0;
        char c;
        while ((c = j.charAt(q1++)) != '"') v = v * 10 + (c - '0');
        return v;
    }

    // ── fast timestamp parsing (no OffsetDateTime allocation) ────────────────
    // Format: "2026-03-11T18:45:53Z" (always UTC, always this length)

    private static int tsHour(String s) {
        return (s.charAt(11) - '0') * 10 + (s.charAt(12) - '0');
    }

    private static int tsDayOfWeek(String s) {
        int y = digits(s, 0, 4);
        int m = digits(s, 5, 7);
        int d = digits(s, 8, 10);
        final int[] t = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
        // ajuste para anos bissextos
        if (m < 3) y--;

        int dow = (y + y / 4 - y / 100 + y / 400 + t[m - 1] + d) % 7;
        //return dow == 0 ? 7 : dow;
        // Zeller: 0=Sáb, 1=Dom, 2=Seg, 3=Ter, 4=Qua, 5=Qui, 6=Sex
        // Rinha: 0=Seg, 1=Ter, 2=Qua, 3=Qui, 4=Sex, 5=Sáb, 6=Dom
        return switch (dow) {
            case 1 -> 0; // Seg -> 0
            case 2 -> 1; // Ter -> 1
            case 3 -> 2; // Qua -> 2
            case 4 -> 3; // Qui -> 3
            case 5 -> 4; // Sex -> 4
            case 6 -> 5; // Sáb -> 5
            case 0 -> 6; // Dom -> 6
            default -> 0;
        };
    }

    static long tsEpochSeconds(String s) {
        int y   = digits(s, 0, 4);
        int m   = digits(s, 5, 7);
        int d   = digits(s, 8, 10);
        int h   = digits(s, 11, 13);
        int min = digits(s, 14, 16);
        int sec = digits(s, 17, 19);
        if (m <= 2) { y--; m += 9; } else { m -= 3; }
        long era = (y >= 0 ? y : y - 399) / 400;
        int  yoe = (int)(y - era * 400);
        int  doy = (153 * m + 2) / 5 + d - 1;
        int  doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        long days = era * 146097L + doe - 719468L;
        return days * 86400L + h * 3600L + min * 60L + sec;
    }

    private static int digits(String s, int start, int end) {
        int v = 0;
        for (int i = start; i < end; i++) v = v * 10 + (s.charAt(i) - '0');
        return v;
    }
}

