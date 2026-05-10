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

    public static TransactionRequest parse(String json) {
        String id = extractStr(json, 0, "id");

        int txStart          = sectionStart(json, 0, "transaction");
        float txAmount       = extractFloat(json, txStart, "amount");
        int txInstallments   = extractInt(json, txStart, "installments");
        String reqAtStr      = extractStr(json, txStart, "requested_at");
        int txHour           = tsHour(reqAtStr);
        int txDow            = tsDayOfWeek(reqAtStr);
        long txEpoch         = tsEpochSeconds(reqAtStr);

        int custStart        = sectionStart(json, 0, "customer");
        float custAvgAmount  = extractFloat(json, custStart, "avg_amount");
        int custTxCount      = extractInt(json, custStart, "tx_count_24h");

        int merchStart       = sectionStart(json, 0, "merchant");
        String merchId       = extractStr(json, merchStart, "id");
        int mccCode          = extractIntStr(json, merchStart, "mcc");
        float merchAvg       = extractFloat(json, merchStart, "avg_amount");

        boolean unknownMerchant = !merchantIsKnown(json, custStart, "known_merchants", merchId);

        int termStart        = sectionStart(json, 0, "terminal");
        boolean isOnline     = extractBool(json, termStart, "is_online");
        boolean cardPresent  = extractBool(json, termStart, "card_present");
        float kmFromHome     = extractFloat(json, termStart, "km_from_home");

        LastTransactionData lastTx = null;
        int ltIdx = json.indexOf("\"last_transaction\"");
        if (ltIdx >= 0) {
            int colon = json.indexOf(':', ltIdx) + 1;
            while (json.charAt(colon) == ' ' || json.charAt(colon) == '\n' || json.charAt(colon) == '\r') colon++;
            if (json.charAt(colon) == '{') {
                long lastEpoch      = tsEpochSeconds(extractStr(json, colon, "timestamp"));
                float kmFromCurrent = extractFloat(json, colon, "km_from_current");
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

    public static float[] toRequestArray(String json) {
        float[] txArray = new float[14];

        // ── TRANSACTION ──
        int txStart = sectionStart(json, 0, "transaction");
        float txAmount = extractFloat(json, txStart, "amount");
        int txInstallments = extractInt(json, txStart, "installments");
        String reqAt = extractStr(json, txStart, "requested_at");
        int txHour = tsHour(reqAt);
        int txDow = tsDayOfWeek(reqAt);
        long txEpoch = tsEpochSeconds(reqAt);
        // ── CUSTOMER ──
        int custStart = sectionStart(json, 0, "customer");
        float custAvgAmount = extractFloat(json, custStart, "avg_amount");
        int custTxCount = extractInt(json, custStart, "tx_count_24h");
        // ── MERCHANT ──
        int merchStart = sectionStart(json, 0, "merchant");
        String merchId = extractStr(json, merchStart, "id");
        int mccCode = extractIntStr(json, merchStart, "mcc");
        float merchAvg = extractFloat(json, merchStart, "avg_amount");
        // ── unknown merchant ──
        boolean unknownMerchant = !merchantIsKnown(json, custStart, "known_merchants", merchId);
        // ── TERMINAL ──
        int termStart = sectionStart(json, 0, "terminal");
        boolean isOnline = extractBool(json, termStart, "is_online");
        boolean cardPresent = extractBool(json, termStart, "card_present");
        float kmFromHome = extractFloat(json, termStart, "km_from_home");
        // ── LAST TRANSACTION (pode ser null) ──
        int lastTransaction = json.indexOf("\"last_transaction\"");
        long lastEpoch = 0;
        float kmFromCurrent = 0;
        if (lastTransaction >= 0) {
            int colon = json.indexOf(':', lastTransaction) + 1;
            while (json.charAt(colon) == ' ' || json.charAt(colon) == '\n' || json.charAt(colon) == '\r') colon++;
            if (json.charAt(colon) == '{') {
                lastEpoch = tsEpochSeconds(extractStr(json, colon, "timestamp"));
                kmFromCurrent = extractFloat(json, colon, "km_from_current");
            }
        }

        // [0] amount
        txArray[0] = txAmount;

        // [1] installments
        txArray[1] = txInstallments;

        // [2] amount_vs_avg
        txArray[2] = custAvgAmount > 0 ? (txAmount / custAvgAmount) : 0f;

        // [3] hour_of_day
        txArray[3] = txHour;

        // [4] day_of_week
        txArray[4] = txDow;

        // [5] minutes_since_last_tx, [6] km_from_last_tx
        if (lastEpoch == 0 && kmFromCurrent == 0) {
            txArray[5] = -1f;
            txArray[6] = -1f;
        } else {
            long minutes = (txEpoch - lastEpoch) / 60L;  // positivo se tx é posterior
            txArray[5] = minutes;
            txArray[6] = kmFromCurrent;
        }

        // [7] km_from_home
        txArray[7] = kmFromHome;

        // [8] tx_count_24h
        txArray[8] = custTxCount;

        // [9] is_online
        txArray[9] = isOnline ? 1f : 0f;

        // [10] card_present
        txArray[10] = cardPresent ? 1f : 0f;

        // [11] unknown_merchant
        txArray[11] = unknownMerchant ? 1f : 0f;

        // [12] mcc_risk (índice direto, sem parse)
        txArray[12] = mccCode;

        // [13] merchant_avg_amount
        txArray[13] = merchAvg;

        return txArray;
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

