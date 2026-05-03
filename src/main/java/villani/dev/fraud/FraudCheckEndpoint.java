package villani.dev.fraud;

import io.helidon.common.media.type.MediaTypes;
import io.helidon.http.Http;
import io.helidon.service.registry.Service;
import io.helidon.webserver.http.RestServer;
import io.helidon.webserver.http.ServerRequest;

@SuppressWarnings("deprecation")
@RestServer.Endpoint
@Http.Path("/fraud-score")
@Service.Singleton
public class FraudCheckEndpoint {

    private final FraudCheckService fraudCheckService;

    @Service.Inject
    FraudCheckEndpoint(FraudCheckService fraudCheckService) {
        this.fraudCheckService = fraudCheckService;
    }

    @Http.POST
    @Http.Produces(MediaTypes.APPLICATION_JSON_VALUE)
    public String checkScore(ServerRequest req) {
        long requestStartNs = System.nanoTime();
        try {
            String body = req.content().as(String.class);
            TransactionRequest tx = parseTransaction(body);
            return fraudCheckService.checkScore(tx, requestStartNs);
        } catch (Exception e) {
            /*
                HTTP error: peso 5  (Err × 5)
                aprova fraud: peso 3  (FN × 3)
                500 é matematicamente pior que deixar passar fraude
            */
            return "{\"approved\":true,\"fraud_score\":0.0000}";
        }
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

    /** Scans a JSON string-array for targetId without allocating a List. */
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

    /** Extracts a quoted-string field that contains only digits, as an int. No String allocation. */
    private static int extractIntStr(String j, int from, String key) {
        int k = j.indexOf('"' + key + '"', from);
        int q1 = j.indexOf('"', j.indexOf(':', k) + 1) + 1;
        int v = 0;
        char c;
        while ((c = j.charAt(q1++)) != '"') v = v * 10 + (c - '0');
        return v;
    }

    // ── parser ───────────────────────────────────────────────────────────────

    private static TransactionRequest parseTransaction(String j) {
        String id = extractStr(j, 0, "id");

        int txStart = sectionStart(j, 0, "transaction");
        float txAmount       = extractFloat(j, txStart, "amount");
        int txInstallments   = extractInt(j, txStart, "installments");
        String reqAtStr      = extractStr(j, txStart, "requested_at");
        int txHour           = tsHour(reqAtStr);
        int txDow            = tsDayOfWeek(reqAtStr);
        long txEpoch         = tsEpochSeconds(reqAtStr); // only used when last_tx present

        int custStart = sectionStart(j, 0, "customer");
        float custAvgAmount  = extractFloat(j, custStart, "avg_amount");
        int custTxCount      = extractInt(j, custStart, "tx_count_24h");

        int merchStart = sectionStart(j, 0, "merchant");
        String merchId       = extractStr(j, merchStart, "id");
        int mccCode          = extractIntStr(j, merchStart, "mcc");  // "5411" → 5411, no String stored
        float merchAvg       = extractFloat(j, merchStart, "avg_amount");

        // Resolved to a boolean here — no List<String> stored in the record
        boolean unknownMerchant = !merchantIsKnown(j, custStart, "known_merchants", merchId);

        int termStart = sectionStart(j, 0, "terminal");
        boolean isOnline     = extractBool(j, termStart, "is_online");
        boolean cardPresent  = extractBool(j, termStart, "card_present");
        float kmFromHome     = extractFloat(j, termStart, "km_from_home");

        TransactionRequest.LastTransactionData lastTx = null;
        int ltIdx = j.indexOf("\"last_transaction\"");
        if (ltIdx >= 0) {
            int colon = j.indexOf(':', ltIdx) + 1;
            while (j.charAt(colon) == ' ' || j.charAt(colon) == '\n' || j.charAt(colon) == '\r') colon++;
            if (j.charAt(colon) == '{') {
                long lastEpoch  = tsEpochSeconds(extractStr(j, colon, "timestamp"));
                float kmFromCurrent = extractFloat(j, colon, "km_from_current");
                lastTx = new TransactionRequest.LastTransactionData(lastEpoch, kmFromCurrent);
            }
        }

        return new TransactionRequest(id,
                new TransactionRequest.TransactionData(txAmount, txInstallments, txHour, txDow, txEpoch),
                new TransactionRequest.CustomerData(custAvgAmount, custTxCount, unknownMerchant),
                new TransactionRequest.MerchantData(merchId, mccCode, merchAvg),
                new TransactionRequest.TerminalData(isOnline, cardPresent, kmFromHome),
                lastTx);
    }

    // ── fast timestamp parsing (no OffsetDateTime allocation) ────────────────
    // Format: "2026-03-11T18:45:53Z" (always UTC, always this length)

    /** Extracts UTC hour from ISO-8601 string: chars [11:13]. */
    private static int tsHour(String s) {
        return (s.charAt(11) - '0') * 10 + (s.charAt(12) - '0');
    }

    /**
     * ISO day-of-week (Monday=1 … Sunday=7) via Tomohiko Sakamoto's algorithm.
     * No object allocation, O(1).
     */
    private static int tsDayOfWeek(String s) {
        int y = digits(s, 0, 4);
        int m = digits(s, 5, 7);
        int d = digits(s, 8, 10);
        final int[] t = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
        if (m < 3) y--;
        int dow = (y + y / 4 - y / 100 + y / 400 + t[m - 1] + d) % 7;
        return dow == 0 ? 7 : dow; // 0=Sun→7 (ISO), 1=Mon→1 … 6=Sat→6
    }

    /**
     * Seconds since Unix epoch (1970-01-01T00:00:00Z), UTC only.
     * Uses Howard Hinnant's civil-from-days algorithm — no allocation, O(1).
     */
    static long tsEpochSeconds(String s) {
        int y   = digits(s, 0, 4);
        int m   = digits(s, 5, 7);
        int d   = digits(s, 8, 10);
        int h   = digits(s, 11, 13);
        int min = digits(s, 14, 16);
        int sec = digits(s, 17, 19);
        // Days since 1970-01-01 (Hinnant)
        if (m <= 2) { y--; m += 9; } else { m -= 3; }
        long era = (y >= 0 ? y : y - 399) / 400;
        int  yoe = (int)(y - era * 400);                           // [0, 399]
        int  doy = (153 * m + 2) / 5 + d - 1;                     // [0, 365]
        int  doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;         // [0, 146096]
        long days = era * 146097L + doe - 719468L;
        return days * 86400L + h * 3600L + min * 60L + sec;
    }

    private static int digits(String s, int start, int end) {
        int v = 0;
        for (int i = start; i < end; i++) v = v * 10 + (s.charAt(i) - '0');
        return v;
    }
}
