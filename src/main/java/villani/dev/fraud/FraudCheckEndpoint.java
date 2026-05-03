package villani.dev.fraud;

import io.helidon.common.media.type.MediaTypes;
import io.helidon.http.Http;
import io.helidon.service.registry.Service;
import io.helidon.webserver.http.RestServer;
import io.helidon.webserver.http.ServerRequest;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;

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
        String body = req.content().as(String.class);
        TransactionRequest tx = parseTransaction(body);
        return fraudCheckService.checkScore(tx, requestStartNs);
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

    private static List<String> extractStringArray(String j, int from, String key) {
        int k = j.indexOf('"' + key + '"', from);
        int bracket = j.indexOf('[', k);
        int end = j.indexOf(']', bracket);
        List<String> result = new ArrayList<>();
        int pos = bracket + 1;
        while (pos < end) {
            int q1 = j.indexOf('"', pos);
            if (q1 < 0 || q1 >= end) break;
            int q2 = j.indexOf('"', q1 + 1);
            result.add(j.substring(q1 + 1, q2));
            pos = q2 + 1;
        }
        return result;
    }

    // ── parser ───────────────────────────────────────────────────────────────

    private static TransactionRequest parseTransaction(String j) {
        String id = extractStr(j, 0, "id");

        int txStart = sectionStart(j, 0, "transaction");
        float txAmount       = extractFloat(j, txStart, "amount");
        int txInstallments   = extractInt(j, txStart, "installments");
        OffsetDateTime reqAt = OffsetDateTime.parse(extractStr(j, txStart, "requested_at"));

        int custStart = sectionStart(j, 0, "customer");
        float custAvgAmount  = extractFloat(j, custStart, "avg_amount");
        int custTxCount      = extractInt(j, custStart, "tx_count_24h");
        List<String> known   = extractStringArray(j, custStart, "known_merchants");

        int merchStart = sectionStart(j, 0, "merchant");
        String merchId       = extractStr(j, merchStart, "id");
        String mcc           = extractStr(j, merchStart, "mcc");
        float merchAvg       = extractFloat(j, merchStart, "avg_amount");

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
                OffsetDateTime ts    = OffsetDateTime.parse(extractStr(j, colon, "timestamp"));
                float kmFromCurrent  = extractFloat(j, colon, "km_from_current");
                lastTx = new TransactionRequest.LastTransactionData(ts, kmFromCurrent);
            }
        }

        return new TransactionRequest(id,
                new TransactionRequest.TransactionData(txAmount, txInstallments, reqAt),
                new TransactionRequest.CustomerData(custAvgAmount, custTxCount, known),
                new TransactionRequest.MerchantData(merchId, mcc, merchAvg),
                new TransactionRequest.TerminalData(isOnline, cardPresent, kmFromHome),
                lastTx);
    }
}
