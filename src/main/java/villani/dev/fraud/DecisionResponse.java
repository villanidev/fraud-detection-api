package villani.dev.fraud;

import java.nio.charset.StandardCharsets;

public class DecisionResponse {
    private static final byte[][] CACHE = new byte[6][];

    static {
        for (int fraudCount = 0; fraudCount <= 5; fraudCount++) {
            float score = fraudCount / 5.0f;
            boolean approved = score < 0.6f;
            //System.out.println("Decisao - score: " +score);
            //System.out.println("Decisao - approved: " +approved);
            CACHE[fraudCount] = String.format(
                    "{\"approved\":%b,\"fraud_score\":%.4f}",
                    approved, score
            ).getBytes(StandardCharsets.UTF_8);
        }
    }

    public static byte[] get(int fraudCount) {
        return CACHE[fraudCount];
    }
}
