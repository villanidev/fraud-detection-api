package villani.dev.preprocessing;

import io.helidon.service.registry.Service;
import jakarta.json.Json;
import jakarta.json.JsonArray;
import jakarta.json.JsonObject;
import jakarta.json.JsonReader;
import jakarta.json.stream.JsonParser;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/**
 * Loads reference data files into memory for preprocessing.
 * Uses streaming JSON-P parser for references.json.gz to avoid loading 284 MB at once.
 */
@Service.Singleton
public class DataReader {

    /**
     * Container for the fully loaded reference dataset.
     */
    public record ReferenceData(float[][] vectors, byte[] labels) {}

    /**
     * Streams references.json.gz and extracts vectors + labels in a single pass.
     * Format: JSON array of {"vector":[14 floats], "label":"fraud"|"legit"}
     *
     * @param path path to references.json.gz
     */
    public ReferenceData loadReferences(Path path) throws IOException {
        List<float[]> vectorList = new ArrayList<>(3_100_000);
        List<Byte> labelList = new ArrayList<>(3_100_000);

        try (InputStream fis = Files.newInputStream(path);
             GZIPInputStream gzip = new GZIPInputStream(new BufferedInputStream(fis, 65536));
             JsonParser parser = Json.createParser(gzip)) {

            // Outer structure: START_ARRAY
            parser.next(); // START_ARRAY

            while (parser.hasNext()) {
                JsonParser.Event event = parser.next();
                if (event == JsonParser.Event.END_ARRAY) break;
                if (event != JsonParser.Event.START_OBJECT) continue;

                JsonObject obj = parser.getObject();

                JsonArray vectorJson = obj.getJsonArray("vector");
                float[] vec = new float[14];
                for (int i = 0; i < 14; i++) {
                    vec[i] = (float) vectorJson.getJsonNumber(i).doubleValue();
                }

                byte label = "fraud".equals(obj.getString("label")) ? (byte) 1 : (byte) 0;

                vectorList.add(vec);
                labelList.add(label);
            }
        }

        int N = vectorList.size();
        float[][] vectors = vectorList.toArray(new float[0][]);
        byte[] labels = new byte[N];
        for (int i = 0; i < N; i++) labels[i] = labelList.get(i);

        return new ReferenceData(vectors, labels);
    }

    /**
     * Loads normalization.json into a float[7] array.
     * Order: max_amount, max_installments, amount_vs_avg_ratio,
     *        max_minutes, max_km, max_tx_count_24h, max_merchant_avg_amount
     */
    public float[] loadNormalization(Path path) throws IOException {
        try (JsonReader reader = Json.createReader(Files.newBufferedReader(path))) {
            JsonObject obj = reader.readObject();
            return new float[] {
                (float) obj.getJsonNumber("max_amount").doubleValue(),
                (float) obj.getJsonNumber("max_installments").doubleValue(),
                (float) obj.getJsonNumber("amount_vs_avg_ratio").doubleValue(),
                (float) obj.getJsonNumber("max_minutes").doubleValue(),
                (float) obj.getJsonNumber("max_km").doubleValue(),
                (float) obj.getJsonNumber("max_tx_count_24h").doubleValue(),
                (float) obj.getJsonNumber("max_merchant_avg_amount").doubleValue()
            };
        }
    }

    /**
     * Loads mcc_risk.json into a float[10000] table indexed by integer MCC code.
     * Unknown MCCs default to 0.5.
     */
    public float[] loadMccRisks(Path path) throws IOException {
        float[] table = new float[10_000];
        Arrays.fill(table, 0.5f);

        try (JsonReader reader = Json.createReader(Files.newBufferedReader(path))) {
            JsonObject obj = reader.readObject();
            for (Map.Entry<String, jakarta.json.JsonValue> entry : obj.entrySet()) {
                try {
                    int code = Integer.parseInt(entry.getKey());
                    if (code >= 0 && code < table.length) {
                        table[code] = (float) ((jakarta.json.JsonNumber) entry.getValue()).doubleValue();
                    }
                } catch (NumberFormatException ignored) {
                    // skip non-numeric MCC keys
                }
            }
        }
        return table;
    }
}
