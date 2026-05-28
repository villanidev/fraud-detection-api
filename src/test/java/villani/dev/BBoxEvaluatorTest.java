package villani.dev;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

import villani.dev.vectorsearch.index.strategies.bbox.BBoxEvaluator;

public class BBoxEvaluatorTest {

    @Test
    public void testInsideBoxIsZero() {
        float[] query = new float[14];
        for (int i = 0; i < 14; i++) query[i] = 0.5f;
        float[] min = new float[14];
        float[] max = new float[14];
        for (int i = 0; i < 14; i++) { min[i] = 0.0f; max[i] = 1.0f; }

        float dist = BBoxEvaluator.minDistToBBox(query, min, max, 0);
        assertEquals(0.0f, dist, 0.0f);
    }

    @Test
    public void testOutsideBoxMatchesReference() {
        float[] query = new float[14];
        for (int i = 0; i < 14; i++) query[i] = 0.5f;
        query[0] = -1.0f; // below min by 1.0
        query[13] = 3.0f; // above max by 2.0

        float[] min = new float[14];
        float[] max = new float[14];
        for (int i = 0; i < 14; i++) { min[i] = 0.0f; max[i] = 1.0f; }

        float actual = BBoxEvaluator.minDistToBBox(query, min, max, 0);

        float expected = 1.0f * 1.0f + 2.0f * 2.0f;
        assertEquals(expected, actual, 1e-6f);
    }

    @Test
    public void testWithBaseIdx() {
        float[] query = new float[14];
        for (int i = 0; i < 14; i++) query[i] = 2.0f;
        int K = 2;
        float[] min = new float[K * 14];
        float[] max = new float[K * 14];
        for (int c = 0; c < K; c++) {
            for (int i = 0; i < 14; i++) { min[c * 14 + i] = 0.0f; max[c * 14 + i] = 1.0f; }
        }

        float d = BBoxEvaluator.minDistToBBox(query, min, max, 14);
        // each dimension diff = 1.0 -> squared = 1.0, 14 dims => 14.0
        assertEquals(14.0f, d, 1e-6f);
    }
}
