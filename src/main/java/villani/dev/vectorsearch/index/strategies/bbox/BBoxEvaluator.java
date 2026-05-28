package villani.dev.vectorsearch.index.strategies.bbox;

/**
 * High-performance Bounding-Box evaluator for 14-dimensional vectors.
 *
 * - Zero allocations on the hot path
 * - Manual loop unrolling for 14 dimensions
 * - Uses linearized float[] arrays with cluster-major layout (baseIdx = clusterId * 14)
 */
public final class BBoxEvaluator {
    private BBoxEvaluator() { }

    /**
     * Computes the squared minimum distance (lower bound) from the query vector to the axis-aligned
     * bounding box specified by bboxMin and bboxMax at the given baseIdx.
     *
     * Assumptions: query.length >= 14, bboxMin/bboxMax contain at least baseIdx + 14 entries.
     *
     * This method is fully unrolled for 14 dimensions and does not allocate.
     */
    public static float minDistToBBox(final float[] query, final float[] bboxMin, final float[] bboxMax, final int baseIdx) {
        float sum = 0.0f;

        float q0 = query[0];
        float min0 = bboxMin[baseIdx + 0];
        float max0 = bboxMax[baseIdx + 0];
        float d0 = q0 < min0 ? (min0 - q0) : (q0 > max0 ? (q0 - max0) : 0.0f);
        sum += d0 * d0;

        float q1 = query[1];
        float min1 = bboxMin[baseIdx + 1];
        float max1 = bboxMax[baseIdx + 1];
        float d1 = q1 < min1 ? (min1 - q1) : (q1 > max1 ? (q1 - max1) : 0.0f);
        sum += d1 * d1;

        float q2 = query[2];
        float min2 = bboxMin[baseIdx + 2];
        float max2 = bboxMax[baseIdx + 2];
        float d2 = q2 < min2 ? (min2 - q2) : (q2 > max2 ? (q2 - max2) : 0.0f);
        sum += d2 * d2;

        float q3 = query[3];
        float min3 = bboxMin[baseIdx + 3];
        float max3 = bboxMax[baseIdx + 3];
        float d3 = q3 < min3 ? (min3 - q3) : (q3 > max3 ? (q3 - max3) : 0.0f);
        sum += d3 * d3;

        float q4 = query[4];
        float min4 = bboxMin[baseIdx + 4];
        float max4 = bboxMax[baseIdx + 4];
        float d4 = q4 < min4 ? (min4 - q4) : (q4 > max4 ? (q4 - max4) : 0.0f);
        sum += d4 * d4;

        float q5 = query[5];
        float min5 = bboxMin[baseIdx + 5];
        float max5 = bboxMax[baseIdx + 5];
        float d5 = q5 < min5 ? (min5 - q5) : (q5 > max5 ? (q5 - max5) : 0.0f);
        sum += d5 * d5;

        float q6 = query[6];
        float min6 = bboxMin[baseIdx + 6];
        float max6 = bboxMax[baseIdx + 6];
        float d6 = q6 < min6 ? (min6 - q6) : (q6 > max6 ? (q6 - max6) : 0.0f);
        sum += d6 * d6;

        float q7 = query[7];
        float min7 = bboxMin[baseIdx + 7];
        float max7 = bboxMax[baseIdx + 7];
        float d7 = q7 < min7 ? (min7 - q7) : (q7 > max7 ? (q7 - max7) : 0.0f);
        sum += d7 * d7;

        float q8 = query[8];
        float min8 = bboxMin[baseIdx + 8];
        float max8 = bboxMax[baseIdx + 8];
        float d8 = q8 < min8 ? (min8 - q8) : (q8 > max8 ? (q8 - max8) : 0.0f);
        sum += d8 * d8;

        float q9 = query[9];
        float min9 = bboxMin[baseIdx + 9];
        float max9 = bboxMax[baseIdx + 9];
        float d9 = q9 < min9 ? (min9 - q9) : (q9 > max9 ? (q9 - max9) : 0.0f);
        sum += d9 * d9;

        float q10 = query[10];
        float min10 = bboxMin[baseIdx + 10];
        float max10 = bboxMax[baseIdx + 10];
        float d10 = q10 < min10 ? (min10 - q10) : (q10 > max10 ? (q10 - max10) : 0.0f);
        sum += d10 * d10;

        float q11 = query[11];
        float min11 = bboxMin[baseIdx + 11];
        float max11 = bboxMax[baseIdx + 11];
        float d11 = q11 < min11 ? (min11 - q11) : (q11 > max11 ? (q11 - max11) : 0.0f);
        sum += d11 * d11;

        float q12 = query[12];
        float min12 = bboxMin[baseIdx + 12];
        float max12 = bboxMax[baseIdx + 12];
        float d12 = q12 < min12 ? (min12 - q12) : (q12 > max12 ? (q12 - max12) : 0.0f);
        sum += d12 * d12;

        float q13 = query[13];
        float min13 = bboxMin[baseIdx + 13];
        float max13 = bboxMax[baseIdx + 13];
        float d13 = q13 < min13 ? (min13 - q13) : (q13 > max13 ? (q13 - max13) : 0.0f);
        sum += d13 * d13;

        return sum;
    }
}
