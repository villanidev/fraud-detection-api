package villani.dev.health;

import io.helidon.service.registry.Service;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Thread-safe in-memory histogram for request phase timings.
 * Zero I/O per request — only AtomicLong increments.
 *
 * Phases: PARSE, EMBED, SEARCH, TOTAL
 * Buckets (ms): [0,1) [1,5) [5,10) [10,50) [50,100) [100,500) [500,∞)
 */
@Service.Singleton
public class PerformanceStats {

    public enum Phase { PARSE, EMBED, SEARCH, TOTAL }

    // Upper bounds in nanoseconds for each bucket (last bucket is unbounded)
    private static final long[] BOUNDS_NS = {
        1_000_000L,    // 1ms
        5_000_000L,    // 5ms
        10_000_000L,   // 10ms
        50_000_000L,   // 50ms
        100_000_000L,  // 100ms
        500_000_000L,  // 500ms
        Long.MAX_VALUE // ∞
    };

    private static final String[] BUCKET_LABELS = {
        "<1ms", "1-5ms", "5-10ms", "10-50ms", "50-100ms", "100-500ms", ">=500ms"
    };

    private static final int NUM_BUCKETS = BOUNDS_NS.length;
    private static final int NUM_PHASES  = Phase.values().length;

    private final AtomicLong[][] counts = new AtomicLong[NUM_PHASES][NUM_BUCKETS];
    private final AtomicLong[]   sums   = new AtomicLong[NUM_PHASES];
    private final AtomicLong[]   maxes  = new AtomicLong[NUM_PHASES];
    private final AtomicLong     total  = new AtomicLong();

    public PerformanceStats() {
        for (int p = 0; p < NUM_PHASES; p++) {
            sums[p]  = new AtomicLong(0);
            maxes[p] = new AtomicLong(0);
            for (int b = 0; b < NUM_BUCKETS; b++) {
                counts[p][b] = new AtomicLong(0);
            }
        }
    }

    public void record(Phase phase, long durationNs) {
        int p = phase.ordinal();
        sums[p].addAndGet(durationNs);
        // CAS loop for max
        long prev = maxes[p].get();
        while (durationNs > prev && !maxes[p].compareAndSet(prev, durationNs)) {
            prev = maxes[p].get();
        }
        for (int b = 0; b < NUM_BUCKETS; b++) {
            if (durationNs < BOUNDS_NS[b]) {
                counts[p][b].incrementAndGet();
                break;
            }
        }
        if (phase == Phase.TOTAL) total.incrementAndGet();
    }

    /** Records all 4 phases at once (parse, embed, search, total) in nanoseconds. */
    public void recordAll(long parseNs, long embedNs, long searchNs, long totalNs) {
        record(Phase.PARSE,  parseNs);
        record(Phase.EMBED,  embedNs);
        record(Phase.SEARCH, searchNs);
        record(Phase.TOTAL,  totalNs);
    }

    public long getTotalRequests() { return total.get(); }

    /**
     * Returns a JSON string with stats for all phases.
     * Percentiles are approximated from bucket boundaries.
     */
    public String toJson() {
        long n = total.get();
        StringBuilder sb = new StringBuilder();
        sb.append("{\"total_requests\":").append(n).append(",\"phases\":{");
        Phase[] phases = Phase.values();
        for (int p = 0; p < NUM_PHASES; p++) {
            if (p > 0) sb.append(',');
            sb.append('"').append(phases[p].name().toLowerCase()).append("\":{");
            appendPhaseJson(sb, p, n);
            sb.append('}');
        }
        sb.append("}}");
        return sb.toString();
    }

    private void appendPhaseJson(StringBuilder sb, int p, long n) {
        if (n == 0) {
            sb.append("\"count\":0");
            return;
        }
        long[] snap = new long[NUM_BUCKETS];
        long count = 0;
        for (int b = 0; b < NUM_BUCKETS; b++) {
            snap[b] = counts[p][b].get();
            count += snap[b];
        }
        double avgMs = count == 0 ? 0 : sums[p].get() / 1_000_000.0 / count;
        double maxMs = maxes[p].get() / 1_000_000.0;

        sb.append("\"count\":").append(count)
          .append(",\"avg_ms\":").append(String.format("%.3f", avgMs))
          .append(",\"max_ms\":").append(String.format("%.3f", maxMs))
          .append(",\"p50_ms\":").append(String.format("%.1f", percentile(snap, count, 0.50)))
          .append(",\"p90_ms\":").append(String.format("%.1f", percentile(snap, count, 0.90)))
          .append(",\"p99_ms\":").append(String.format("%.1f", percentile(snap, count, 0.99)))
          .append(",\"buckets\":{");
        for (int b = 0; b < NUM_BUCKETS; b++) {
            if (b > 0) sb.append(',');
            sb.append('"').append(BUCKET_LABELS[b]).append("\":").append(snap[b]);
        }
        sb.append('}');
    }

    /** Returns upper bound of the bucket containing the given percentile rank. */
    private static double percentile(long[] snap, long total, double p) {
        if (total == 0) return 0;
        long target = (long) Math.ceil(p * total);
        long cumul = 0;
        for (int b = 0; b < NUM_BUCKETS; b++) {
            cumul += snap[b];
            if (cumul >= target) {
                // Return upper bound of this bucket in ms
                return b < NUM_BUCKETS - 1 ? BOUNDS_NS[b] / 1_000_000.0 : BOUNDS_NS[NUM_BUCKETS - 2] / 1_000_000.0;
            }
        }
        return BOUNDS_NS[NUM_BUCKETS - 2] / 1_000_000.0;
    }
}
