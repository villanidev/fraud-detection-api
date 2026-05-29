package villani.dev.vectorsearch.index;

/**
 * Strategy contract for vector similarity search.
 * Implementations must be thread-safe after construction.
 */
public interface VectorIndex extends AutoCloseable {

    /**
     * Finds the topK nearest neighbors of the query vector.
     *
     * @param query     14-dimensional query vector
     * @param topK         number of neighbors to return
     * @param candidates   number of coarse candidates the caller expects (caller-owned buffer size)
     * @param neighbors pre-allocated array to receive neighbor IDs
     * @param distances pre-allocated array to receive squared distances
     * @return number of fraud labels among the topK neighbors
     */
    int search(float[] query, int topK, int candidates, int[] neighbors, float[] distances);

    /**
     * Backwards-compatible overload: delegates to the new API using the provided
     * distances buffer length as the caller's declared `candidates`. New callers
     * should prefer the explicit `candidates` overload.
     */
    default int search(float[] query, int topK, int[] neighbors, float[] distances) {
        return search(query, topK, distances.length, neighbors, distances);
    }

    /**
     * Optional per-call override for `nprobe` and `candidates` used by some
     * index implementations (e.g. IVF+PQ). Default implementation delegates to
     * the explicit `candidates` search and ignores `nprobeParam`.
     */
    default int searchWithParams(float[] query, int topK, int nprobeParam, int candidatesParam, int[] neighbors, float[] distances) {
        return search(query, topK, candidatesParam, neighbors, distances);
    }

    /** Reference labels: 0=legit, 1=fraud. */
    byte[] getLabels();

    @Override
    default void close() {}
}
