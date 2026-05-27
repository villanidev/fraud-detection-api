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
     * @param neighbors pre-allocated array to receive neighbor IDs
     * @param distances pre-allocated array to receive squared distances
     * @return number of fraud labels among the topK neighbors
     */
    int search(float[] query, int topK, int[] neighbors, float[] distances);

    /** Reference labels: 0=legit, 1=fraud. */
    byte[] getLabels();

    @Override
    default void close() {}
}
