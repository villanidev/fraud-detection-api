package villani.dev.vectorsearch.index.strategies.hnsw;

import villani.dev.vectorsearch.index.VectorIndex;

/**
 * HNSW (Hierarchical Navigable Small World) index — stub.
 * Placeholder for a future implementation.
 * Graph construction happens at preprocessing time; runtime search is O(log N).
 *
 * Plain Java — no DI annotations; created by VectorIndexFactory.
 */
public class HNSWIndex implements VectorIndex {

    private final byte[] labels;

    public HNSWIndex(byte[] labels) {
        this.labels = labels;
    }

    @Override
    public int search(float[] query, int topK, int candidates, int[] neighbors, float[] distances) {
        throw new UnsupportedOperationException("HNSW index is not yet implemented");
    }

    @Override
    public byte[] getLabels() {
        return labels;
    }
}
