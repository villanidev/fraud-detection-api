package villani.dev.vectorsearch.index.strategies.ivfpq;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * K-Means clustering with K-Means++ initialization.
 * Plain Java — no DI annotations; created by ProductQuantizer and VectorIndexFactory.
 * Complexity:
 *   initPlusPlus  — O(N×K×D): incremental distance update per new centroid (not O(N×K²×D))
 *   cluster loop  — O(iter × N×K×D): assignment parallelized across CPU cores
 *   assign        — O(N×K×D): single pass stores assignments, second pass fills lists
 */
public class KMeans {

    private static final int VECTOR_DIMS = 14;
    private static final int PQ_SUB_DIMS = 2;

    // Classe auxiliar para acumulação parcial
    private static class PartialResult {
        final int[] counts;
        final float[][] sum;
        final double[] sumSquaredNorms;
        boolean changed;
        PartialResult(int k, int dim) {
            counts = new int[k];
            sum = new float[k][dim];
            sumSquaredNorms = new double[k];
        }
    }

    private static final int ITERATIONS_IVF = 120;         // teto; a parada real ocorre por convergência
    private static final int ITERATIONS_PQ  = 120;         // PQ com K alto costuma precisar de mais folga
    private static final int MIN_ITERATIONS_IVF = 4;
    private static final int MIN_ITERATIONS_PQ = 6;
    private static final int N_TRIALS_IVF  = 1;            // IVF se beneficia menos de múltiplos trials
    private static final int N_TRIALS_PQ   = 3;            // PQ é mais sensível, 3 trials já ajudam
    private static final int SAMPLE_SIZE   = 600_000;      // usar amostras para IVF (defina 0 para todos)
    private static final double CENTROID_SHIFT_TOL_IVF = 1e-4;
    private static final double CENTROID_SHIFT_TOL_PQ = 1e-5;
    private static final double RELATIVE_INERTIA_TOL_IVF = 1e-4;
    private static final double RELATIVE_INERTIA_TOL_PQ = 1e-5;

    private final ExecutorService executor = Executors.newWorkStealingPool();

    /**
     * Treina os centroides IVF (14D) usando K-Means com múltiplas tentativas e
     * escolha do modelo de menor inércia.
     *
     * @param vectors  vetores de treinamento (cada um com dimensão original, ex: 14)
     * @param k        número de clusters
     * @param seed     semente para reprodutibilidade
     * @return centróides [k][dimensão]
     */
    public float[][] cluster(float[][] vectors, int k, long seed) {
        float[][] trainVectors = vectors;
        if (SAMPLE_SIZE > 0 && vectors.length > SAMPLE_SIZE) {
            trainVectors = sample(vectors, seed);
            System.out.printf("[KMeans] IVF: Using %d samples (total %d)%n", SAMPLE_SIZE, vectors.length);
        }
        float[][] bestCentroids = null;
        double bestInertia = Double.MAX_VALUE;

        for (int trial = 0; trial < N_TRIALS_IVF; trial++) {
            long trialSeed = seed + trial;
            float[][] centroids = initializeCentroids(trainVectors, k, trialSeed);
                double inertia = runKMeans(trainVectors, centroids, ITERATIONS_IVF, MIN_ITERATIONS_IVF,
                    CENTROID_SHIFT_TOL_IVF, RELATIVE_INERTIA_TOL_IVF);
            if (inertia < bestInertia) {
                bestInertia = inertia;
                bestCentroids = centroids;
            }
        }
        return bestCentroids;
    }

    /**
     * Overload that accepts a flat array row-major: length = N * dim (dim=14).
     * Only materializes a sample (SAMPLE_SIZE) when necessary to avoid allocating the
     * full matrix for very large N.
     */
    public float[][] cluster(float[] vectorsFlat, int N, int k, long seed) {
        float[][] trainVectors = null;
        if (SAMPLE_SIZE > 0 && N > SAMPLE_SIZE) {
            trainVectors = sampleFlat(vectorsFlat, N, seed);
            System.out.printf("[KMeans] IVF: Using %d samples (total %d)%n", SAMPLE_SIZE, N);
        } else {
            // materialize full matrix only if N is small enough
            int dim = 14;
            trainVectors = new float[N][dim];
            for (int i = 0; i < N; i++) {
                System.arraycopy(vectorsFlat, i * dim, trainVectors[i], 0, dim);
            }
        }

        float[][] bestCentroids = null;
        double bestInertia = Double.MAX_VALUE;

        for (int trial = 0; trial < N_TRIALS_IVF; trial++) {
            long trialSeed = seed + trial;
            float[][] centroids = initializeCentroids(trainVectors, k, trialSeed);
                double inertia = runKMeans(trainVectors, centroids, ITERATIONS_IVF, MIN_ITERATIONS_IVF,
                    CENTROID_SHIFT_TOL_IVF, RELATIVE_INERTIA_TOL_IVF);
            if (inertia < bestInertia) {
                bestInertia = inertia;
                bestCentroids = centroids;
            }
        }
        return bestCentroids;
    }

    /**
     * Atribui cada vetor ao centróide mais próximo.
     *
     * @param vectors    vetores de entrada (ex: 3M × 14)
     * @param centroids  centróides IVF (K × 14)
     * @return array de listas de índices [K][count]
     */
    public int[][] assign(float[][] vectors, float[][] centroids) {
        int n = vectors.length;
        int k = centroids.length;
        int dim = centroids[0].length;
        int[][] clusters = new int[k][];
        int[] counts = new int[k];
        int[] assignment = new int[n];

        for (int i = 0; i < n; i++) {
            int nearest = nearestSub(vectors[i], centroids, dim);
            assignment[i] = nearest;
            counts[nearest]++;
        }
        for (int c = 0; c < k; c++) {
            clusters[c] = new int[counts[c]];
        }
        int[] pos = new int[k];
        for (int i = 0; i < n; i++) {
            int c = assignment[i];
            clusters[c][pos[c]++] = i;
        }
        return clusters;
    }

    /** Assign using flat vectors (row-major) without materializing full matrix. */
    public int[][] assignFlat(float[] vectorsFlat, int N, float[][] centroids) {
        int n = N;
        int k = centroids.length;
        int dim = centroids[0].length;
        int[][] clusters = new int[k][];
        int[] counts = new int[k];
        int[] assignment = new int[n];

        for (int i = 0; i < n; i++) {
            int nearest = nearestCentroidFlat(vectorsFlat, i, centroids, dim);
            assignment[i] = nearest;
            counts[nearest]++;
        }
        for (int c = 0; c < k; c++) {
            clusters[c] = new int[counts[c]];
        }
        int[] pos = new int[k];
        for (int i = 0; i < n; i++) {
            int c = assignment[i];
            clusters[c][pos[c]++] = i;
        }
        return clusters;
    }

    private int nearestCentroidFlat(float[] flat, int idx, float[][] centroids, int dim) {
        int base = idx * dim;
        int best = 0;
        double bestDist = Double.MAX_VALUE;
        for (int c = 0; c < centroids.length; c++) {
            double dist = 0.0;
            float[] cent = centroids[c];
            for (int d = 0; d < dim; d++) {
                double diff = flat[base + d] - cent[d];
                dist += diff * diff;
            }
            if (dist < bestDist) {
                bestDist = dist;
                best = c;
            }
        }
        return best;
    }

    /**
     * Treina os codebooks PQ (subespaços de dimensão SUB_D) usando K-Means com
     * múltiplas tentativas e escolha do modelo de menor inércia.
     *
     * @param vectors  subvetores de treinamento (cada um com dimensão SUB_D)
     * @param k        número de centróides por subespaço (ex: 256)
     * @param seed     semente base (será modificada por trial)
     * @return centróides [k][SUB_D]
     */
    public float[][] clusterSub(float[][] vectors, int k, long seed) {
        float[][] bestCentroids = null;
        double bestInertia = Double.MAX_VALUE;

        for (int trial = 0; trial < N_TRIALS_PQ; trial++) {
            long trialSeed = seed + trial;
            float[][] centroids = initializeCentroids(vectors, k, trialSeed);
                double inertia = runKMeans(vectors, centroids, ITERATIONS_PQ, MIN_ITERATIONS_PQ,
                    CENTROID_SHIFT_TOL_PQ, RELATIVE_INERTIA_TOL_PQ);
            if (inertia < bestInertia) {
                bestInertia = inertia;
                bestCentroids = centroids;
            }
        }
        return bestCentroids;
    }

    public float[][] clusterSubFlat(float[] vectorsFlat, int N, int subspaceOffset, int k, long seed) {
        float[][] bestCentroids = null;
        double bestInertia = Double.MAX_VALUE;

        for (int trial = 0; trial < N_TRIALS_PQ; trial++) {
            long trialSeed = seed + trial;
            float[][] centroids = initializeCentroidsSubspaceFlat(vectorsFlat, N, subspaceOffset, k, trialSeed);
                double inertia = runKMeansSubspaceFlat(vectorsFlat, N, subspaceOffset, centroids,
                    ITERATIONS_PQ, MIN_ITERATIONS_PQ, CENTROID_SHIFT_TOL_PQ, RELATIVE_INERTIA_TOL_PQ);
            if (inertia < bestInertia) {
                bestInertia = inertia;
                bestCentroids = centroids;
            }
        }
        return bestCentroids;
    }

    // ---------- Métodos internos ----------

    /**
     * Encontra o índice do centróide mais próximo de um vetor.
     * Usado durante o treinamento e na codificação PQ.
     */
    public int nearestSub(float[] sub, float[][] codebook, int dim) {
        int best = 0;
        double bestDist = Double.MAX_VALUE;
        for (int c = 0; c < codebook.length; c++) {
            double dist = 0.0;
            float[] cb = codebook[c];
            for (int d = 0; d < dim; d++) {
                double diff = sub[d] - cb[d];
                dist += diff * diff;
            }
            if (dist < bestDist) {
                bestDist = dist;
                best = c;
            }
        }
        return best;
    }

    /**
     * Executa o algoritmo K-Means padrão (atribuição + atualização) por um número
     * máximo de iterações, parando se houver convergência.
     *
     * @return inércia final (soma das distâncias quadradas de cada ponto ao seu centróide)
     */
    private double runKMeans(float[][] vectors,
                             float[][] centroids,
                             int maxIter,
                             int minIterations,
                             double centroidShiftTolerance,
                             double relativeInertiaTolerance) {
        int n = vectors.length;
        int k = centroids.length;
        int dim = centroids[0].length;
        int[] assignments = new int[n];
        Arrays.fill(assignments, -1);
        int executedIterations = 0;
        String stopReason = "max-iterations";
        double lastMaxCentroidShift = Double.NaN;
        double previousInertia = Double.POSITIVE_INFINITY;
        double relativeImprovement = Double.NaN;
        int lastEmptyClusters = 0;

        for (int iter = 0; iter < maxIter; iter++) {
            executedIterations = iter + 1;
            int numThreads = Runtime.getRuntime().availableProcessors();
            int chunkSize = (n + numThreads - 1) / numThreads;
            List<Future<PartialResult>> futures = new ArrayList<>();
            for (int t = 0; t < numThreads; t++) {
                final int start = t * chunkSize;
                final int end = Math.min(n, (t + 1) * chunkSize);
                if (start >= end) continue;

                futures.add(executor.submit(() -> {
                    PartialResult pr = new PartialResult(k, dim);
                    for (int i = start; i < end; i++) {
                        int cluster = nearestSub(vectors[i], centroids, dim);
                        if (assignments[i] != cluster) {
                            assignments[i] = cluster;
                            pr.changed = true;
                        }
                        pr.counts[cluster]++;
                        float[] vec = vectors[i];
                        float[] sum = pr.sum[cluster];
                        double squaredNorm = 0.0;
                        for (int d = 0; d < dim; d++) {
                            sum[d] += vec[d];
                            squaredNorm += vec[d] * vec[d];
                        }
                        pr.sumSquaredNorms[cluster] += squaredNorm;
                    }
                    return pr;
                }));
            }

            // Combinar resultados
            int[] counts = new int[k];
            float[][] sum = new float[k][dim];
            double[] sumSquaredNorms = new double[k];
            boolean assignmentsChanged = false;
            for (Future<PartialResult> f : futures) {
                try {
                    PartialResult pr = f.get();
                    assignmentsChanged |= pr.changed;
                    for (int j = 0; j < k; j++) {
                        counts[j] += pr.counts[j];
                        sumSquaredNorms[j] += pr.sumSquaredNorms[j];
                        float[] s = sum[j];
                        float[] ps = pr.sum[j];
                        for (int d = 0; d < dim; d++) {
                            s[d] += ps[d];
                        }
                    }
                } catch (InterruptedException | ExecutionException e) {
                    throw new RuntimeException(e);
                }
            }

            // --- Atualizar centróides ---
            double maxCentroidShift = 0.0;
            for (int j = 0; j < k; j++) {
                if (counts[j] > 0) {
                    float[] cent = centroids[j];
                    float[] s = sum[j];
                    double centroidShift = 0.0;
                    for (int d = 0; d < dim; d++) {
                        float updated = s[d] / counts[j];
                        double diff = updated - cent[d];
                        centroidShift += diff * diff;
                        cent[d] = updated;
                    }
                    if (centroidShift > maxCentroidShift) maxCentroidShift = centroidShift;
                }
            }

            int emptyClusters = reassignEmptyCentroids(vectors, assignments, counts, centroids, dim);
            if (emptyClusters > 0) {
                maxCentroidShift = Double.POSITIVE_INFINITY;
            }
            lastEmptyClusters = emptyClusters;

            double currentInertia = computeInertiaFromAggregates(sum, sumSquaredNorms, counts);
            if (Double.isFinite(previousInertia) && previousInertia > 0.0) {
                relativeImprovement = (previousInertia - currentInertia) / previousInertia;
            }
            previousInertia = currentInertia;

            if (!assignmentsChanged && emptyClusters == 0) {
                stopReason = "assignments-stable";
                lastMaxCentroidShift = 0.0;
                break;
            }

            if (iter + 1 >= minIterations && emptyClusters == 0 && Math.sqrt(maxCentroidShift) <= centroidShiftTolerance) {
                stopReason = "centroid-shift";
                lastMaxCentroidShift = Math.sqrt(maxCentroidShift);
                break;
            }

            if (iter + 1 >= minIterations && emptyClusters == 0
                    && Double.isFinite(relativeImprovement)
                    && relativeImprovement >= 0.0
                    && relativeImprovement <= relativeInertiaTolerance) {
                stopReason = "relative-inertia";
                lastMaxCentroidShift = Math.sqrt(maxCentroidShift);
                break;
            }

            if (iter + 1 == maxIter) {
                stopReason = "max-iterations";
                lastMaxCentroidShift = Math.sqrt(maxCentroidShift);
            }
        }

        double inertia = computeInertia(vectors, centroids, assignments, dim);
        logConvergence("full", executedIterations, stopReason, inertia, lastMaxCentroidShift,
                centroidShiftTolerance, relativeImprovement, relativeInertiaTolerance, lastEmptyClusters);
        return inertia;
    }

    private double runKMeansSubspaceFlat(float[] vectorsFlat,
                                         int N,
                                         int subspaceOffset,
                                         float[][] centroids,
                                         int maxIter,
                                         int minIterations,
                                         double centroidShiftTolerance,
                                         double relativeInertiaTolerance) {
        int k = centroids.length;
        int[] assignments = new int[N];
        Arrays.fill(assignments, -1);
        int executedIterations = 0;
        String stopReason = "max-iterations";
        double lastMaxCentroidShift = Double.NaN;
        double previousInertia = Double.POSITIVE_INFINITY;
        double relativeImprovement = Double.NaN;
        int lastEmptyClusters = 0;

        for (int iter = 0; iter < maxIter; iter++) {
            executedIterations = iter + 1;
            int numThreads = Runtime.getRuntime().availableProcessors();
            int chunkSize = (N + numThreads - 1) / numThreads;
            List<Future<PartialResult>> futures = new ArrayList<>();
            for (int t = 0; t < numThreads; t++) {
                final int start = t * chunkSize;
                final int end = Math.min(N, (t + 1) * chunkSize);
                if (start >= end) continue;

                futures.add(executor.submit(() -> {
                    PartialResult pr = new PartialResult(k, PQ_SUB_DIMS);
                    for (int i = start; i < end; i++) {
                        int cluster = nearestSubspaceFlat(vectorsFlat, i, subspaceOffset, centroids);
                        if (assignments[i] != cluster) {
                            assignments[i] = cluster;
                            pr.changed = true;
                        }
                        pr.counts[cluster]++;
                        int base = i * VECTOR_DIMS + subspaceOffset;
                        float[] sum = pr.sum[cluster];
                        float v0 = vectorsFlat[base];
                        float v1 = vectorsFlat[base + 1];
                        sum[0] += v0;
                        sum[1] += v1;
                        pr.sumSquaredNorms[cluster] += v0 * v0 + v1 * v1;
                    }
                    return pr;
                }));
            }

            int[] counts = new int[k];
            float[][] sum = new float[k][PQ_SUB_DIMS];
            double[] sumSquaredNorms = new double[k];
            boolean assignmentsChanged = false;
            for (Future<PartialResult> f : futures) {
                try {
                    PartialResult pr = f.get();
                    assignmentsChanged |= pr.changed;
                    for (int j = 0; j < k; j++) {
                        counts[j] += pr.counts[j];
                        sumSquaredNorms[j] += pr.sumSquaredNorms[j];
                        sum[j][0] += pr.sum[j][0];
                        sum[j][1] += pr.sum[j][1];
                    }
                } catch (InterruptedException | ExecutionException e) {
                    throw new RuntimeException(e);
                }
            }

            double maxCentroidShift = 0.0;
            for (int j = 0; j < k; j++) {
                if (counts[j] > 0) {
                    float[] cent = centroids[j];
                    float updated0 = sum[j][0] / counts[j];
                    float updated1 = sum[j][1] / counts[j];
                    double d0 = updated0 - cent[0];
                    double d1 = updated1 - cent[1];
                    double centroidShift = d0 * d0 + d1 * d1;
                    cent[0] = updated0;
                    cent[1] = updated1;
                    if (centroidShift > maxCentroidShift) maxCentroidShift = centroidShift;
                }
            }

            int emptyClusters = reassignEmptySubspaceCentroids(vectorsFlat, assignments, counts, centroids, subspaceOffset);
            if (emptyClusters > 0) {
                maxCentroidShift = Double.POSITIVE_INFINITY;
            }
            lastEmptyClusters = emptyClusters;

            double currentInertia = computeInertiaFromAggregates(sum, sumSquaredNorms, counts);
            if (Double.isFinite(previousInertia) && previousInertia > 0.0) {
                relativeImprovement = (previousInertia - currentInertia) / previousInertia;
            }
            previousInertia = currentInertia;

            if (!assignmentsChanged && emptyClusters == 0) {
                stopReason = "assignments-stable";
                lastMaxCentroidShift = 0.0;
                break;
            }

            if (iter + 1 >= minIterations && emptyClusters == 0 && Math.sqrt(maxCentroidShift) <= centroidShiftTolerance) {
                stopReason = "centroid-shift";
                lastMaxCentroidShift = Math.sqrt(maxCentroidShift);
                break;
            }

            if (iter + 1 >= minIterations && emptyClusters == 0
                    && Double.isFinite(relativeImprovement)
                    && relativeImprovement >= 0.0
                    && relativeImprovement <= relativeInertiaTolerance) {
                stopReason = "relative-inertia";
                lastMaxCentroidShift = Math.sqrt(maxCentroidShift);
                break;
            }

            if (iter + 1 == maxIter) {
                stopReason = "max-iterations";
                lastMaxCentroidShift = Math.sqrt(maxCentroidShift);
            }
        }

        double inertia = computeInertiaSubspaceFlat(vectorsFlat, N, subspaceOffset, centroids, assignments);
        logConvergence("subspace-flat", executedIterations, stopReason, inertia, lastMaxCentroidShift,
                centroidShiftTolerance, relativeImprovement, relativeInertiaTolerance, lastEmptyClusters);
        return inertia;
    }

    private double computeInertiaFromAggregates(float[][] sum, double[] sumSquaredNorms, int[] counts) {
        double inertia = 0.0;
        for (int cluster = 0; cluster < counts.length; cluster++) {
            if (counts[cluster] == 0) {
                continue;
            }
            double squaredSumNorm = 0.0;
            float[] clusterSum = sum[cluster];
            for (float value : clusterSum) {
                squaredSumNorm += value * value;
            }
            inertia += sumSquaredNorms[cluster] - (squaredSumNorm / counts[cluster]);
        }
        return inertia;
    }

    private double computeInertia(float[][] vectors, float[][] centroids, int[] assignments, int dim) {
        int n = vectors.length;
        int numThreads = Runtime.getRuntime().availableProcessors();
        int chunkSize = (n + numThreads - 1) / numThreads;
        List<Future<Double>> futures = new ArrayList<>();
        for (int t = 0; t < numThreads; t++) {
            final int start = t * chunkSize;
            final int end = Math.min(n, (t + 1) * chunkSize);
            if (start >= end) continue;

            futures.add(executor.submit(() -> {
                double partialInertia = 0.0;
                for (int i = start; i < end; i++) {
                    float[] vec = vectors[i];
                    float[] cent = centroids[assignments[i]];
                    for (int d = 0; d < dim; d++) {
                        double diff = vec[d] - cent[d];
                        partialInertia += diff * diff;
                    }
                }
                return partialInertia;
            }));
        }

        double inertia = 0.0;
        for (Future<Double> future : futures) {
            try {
                inertia += future.get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
        return inertia;
    }

    /**
     * Inicialização K-Means++: escolhe centróides iniciais de forma probabilística,
     * proporcional à distância ao centróide mais próximo já escolhido.
     */
    private float[][] initializeCentroids(float[][] vectors, int k, long seed) {
        int n = vectors.length;
        int dim = vectors[0].length;
        Random random = new Random(seed);
        float[][] centroids = new float[k][dim];
        double[] minDistances = new double[n];
        Arrays.fill(minDistances, Double.MAX_VALUE);

        // Primeiro centróide aleatório
        int firstIdx = random.nextInt(n);
        System.arraycopy(vectors[firstIdx], 0, centroids[0], 0, dim);

        for (int c = 1; c < k; c++) {
            double totalWeight = 0.0;
            float[] lastCentroid = centroids[c - 1];
            for (int i = 0; i < n; i++) {
                double dist = 0.0;
                float[] vec = vectors[i];
                for (int d = 0; d < dim; d++) {
                    double diff = vec[d] - lastCentroid[d];
                    dist += diff * diff;
                }
                if (dist < minDistances[i]) {
                    minDistances[i] = dist;
                }
                totalWeight += minDistances[i];
            }
            double threshold = random.nextDouble() * totalWeight;
            double cumulative = 0.0;
            int chosen = 0;
            for (int i = 0; i < n; i++) {
                cumulative += minDistances[i];
                if (cumulative >= threshold) {
                    chosen = i;
                    break;
                }
            }
            System.arraycopy(vectors[chosen], 0, centroids[c], 0, dim);
        }
        return centroids;
    }

    private float[][] initializeCentroidsSubspaceFlat(float[] vectorsFlat, int N, int subspaceOffset, int k, long seed) {
        Random random = new Random(seed);
        float[][] centroids = new float[k][PQ_SUB_DIMS];
        double[] minDistances = new double[N];
        Arrays.fill(minDistances, Double.MAX_VALUE);

        int firstIdx = random.nextInt(N);
        int firstBase = firstIdx * VECTOR_DIMS + subspaceOffset;
        centroids[0][0] = vectorsFlat[firstBase];
        centroids[0][1] = vectorsFlat[firstBase + 1];

        for (int c = 1; c < k; c++) {
            double totalWeight = 0.0;
            float[] lastCentroid = centroids[c - 1];
            for (int i = 0; i < N; i++) {
                int base = i * VECTOR_DIMS + subspaceOffset;
                double d0 = vectorsFlat[base] - lastCentroid[0];
                double d1 = vectorsFlat[base + 1] - lastCentroid[1];
                double dist = d0 * d0 + d1 * d1;
                if (dist < minDistances[i]) {
                    minDistances[i] = dist;
                }
                totalWeight += minDistances[i];
            }

            double threshold = random.nextDouble() * totalWeight;
            double cumulative = 0.0;
            int chosen = 0;
            for (int i = 0; i < N; i++) {
                cumulative += minDistances[i];
                if (cumulative >= threshold) {
                    chosen = i;
                    break;
                }
            }

            int chosenBase = chosen * VECTOR_DIMS + subspaceOffset;
            centroids[c][0] = vectorsFlat[chosenBase];
            centroids[c][1] = vectorsFlat[chosenBase + 1];
        }
        return centroids;
    }

    private int nearestSubspaceFlat(float[] vectorsFlat, int vectorIndex, int subspaceOffset, float[][] centroids) {
        int base = vectorIndex * VECTOR_DIMS + subspaceOffset;
        float v0 = vectorsFlat[base];
        float v1 = vectorsFlat[base + 1];
        int best = 0;
        double bestDist = Double.MAX_VALUE;
        for (int c = 0; c < centroids.length; c++) {
            float[] cent = centroids[c];
            double d0 = v0 - cent[0];
            double d1 = v1 - cent[1];
            double dist = d0 * d0 + d1 * d1;
            if (dist < bestDist) {
                bestDist = dist;
                best = c;
            }
        }
        return best;
    }

    private double computeInertiaSubspaceFlat(float[] vectorsFlat, int N, int subspaceOffset, float[][] centroids, int[] assignments) {
        int numThreads = Runtime.getRuntime().availableProcessors();
        int chunkSize = (N + numThreads - 1) / numThreads;
        List<Future<Double>> futures = new ArrayList<>();
        for (int t = 0; t < numThreads; t++) {
            final int start = t * chunkSize;
            final int end = Math.min(N, (t + 1) * chunkSize);
            if (start >= end) continue;

            futures.add(executor.submit(() -> {
                double partialInertia = 0.0;
                for (int i = start; i < end; i++) {
                    int base = i * VECTOR_DIMS + subspaceOffset;
                    float[] cent = centroids[assignments[i]];
                    double d0 = vectorsFlat[base] - cent[0];
                    double d1 = vectorsFlat[base + 1] - cent[1];
                    partialInertia += d0 * d0 + d1 * d1;
                }
                return partialInertia;
            }));
        }

        double inertia = 0.0;
        for (Future<Double> future : futures) {
            try {
                inertia += future.get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
        return inertia;
    }

    /**
     * Amostra aleatória sem reposição.
     */
    private float[][] sample(float[][] vectors, long seed) {
        int n = vectors.length;
        float[][] sample = new float[SAMPLE_SIZE][];
        Random rnd = new Random(seed);
        // Amostragem por reservatório para eficiência (não aloca array de índices)
        System.arraycopy(vectors, 0, sample, 0, SAMPLE_SIZE);
        for (int i = SAMPLE_SIZE; i < n; i++) {
            int j = rnd.nextInt(i + 1);
            if (j < SAMPLE_SIZE) {
                sample[j] = vectors[i];
            }
        }
        return sample;
    }

    private float[][] sampleFlat(float[] flat, int N, long seed) {
        float[][] sample = new float[SAMPLE_SIZE][VECTOR_DIMS];
        Random rnd = new Random(seed);
        // copy first SAMPLE_SIZE
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            System.arraycopy(flat, i * VECTOR_DIMS, sample[i], 0, VECTOR_DIMS);
        }
        for (int i = SAMPLE_SIZE; i < N; i++) {
            int j = rnd.nextInt(i + 1);
            if (j < SAMPLE_SIZE) {
                System.arraycopy(flat, i * VECTOR_DIMS, sample[j], 0, VECTOR_DIMS);
            }
        }
        return sample;
    }

    private int reassignEmptyCentroids(float[][] vectors,
                                       int[] assignments,
                                       int[] counts,
                                       float[][] centroids,
                                       int dim) {
        int emptyClusters = 0;
        boolean[] usedVectors = new boolean[vectors.length];
        for (int cluster = 0; cluster < counts.length; cluster++) {
            if (counts[cluster] != 0) {
                continue;
            }
            int farthestVector = farthestAssignedVector(vectors, assignments, centroids, usedVectors, dim, -1);
            if (farthestVector < 0) {
                continue;
            }
            System.arraycopy(vectors[farthestVector], 0, centroids[cluster], 0, dim);
            usedVectors[farthestVector] = true;
            emptyClusters++;
        }
        return emptyClusters;
    }

    private int reassignEmptySubspaceCentroids(float[] vectorsFlat,
                                               int[] assignments,
                                               int[] counts,
                                               float[][] centroids,
                                               int subspaceOffset) {
        int emptyClusters = 0;
        boolean[] usedVectors = new boolean[assignments.length];
        for (int cluster = 0; cluster < counts.length; cluster++) {
            if (counts[cluster] != 0) {
                continue;
            }
            int farthestVector = farthestAssignedSubspaceVector(vectorsFlat, assignments, centroids, usedVectors, subspaceOffset);
            if (farthestVector < 0) {
                continue;
            }
            int base = farthestVector * VECTOR_DIMS + subspaceOffset;
            centroids[cluster][0] = vectorsFlat[base];
            centroids[cluster][1] = vectorsFlat[base + 1];
            usedVectors[farthestVector] = true;
            emptyClusters++;
        }
        return emptyClusters;
    }

    private int farthestAssignedVector(float[][] vectors,
                                       int[] assignments,
                                       float[][] centroids,
                                       boolean[] usedVectors,
                                       int dim,
                                       int ignoredCluster) {
        int bestVector = -1;
        double bestDistance = Double.NEGATIVE_INFINITY;
        for (int index = 0; index < vectors.length; index++) {
            if (usedVectors[index]) {
                continue;
            }
            int cluster = assignments[index];
            if (cluster < 0 || cluster == ignoredCluster) {
                continue;
            }
            double distance = squaredDistance(vectors[index], centroids[cluster], dim);
            if (distance > bestDistance) {
                bestDistance = distance;
                bestVector = index;
            }
        }
        return bestVector;
    }

    private int farthestAssignedSubspaceVector(float[] vectorsFlat,
                                               int[] assignments,
                                               float[][] centroids,
                                               boolean[] usedVectors,
                                               int subspaceOffset) {
        int bestVector = -1;
        double bestDistance = Double.NEGATIVE_INFINITY;
        for (int index = 0; index < assignments.length; index++) {
            if (usedVectors[index]) {
                continue;
            }
            int cluster = assignments[index];
            if (cluster < 0) {
                continue;
            }
            int base = index * VECTOR_DIMS + subspaceOffset;
            float[] centroid = centroids[cluster];
            double d0 = vectorsFlat[base] - centroid[0];
            double d1 = vectorsFlat[base + 1] - centroid[1];
            double distance = d0 * d0 + d1 * d1;
            if (distance > bestDistance) {
                bestDistance = distance;
                bestVector = index;
            }
        }
        return bestVector;
    }

    private double squaredDistance(float[] vector, float[] centroid, int dim) {
        double distance = 0.0;
        for (int d = 0; d < dim; d++) {
            double diff = vector[d] - centroid[d];
            distance += diff * diff;
        }
        return distance;
    }

    private void logConvergence(String mode,
                                int iterations,
                                String reason,
                                double inertia,
                                double maxCentroidShift,
                                double centroidShiftTolerance,
                                double relativeImprovement,
                                double relativeInertiaTolerance,
                                int emptyClusters) {
        if (Double.isNaN(maxCentroidShift)) {
            System.out.printf("[KMeans] %s converged after %d iterations (%s, inertia=%.6f, relImprove=%.8f, emptyClusters=%d)%n",
                    mode, iterations, reason, inertia, relativeImprovement, emptyClusters);
            return;
        }
        System.out.printf("[KMeans] %s converged after %d iterations (%s, inertia=%.6f, maxShift=%.8f, shiftTol=%.8f, relImprove=%.8f, relTol=%.8f, emptyClusters=%d)%n",
                mode, iterations, reason, inertia, maxCentroidShift, centroidShiftTolerance,
                relativeImprovement, relativeInertiaTolerance, emptyClusters);
    }
}
