package villani.dev.vectorsearch.index.strategies.ivfpq;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

/**
 * K-Means clustering with K-Means++ initialization.
 * Plain Java — no DI annotations; created by ProductQuantizer and VectorIndexFactory.
 * Complexity:
 *   initPlusPlus  — O(N×K×D): incremental distance update per new centroid (not O(N×K²×D))
 *   cluster loop  — O(iter × N×K×D): assignment parallelized across CPU cores
 *   assign        — O(N×K×D): single pass stores assignments, second pass fills lists
 */
public class KMeans {

    // Classe auxiliar para acumulação parcial
    private static class PartialResult {
        final int[] counts;
        final float[][] sum;
        PartialResult(int k, int dim) {
            counts = new int[k];
            sum = new float[k][dim];
        }
    }

    private static final int ITERATIONS_IVF = 20;          // para os centróides principais
    private static final int ITERATIONS_PQ  = 50;          // para os codebooks PQ (mais iterações = melhor)
    private static final int N_TRIALS_IVF  = 1;            // IVF se beneficia menos de múltiplos trials
    private static final int N_TRIALS_PQ   = 3;            // PQ é mais sensível, 3 trials já ajudam
    private static final int SAMPLE_SIZE   = 300_000;      // usar 300k amostras para IVF (defina 0 para todos)

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
            System.out.printf("[KMeans] IVF: usando %d amostras (total %d)%n", SAMPLE_SIZE, vectors.length);
        }
        float[][] bestCentroids = null;
        double bestInertia = Double.MAX_VALUE;

        for (int trial = 0; trial < N_TRIALS_IVF; trial++) {
            long trialSeed = seed + trial;
            float[][] centroids = initializeCentroids(trainVectors, k, trialSeed);
            double inertia = runKMeans(trainVectors, centroids, ITERATIONS_IVF);
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
            double inertia = runKMeans(vectors, centroids, ITERATIONS_PQ);
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
    private double runKMeans(float[][] vectors, float[][] centroids, int maxIter) {
        int n = vectors.length;
        int k = centroids.length;
        int dim = centroids[0].length;
        int[] assignments = new int[n];

        for (int iter = 0; iter < maxIter; iter++) {
            // --- Atribuição paralela ---
            boolean[] changed = {false};
            IntStream.range(0, n).parallel().forEach(i -> {
                int nearest = nearestSub(vectors[i], centroids, dim);
                if (assignments[i] != nearest) {
                    assignments[i] = nearest;
                    changed[0] = true;
                }
            });
            if (!changed[0]) break; // convergiu

            // --- Acumulação paralela (reduce) ---
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
                        int cluster = assignments[i];
                        pr.counts[cluster]++;
                        float[] vec = vectors[i];
                        float[] sum = pr.sum[cluster];
                        for (int d = 0; d < dim; d++) {
                            sum[d] += vec[d];
                        }
                    }
                    return pr;
                }));
            }

            // Combinar resultados
            int[] counts = new int[k];
            float[][] sum = new float[k][dim];
            for (Future<PartialResult> f : futures) {
                try {
                    PartialResult pr = f.get();
                    for (int j = 0; j < k; j++) {
                        counts[j] += pr.counts[j];
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
            Random rnd = new Random(iter * 12345L);
            for (int j = 0; j < k; j++) {
                if (counts[j] > 0) {
                    float[] cent = centroids[j];
                    float[] s = sum[j];
                    for (int d = 0; d < dim; d++) {
                        cent[d] = s[d] / counts[j];
                    }
                } else {
                    // Cluster vazio: reinicializa com um vetor aleatório
                    int randomIdx = rnd.nextInt(n);
                    System.arraycopy(vectors[randomIdx], 0, centroids[j], 0, dim);
                }
            }
        }

        // --- Inércia final (paralela) ---
        return IntStream.range(0, n).parallel().mapToDouble(i -> {
            double dist = 0.0;
            float[] vec = vectors[i];
            float[] cent = centroids[assignments[i]];
            for (int d = 0; d < dim; d++) {
                double diff = vec[d] - cent[d];
                dist += diff * diff;
            }
            return dist;
        }).sum();
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
}
