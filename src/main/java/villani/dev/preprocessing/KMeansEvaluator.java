package villani.dev.preprocessing;

import villani.dev.vectorsearch.index.strategies.ivfpq.KMeans;

import java.util.*;

public class KMeansEvaluator {

    public record ClusterOccupancyStats(int minSize,
                                        int p50Size,
                                        int p90Size,
                                        int p99Size,
                                        int maxSize,
                                        int emptyClusters,
                                        double meanSize,
                                        double stdDevSize) {
    }

    public record CoarseCoverageStats(int k,
                                      int trainingSampleSize,
                                      int probe,
                                      double top1ClusterCoverage,
                                      double topKClusterCoverage,
                                      ClusterOccupancyStats occupancy) {
    }

    public record EvaluationStats(int k,
                                  double meanWcss,
                                  double stdDevWcss,
                                  double minWcss,
                                  double maxWcss,
                                  double relativeImprovementFromPrevious,
                                  double elbowScore) {
    }

    private final float[] fullVectorsFlat;
    private final int fullVectorCount;
    private final float[][] evaluationVectors;   // amostra fixa para avaliação justa
    private final int[] kCandidates;
    private final long baseSeed;
    private final int trialsPerK;
    private final boolean verbose;

    /**
     * @param fullVectorsFlat todos os vetores de treino (p. ex. 3M)
     * @param kCandidates valores de K a testar (ex: {512, 1024, 1536, 2048, 2560, 3072})
     * @param baseSeed    semente base para reprodutibilidade
     * @param sampleSize  tamanho da amostra fixa usada em todos os testes (ex: 300000)
     * @param trialsPerK  quantas execuções por K para média (ex: 3)
     */
    public KMeansEvaluator(float[] fullVectorsFlat, int N, int[] kCandidates,
                           long baseSeed, int sampleSize, int trialsPerK) {
        this.fullVectorsFlat = fullVectorsFlat;
        this.fullVectorCount = N;
        this.kCandidates = kCandidates.clone();
        this.baseSeed = baseSeed;
        this.trialsPerK = trialsPerK;
        this.verbose = true;
        // build evaluation sample from flat array
        if (sampleSize > 0 && N > sampleSize) {
            this.evaluationVectors = sampleFlat(fullVectorsFlat, N, sampleSize, baseSeed);
            System.out.printf("[KMeansEvaluator] Using %d-sample from %d vectors%n", sampleSize, N);
        } else {
            // materialize full matrix
            int dim = 14;
            this.evaluationVectors = new float[N][dim];
            for (int i = 0; i < N; i++) {
                System.arraycopy(fullVectorsFlat, i * dim, this.evaluationVectors[i], 0, dim);
            }
        }
    }

    public List<CoarseCoverageStats> evaluateCoarseDetailed(int[] probeCandidates,
                                                            int querySampleSize,
                                                            int groundTruthK,
                                                            int trainingSampleSize) {
        float[][] queries = sampleQueries(fullVectorsFlat, fullVectorCount, querySampleSize, baseSeed + 7_919L);
        int[][] groundTruth = RecallEvaluator.computeGroundTruth(queries, fullVectorsFlat, groundTruthK);
        List<CoarseCoverageStats> stats = new ArrayList<>();

        for (int k : kCandidates) {
            log("[coarse-eval] Training IVF centroids for K=%d sample=%d ...", k, trainingSampleSize);
            KMeans kmeans = new KMeans();
            float[][] centroids = kmeans.cluster(fullVectorsFlat, fullVectorCount, k, baseSeed + k * 31L, trainingSampleSize);

            log("[coarse-eval] Assigning vectors for K=%d ...", k);
            int[][] idsByCluster = kmeans.assignFlat(fullVectorsFlat, fullVectorCount, centroids);
            int[] assignedClusterById = buildAssignedClusterById(idsByCluster, fullVectorCount);
            ClusterOccupancyStats occupancy = computeOccupancy(idsByCluster);

            int maxProbe = Arrays.stream(probeCandidates).max().orElse(1);
            int[][] probedClustersByQuery = computeProbedClusters(queries, centroids, maxProbe);
            for (int probe : probeCandidates) {
                double top1Coverage = 0.0;
                double topKCoverage = 0.0;
                for (int q = 0; q < queries.length; q++) {
                    int[] probed = probedClustersByQuery[q];
                    int top1Cluster = assignedClusterById[groundTruth[q][0]];
                    if (containsCluster(probed, probe, top1Cluster)) {
                        top1Coverage += 1.0;
                    }

                    int hits = 0;
                    for (int neighbor = 0; neighbor < groundTruthK; neighbor++) {
                        int clusterId = assignedClusterById[groundTruth[q][neighbor]];
                        if (containsCluster(probed, probe, clusterId)) {
                            hits++;
                        }
                    }
                    topKCoverage += (double) hits / groundTruthK;
                }

                stats.add(new CoarseCoverageStats(
                        k,
                        trainingSampleSize,
                        probe,
                        top1Coverage / queries.length,
                        topKCoverage / queries.length,
                        occupancy));
            }
        }

        return stats;
    }

    /**
     * Executa a avaliação e retorna um mapa K → WCSS médio.
     */
    public Map<Integer, Double> evaluate() {
        Map<Integer, EvaluationStats> stats = evaluateDetailed();
        Map<Integer, Double> averages = new LinkedHashMap<>();
        for (var entry : stats.entrySet()) {
            averages.put(entry.getKey(), entry.getValue().meanWcss());
        }
        return averages;
    }

    public Map<Integer, EvaluationStats> evaluateDetailed() {
        Map<Integer, List<Double>> raw = new LinkedHashMap<>();

        for (int k : kCandidates) {
            log("Evaluating K=%d ...", k);
            List<Double> inertias = new ArrayList<>();
            for (int t = 0; t < trialsPerK; t++) {
                long seed = baseSeed + k * 31L + t * 17L;  // semente determinística
                KMeans kmeans = new KMeans();               // usa suas configs atuais (amostragem desligada automaticamente se evaluationVectors.length == SAMPLE_SIZE)
                float[][] centroids = kmeans.cluster(evaluationVectors, k, seed);
                double wcss = computeWCSS(evaluationVectors, centroids, kmeans);
                inertias.add(wcss);
            }
            double avg = inertias.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
            raw.put(k, inertias);
            log("  K=%4d → average WCSS = %.2f", k, avg);
        }

        Map<Integer, EvaluationStats> stats = new LinkedHashMap<>();
        Integer previousK = null;
        for (var entry : raw.entrySet()) {
            int k = entry.getKey();
            List<Double> values = entry.getValue();
            double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            double variance = values.stream()
                    .mapToDouble(value -> {
                        double delta = value - mean;
                        return delta * delta;
                    })
                    .average()
                    .orElse(0.0);
            double stdDev = Math.sqrt(variance);
            double min = values.stream().mapToDouble(Double::doubleValue).min().orElse(mean);
            double max = values.stream().mapToDouble(Double::doubleValue).max().orElse(mean);
            double relativeImprovement = 0.0;
            if (previousK != null) {
                double previousMean = stats.get(previousK).meanWcss();
                relativeImprovement = relativeImprovement(previousMean, mean);
            }
            stats.put(k, new EvaluationStats(k, mean, stdDev, min, max, relativeImprovement, 0.0));
            previousK = k;
        }

        Map<Integer, Double> elbowScores = computeElbowScores(stats);
        Map<Integer, EvaluationStats> enriched = new LinkedHashMap<>();
        for (var entry : stats.entrySet()) {
            EvaluationStats current = entry.getValue();
            enriched.put(entry.getKey(), new EvaluationStats(
                    current.k(),
                    current.meanWcss(),
                    current.stdDevWcss(),
                    current.minWcss(),
                    current.maxWcss(),
                    current.relativeImprovementFromPrevious(),
                    elbowScores.getOrDefault(entry.getKey(), 0.0)));
        }
        return enriched;
    }

    /**
     * Sugere o melhor K com base no método do cotovelo (distância à linha entre extremos).
     */
    public int suggestK(Map<Integer, Double> wcssMap) {
        // Ordena os Ks
        List<Integer> ks = new ArrayList<>(wcssMap.keySet());
        Collections.sort(ks);

        int n = ks.size();
        if (n < 3) {
            return ks.get(n - 1);  // fallback: maior K
        }

        // Normalização min-max
        double[] x = new double[n];
        double[] y = new double[n];
        double minK = ks.get(0);
        double maxK = ks.get(n - 1);
        double minW = wcssMap.get(ks.get(0));
        double maxW = wcssMap.get(ks.get(n - 1));

        for (int i = 0; i < n; i++) {
            x[i] = (ks.get(i) - minK) / (maxK - minK);
            y[i] = (wcssMap.get(ks.get(i)) - minW) / (maxW - minW);
        }

        // Linha do primeiro ao último ponto normalizada
        double dx = x[n - 1] - x[0];
        double dy = y[n - 1] - y[0];
        double len = Math.sqrt(dx * dx + dy * dy);

        /*double px = x[i] - x[0];
          double py = y[i] - y[0];*/

        // Distância de cada ponto à linha
        double maxDist = -1;
        int bestIdx = 0;
        for (int i = 0; i < n; i++) {
            double dist = Math.abs(dy * x[i] - dx * y[i] + x[n - 1] * y[0] - y[n - 1] * x[0]) / len;
            // Produto vetorial escalar (Z-component do cross product) representa a área do paralelogramo
            //double dist = Math.abs(dx * py - dy * px) / len;
            if (dist > maxDist) {
                maxDist = dist;
                bestIdx = i;
            }
        }

        return ks.get(bestIdx);
    }

    public int suggestKFromStats(Map<Integer, EvaluationStats> statsMap) {
        Map<Integer, Double> means = new LinkedHashMap<>();
        for (var entry : statsMap.entrySet()) {
            means.put(entry.getKey(), entry.getValue().meanWcss());
        }
        return suggestK(means);
    }

    /**
     * Calcula o WCSS (inércia) para um conjunto de centróides.
     */
    private double computeWCSS(float[][] vectors, float[][] centroids, KMeans kmeans) {
        int dim = centroids[0].length;
        double wcss = 0.0;
        for (float[] vec : vectors) {
            int nearest = kmeans.nearestSub(vec, centroids, dim);
            float[] cent = centroids[nearest];
            double dist = 0.0;
            for (int d = 0; d < dim; d++) {
                double diff = vec[d] - cent[d];
                dist += diff * diff;
            }
            wcss += dist;
        }
        return wcss;
    }

    private static float[][] sampleFlat(float[] flat, int N, int sampleSize, long seed) {
        float[][] sample = new float[sampleSize][14];
        Random rnd = new Random(seed);
        for (int i = 0; i < sampleSize; i++) {
            System.arraycopy(flat, i * 14, sample[i], 0, 14);
        }
        for (int i = sampleSize; i < N; i++) {
            int j = rnd.nextInt(i + 1);
            if (j < sampleSize) {
                System.arraycopy(flat, i * 14, sample[j], 0, 14);
            }
        }
        return sample;
    }

    private float[][] sampleQueries(float[] flat, int N, int sampleSize, long seed) {
        int actualSample = Math.min(sampleSize, N);
        float[][] sample = new float[actualSample][14];
        Random rnd = new Random(seed);
        int[] indices = rnd.ints(0, N).distinct().limit(actualSample).toArray();
        for (int i = 0; i < actualSample; i++) {
            System.arraycopy(flat, indices[i] * 14, sample[i], 0, 14);
        }
        return sample;
    }

    private int[] buildAssignedClusterById(int[][] idsByCluster, int vectorCount) {
        int[] assignedClusterById = new int[vectorCount];
        for (int cluster = 0; cluster < idsByCluster.length; cluster++) {
            for (int id : idsByCluster[cluster]) {
                assignedClusterById[id] = cluster;
            }
        }
        return assignedClusterById;
    }

    private ClusterOccupancyStats computeOccupancy(int[][] idsByCluster) {
        int[] sizes = new int[idsByCluster.length];
        long total = 0L;
        int empty = 0;
        for (int i = 0; i < idsByCluster.length; i++) {
            sizes[i] = idsByCluster[i].length;
            total += sizes[i];
            if (sizes[i] == 0) {
                empty++;
            }
        }
        Arrays.sort(sizes);
        double mean = sizes.length == 0 ? 0.0 : (double) total / sizes.length;
        double variance = 0.0;
        for (int size : sizes) {
            double delta = size - mean;
            variance += delta * delta;
        }
        variance = sizes.length == 0 ? 0.0 : variance / sizes.length;
        return new ClusterOccupancyStats(
                percentile(sizes, 0.0),
                percentile(sizes, 0.50),
                percentile(sizes, 0.90),
                percentile(sizes, 0.99),
                percentile(sizes, 1.0),
                empty,
                mean,
                Math.sqrt(variance));
    }

    private int percentile(int[] sortedValues, double quantile) {
        if (sortedValues.length == 0) {
            return 0;
        }
        int index = (int) Math.round(quantile * (sortedValues.length - 1));
        return sortedValues[Math.max(0, Math.min(sortedValues.length - 1, index))];
    }

    private int[][] computeProbedClusters(float[][] queries, float[][] centroids, int maxProbe) {
        int k = centroids.length;
        int actualProbe = Math.min(maxProbe, k);
        int[][] probedClusters = new int[queries.length][actualProbe];
        float[] centroidDistances = new float[k];
        int[] order = new int[k];

        for (int q = 0; q < queries.length; q++) {
            for (int cluster = 0; cluster < k; cluster++) {
                centroidDistances[cluster] = squaredDistance(queries[q], centroids[cluster]);
                order[cluster] = cluster;
            }
            partialSort(order, centroidDistances, actualProbe);
            System.arraycopy(order, 0, probedClusters[q], 0, actualProbe);
        }

        return probedClusters;
    }

    private boolean containsCluster(int[] clusters, int probe, int targetCluster) {
        int limit = Math.min(probe, clusters.length);
        for (int i = 0; i < limit; i++) {
            if (clusters[i] == targetCluster) {
                return true;
            }
        }
        return false;
    }

    private float squaredDistance(float[] query, float[] centroid) {
        float distance = 0f;
        for (int dim = 0; dim < query.length; dim++) {
            float diff = query[dim] - centroid[dim];
            distance += diff * diff;
        }
        return distance;
    }

    private void partialSort(int[] order, float[] dist, int top) {
        quickselect(order, dist, 0, order.length - 1, top);
        for (int i = 1; i < top; i++) {
            int oi = order[i];
            float di = dist[oi];
            int j = i - 1;
            while (j >= 0 && dist[order[j]] > di) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = oi;
        }
    }

    private void quickselect(int[] order, float[] dist, int left, int right, int k) {
        while (left < right) {
            int pivotIdx = partition(order, dist, left, right);
            int rank = pivotIdx - left + 1;
            if (rank == k) {
                return;
            }
            if (k < rank) {
                right = pivotIdx - 1;
            } else {
                left = pivotIdx + 1;
                k -= rank;
            }
        }
    }

    private int partition(int[] order, float[] dist, int left, int right) {
        int mid = (left + right) >>> 1;
        if (dist[order[left]] > dist[order[mid]]) swap(order, left, mid);
        if (dist[order[left]] > dist[order[right]]) swap(order, left, right);
        if (dist[order[mid]] > dist[order[right]]) swap(order, mid, right);

        float pivotDist = dist[order[right]];
        int i = left - 1;
        for (int j = left; j < right; j++) {
            if (dist[order[j]] <= pivotDist) {
                i++;
                swap(order, i, j);
            }
        }
        swap(order, i + 1, right);
        return i + 1;
    }

    private void swap(int[] values, int i, int j) {
        int tmp = values[i];
        values[i] = values[j];
        values[j] = tmp;
    }

    private Map<Integer, Double> computeElbowScores(Map<Integer, EvaluationStats> stats) {
        List<Integer> ks = new ArrayList<>(stats.keySet());
        Collections.sort(ks);
        Map<Integer, Double> elbowScores = new LinkedHashMap<>();
        if (ks.size() < 3) {
            for (int k : ks) {
                elbowScores.put(k, 0.0);
            }
            return elbowScores;
        }

        double minK = ks.get(0);
        double maxK = ks.get(ks.size() - 1);
        double minW = stats.get(ks.get(ks.size() - 1)).meanWcss();
        double maxW = stats.get(ks.get(0)).meanWcss();
        double x0 = 0.0;
        double y0 = normalized(stats.get(ks.get(0)).meanWcss(), minW, maxW);
        double x1 = normalized(ks.get(ks.size() - 1), minK, maxK);
        double y1 = normalized(stats.get(ks.get(ks.size() - 1)).meanWcss(), minW, maxW);
        double lineDx = x1 - x0;
        double lineDy = y1 - y0;
        double lineLen = Math.sqrt(lineDx * lineDx + lineDy * lineDy);

        for (int k : ks) {
            double x = normalized(k, minK, maxK);
            double y = normalized(stats.get(k).meanWcss(), minW, maxW);
            double score = lineLen == 0.0
                    ? 0.0
                    : Math.abs(lineDy * x - lineDx * y + x1 * y0 - y1 * x0) / lineLen;
            elbowScores.put(k, score);
        }
        return elbowScores;
    }

    private double relativeImprovement(double previous, double current) {
        if (previous <= 0.0) {
            return 0.0;
        }
        return (previous - current) / previous;
    }

    private double normalized(double value, double min, double max) {
        if (max == min) {
            return 0.0;
        }
        return (value - min) / (max - min);
    }

    private void log(String format, Object... args) {
        if (verbose) {
            System.out.printf((format) + "%n", args);
        }
    }
}
