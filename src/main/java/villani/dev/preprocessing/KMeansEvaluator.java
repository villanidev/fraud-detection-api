package villani.dev.preprocessing;

import villani.dev.vectorsearch.index.strategies.ivfpq.KMeans;

import java.util.*;

public class KMeansEvaluator {

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

    /**
     * Executa a avaliação e retorna um mapa K → WCSS médio.
     */
    public Map<Integer, Double> evaluate() {
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

        // Constrói mapa de médias
        Map<Integer, Double> averages = new LinkedHashMap<>();
        for (var entry : raw.entrySet()) {
            averages.put(entry.getKey(),
                    entry.getValue().stream().mapToDouble(d -> d).average().orElse(0));
        }
        return averages;
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

    /**
     * Amostragem de reservatório – mesmo algoritmo do KMeans, mantido privado para autonomia.
     */
    private static float[][] sample(float[][] vectors, int sampleSize, long seed) {
        int n = vectors.length;
        float[][] sample = new float[sampleSize][];
        Random rnd = new Random(seed);
        System.arraycopy(vectors, 0, sample, 0, sampleSize);
        for (int i = sampleSize; i < n; i++) {
            int j = rnd.nextInt(i + 1);
            if (j < sampleSize) {
                sample[j] = vectors[i];
            }
        }
        return sample;
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

    private void log(String format, Object... args) {
        if (verbose) {
            System.out.printf((format) + "%n", args);
        }
    }
}
