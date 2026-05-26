package villani.dev.preprocessing;

import villani.dev.vectorsearch.index.VectorIndex;
import villani.dev.vectorsearch.index.strategies.bruteforce.BruteForceIndex;

import java.util.*;
import java.util.stream.IntStream;

public class RecallEvaluator {

    // Classe para armazenar resultados do benchmark
    public record BenchmarkResult(int[][] neighbors, float[][] distances, double avgLatencyMs, double qps) {}

    /**
     * Calcula os vizinhos exatos (top‑k) para um conjunto de queries usando força bruta.
     * Utiliza uma instância de {@link BruteForceIndex} para consistência com o resto do sistema.
     *
     * @param queries      vetores de consulta (ex: 10.000 x 14)
     * @param references   vetores de referência (ex: 3.000.000 x 14) – apenas os vetores, sem labels
     * @param k            número de vizinhos desejado (ex: 5)
     * @return array [nQueries][k] com os IDs dos vizinhos exatos
     */
    public static int[][] computeGroundTruth(float[][] queries, float[] referencesFlat, int k) {
        // BruteForceIndex só precisa dos vetores; labels não são usadas para a busca
        int N = referencesFlat.length / 14;
        BruteForceIndex brute = new BruteForceIndex(referencesFlat, new byte[N]);
        int nQueries = queries.length;
        int[][] groundTruth = new int[nQueries][k];

        System.out.printf("[recall] - Calculates the exact (top-%s) neighbors for a set of brute-force queries (%s)%n", k, nQueries);

        IntStream.range(0, nQueries).parallel().forEach(q -> {
            int[] neighbors = new int[k];
            float[] distances = new float[k];
            brute.search(queries[q], k, neighbors, distances); // o retorno (fraudCount) é ignorado
            groundTruth[q] = neighbors;
        });

        return groundTruth;
    }

    /**
     * Calcula o Recall@k médio comparando os vizinhos aproximados com o ground truth.
     *
     * @param groundTruth     IDs dos vizinhos exatos (nQueries x k)
     * @param approxNeighbors IDs dos vizinhos aproximados (nQueries x k)
     * @param k               número de vizinhos considerado
     * @return recall médio (0.0 a 1.0)
     */
    public static double evaluateRecall(int[][] groundTruth, int[][] approxNeighbors, int k) {
        //System.out.printf("[recall] - Calculate the average Recall@%s by comparing the closest neighbors to the ground truth.%n", k);
        double totalRecall = 0.0;
        int nQueries = groundTruth.length;

        for (int q = 0; q < nQueries; q++) {
            Set<Integer> trueSet = new HashSet<>();
            for (int i = 0; i < k; i++) {
                trueSet.add(groundTruth[q][i]);
            }
            int hits = 0;
            for (int i = 0; i < k; i++) {
                if (trueSet.contains(approxNeighbors[q][i])) {
                    hits++;
                }
            }
            totalRecall += (double) hits / k;
        }
        return totalRecall / nQueries;
    }

    /**
     * Executa uma bateria de queries no índice e retorna vizinhos + métricas de tempo.
     *
     * @param index    índice configurado (ex: ReRankingVectorIndex)
     * @param queries  vetores de consulta
     * @param k        número de vizinhos a retornar
     * @return objeto com vizinhos aproximados e métricas (latência, QPS)
     */
    public static BenchmarkResult benchmark(VectorIndex index, float[][] queries, int k) {
        /*System.out.printf("[recall] - Running %s queries in the index (%s) and collecting neighbors + time metrics.%n",
                k, index.getClass().getSimpleName());*/

        int nQueries = queries.length;
        int[][] neighbors = new int[nQueries][k];
        float[][] distances = new float[nQueries][k];

        long totalNanos = 0;
        for (int q = 0; q < nQueries; q++) {
            int[] neigh = new int[k];
            float[] dist = new float[k];
            long start = System.nanoTime();
            index.search(queries[q], k, neigh, dist);
            long elapsed = System.nanoTime() - start;
            totalNanos += elapsed;
            neighbors[q] = neigh;
            distances[q] = dist;
        }

        double avgLatencyMs = (totalNanos / 1e6) / nQueries;
        double qps = 1000.0 / avgLatencyMs;  // queries por segundo (single-thread)

        //System.out.printf("[recall] - Finished Running queries in the index (%s)%n", index.getClass().getSimpleName());
        return new BenchmarkResult(neighbors, distances, avgLatencyMs, qps);
    }
}
