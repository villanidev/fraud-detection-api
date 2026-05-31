package villani.dev;

import io.helidon.logging.common.LogConfig;
import io.helidon.service.registry.Service;
import io.helidon.service.registry.ServiceRegistryManager;
import io.helidon.service.registry.Services;
import io.helidon.webserver.WebServer;
import villani.dev.preprocessing.DataReader;
import villani.dev.preprocessing.KMeansEvaluator;
import villani.dev.preprocessing.RecallEvaluator;
import villani.dev.vectorsearch.index.VectorIndex;
import villani.dev.vectorsearch.index.strategies.ivfpq.IVFPQIndex;
import villani.dev.vectorsearch.retrieval.VectorStore;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Main class responsible for starting the service registry.
 * <p>
 * This class is annotated with {@link io.helidon.service.registry.Service.GenerateBinding}, which (when used in combination
 * with Helidon Maven Plugin) generates a binding class that can be used to bootstrap Helidon without usage of reflection
 * and classpath lookup during discovery of services.
 */
// annotation is required to generate application binding
@Service.GenerateBinding
public class Main {
    static {
        // used when building with GraalVM native image to configure logging during build
        LogConfig.initClass();
    }

    /**
     * Cannot be instantiated.
     */
    private Main() {
    }

    /**
     * Application main entry point.
     *
     * @param args command line arguments.
     */
    public static void main(String[] args) throws Exception {
        LogConfig.configureRuntime();

        if (args.length > 0 && "--preprocess".equals(args[0])) {
            runPreProcessor();
            return;
        }

        if (args.length > 0 && "--evaluation".equals(args[0])) {
            runKmeansEvaluation();
            return;
        }

        if (args.length > 0 && "--recall".equals(args[0])) {
            runRecallEvaluation();
            return;
        }

        // If SERVER_SOCKET_PATH is set, configure Helidon to bind to unix socket instead of TCP.
        // Workaround for Helidon 4.4.1 bug (issue #11842): the config parser strips 7 chars from
        // "unix:..." instead of 6, so "unix:/path" becomes "path" (missing leading slash).
        // Using "unix://" + socketPath (which starts with /) produces "unix:///path" (8 chars),
        // and after stripping 7 we get "/path" — correct.
        String socketPath = System.getenv("SERVER_SOCKET_PATH");
        if (socketPath != null && !socketPath.isBlank()) {
            Files.deleteIfExists(Path.of(socketPath));
            // Make Helidon bing do uds
            System.setProperty("server.bind-address", "unix://" + socketPath);
        }

        ServiceRegistryManager.start(ApplicationBinding.create());

        // Helidon creates the socket during start — make it world-writable so HAProxy
        // (which may run as a different user) can connect to it.
        if (socketPath != null && !socketPath.isBlank()) {
            Files.setPosixFilePermissions(Path.of(socketPath),
                    PosixFilePermissions.fromString("rwxrwxrwx"));
        }

        VectorStore vectorStore = Services.get(VectorStore.class);
        System.out.println("Loading references (data.bin) ...");
        vectorStore.load(Path.of(System.getenv().getOrDefault("DATA_BIN_PATH", "data.bin")));

        WebServer webServer = Services.get(WebServer.class);
        if (socketPath != null && !socketPath.isBlank()) {
            System.out.println("Fraud detection API started on unix socket: " + socketPath);
        } else {
            System.out.println("Fraud detection API started on port: " + webServer.port());
        }
    }

    private static void runPreProcessor() throws IOException {
        ServiceRegistryManager.start(ApplicationBinding.create());
        var env = System.getenv();
        Services.get(villani.dev.preprocessing.PreProcessorService.class)
                .ingestReferences(
                        Path.of(env.getOrDefault("REFERENCES_PATH",   "src/main/resources/references.json.gz")),
                        Path.of(env.getOrDefault("NORMALIZATION_PATH", "src/main/resources/normalization.json")),
                        Path.of(env.getOrDefault("MCC_RISK_PATH",      "src/main/resources/mcc_risk.json")),
                        Path.of(env.getOrDefault("DATA_BIN_PATH",      "data.bin")));
        System.exit(0);
    }

    private static void runKmeansEvaluation() throws IOException {
        DataReader dataReader = Services.get(DataReader.class);
        System.out.println("[evaluation] Loading references...");
        DataReader.ReferenceData ref = dataReader.loadReferences(Path.of("src/main/resources/references.json.gz"));
        float[] vectorsFlat = ref.flat();
        int N = ref.count();
        int[] kCandidates = IntStream.iterate(2048, n -> n <= 4096, n -> n + 64).toArray();

        System.out.printf("[evaluation] Running KMeans evaluation with cluster candidates (%s)...%n",
                Arrays.toString(kCandidates));
        KMeansEvaluator evaluator = new KMeansEvaluator(
            vectorsFlat,          // seus 3M vetores (flat)
            N,
            kCandidates,
            42L,                  // seed base
            600_000,              // amostra fixa (use 0 para todos, mas será lento)
            3                     // trials por K para média
        );

        Map<Integer, Double> results = evaluator.evaluate();
        System.out.println("[evaluation] Finding best cluster (k)...");
        int bestK = evaluator.suggestK(results);
        System.out.println("[evaluation] Best K: " + bestK);
        System.exit(0);
    }

    private static void runRecallEvaluation() throws IOException {
        VectorStore vectorStore = Services.get(VectorStore.class);
        System.out.println("[recall] Loading data.bin...");
        vectorStore.load(Path.of(System.getenv().getOrDefault("DATA_BIN_PATH", "data.bin")));

        DataReader dataReader = Services.get(DataReader.class);
        System.out.println("[recall] Loading references...");
        DataReader.ReferenceData ref = dataReader.loadReferences(Path.of("src/main/resources/references.json.gz"));
        float[] vectorsFlat = ref.flat();
        byte[] labels = ref.labels();
        int N = ref.count();
        // build sampleVectors (10k) by copying from flat array without materializing full matrix
        int sampleSize = 20_000;
        float[][] sampleVectors = new float[sampleSize][14];
        Random rnd = new Random(42L);
        // generate distinct indices
        int[] indices = rnd.ints(0, N).distinct().limit(sampleSize).toArray();
        for (int i = 0; i < sampleSize; i++) {
            System.arraycopy(vectorsFlat, indices[i] * 14, sampleVectors[i], 0, 14);
        }

        int[] nprobes = { 1, 2, 4, 8, 16, 32 };
        int[] candidates = { 10, 15, 20, 30, 40, 50 };

        int[][] groundTruth = RecallEvaluator.computeGroundTruth(sampleVectors, ref.flat(), 5);

        System.out.println("nprobe,candidates,Recall@5,Latência(ms),QPS");
        for (int np : nprobes) {
            //System.out.printf("[recall] running probe - %s from %s", np, Arrays.toString(nprobes));
            for (int cand : candidates) {
                //System.out.printf("[recall] running candidates - %s from %s", cand, Arrays.toString(candidates));
                // Cria índice com os parâmetros atuais
                VectorIndex index = vectorStore.createIndexForBenchmark(np, cand);

                RecallEvaluator.BenchmarkResult res = RecallEvaluator.benchmark(index, sampleVectors, 5);
                double recall = RecallEvaluator.evaluateRecall(groundTruth, res.neighbors(), 5);
                System.out.printf("%d,%d,%.4f,%.2f,%.1f%n",
                        np, cand, recall, res.avgLatencyMs(), res.qps());
            }
        }

        System.exit(0);
    }
}