package villani.dev;

import io.helidon.logging.common.LogConfig;
import io.helidon.service.registry.Service;
import io.helidon.service.registry.ServiceRegistryManager;
import io.helidon.service.registry.Services;
import io.helidon.webserver.WebServer;
import villani.dev.preprocessing.DataReader;
import villani.dev.preprocessing.KMeansEvaluator;
import villani.dev.vectorsearch.retrieval.VectorStore;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Map;

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
        float[][] vectors = ref.vectors();
        int[] kCandidates = { 512, 1024, 1536, 2048, 2560, 3072 };
        System.out.println("[evaluation] Running KMeans evaluation...");
        KMeansEvaluator evaluator = new KMeansEvaluator(
                vectors,              // seus 3M vetores
                kCandidates,
                42L,                  // seed base
                300_000,              // amostra fixa (use 0 para todos, mas será lento)
                3                     // trials por K para média
        );

        Map<Integer, Double> results = evaluator.evaluate();
        System.out.println("[evaluation] Finding best cluster (k)...");
        int bestK = evaluator.suggestK(results);
        System.out.println("[evaluation] Best K: " + bestK);
        System.exit(0);
    }
}