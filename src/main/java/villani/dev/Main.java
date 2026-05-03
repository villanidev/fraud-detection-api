package villani.dev;

import io.helidon.logging.common.LogConfig;
import io.helidon.service.registry.Service;
import io.helidon.service.registry.ServiceRegistryManager;
import io.helidon.service.registry.Services;
import io.helidon.webserver.WebServer;
import villani.dev.vectorsearch.retrieval.VectorStore;

import java.nio.file.Path;

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
            ServiceRegistryManager.start(ApplicationBinding.create());
            var env = System.getenv();
            Services.get(villani.dev.preprocessing.PreProcessorService.class)
                    .ingestReferences(
                            Path.of(env.getOrDefault("REFERENCES_PATH",   "src/main/resources/references.json.gz")),
                            Path.of(env.getOrDefault("NORMALIZATION_PATH", "src/main/resources/normalization.json")),
                            Path.of(env.getOrDefault("MCC_RISK_PATH",      "src/main/resources/mcc_risk.json")),
                            Path.of(env.getOrDefault("DATA_BIN_PATH",      "data.bin")));
            System.exit(0);
            return;
        }

        ServiceRegistryManager.start(ApplicationBinding.create());

        VectorStore vectorStore = Services.get(VectorStore.class);
        System.out.println("Loading references (data.bin) ...");
        vectorStore.load(Path.of(System.getenv().getOrDefault("DATA_BIN_PATH", "data.bin")));

        WebServer webServer = Services.get(WebServer.class);
        System.out.println("Fraud detection API started on port: " + webServer.port());
    }
}