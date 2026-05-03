package villani.dev.health;

import io.helidon.common.media.type.MediaTypes;
import io.helidon.http.Http;
import io.helidon.service.registry.Service;
import io.helidon.webserver.http.RestServer;
import jakarta.json.Json;
import jakarta.json.JsonBuilderFactory;
import jakarta.json.JsonObject;

import java.util.Map;

@SuppressWarnings("deprecation")
@RestServer.Endpoint // identifies this class as a server endpoint
@Http.Path("/ready") // serve this endpoint on /greet context root (path)
@Service.Singleton   // a singleton service (single instance within a service registry)
public class HealthCheckEndpoint {
    private static final JsonBuilderFactory JSON = Json.createBuilderFactory(Map.of());

    @Http.GET   // HTTP GET endpoint
    @Http.Produces(MediaTypes.APPLICATION_JSON_VALUE) // produces entity of application/json media type
    public JsonObject getDefaultMessageHandler() {
        // build the JSON object (requires `helidon-http-media-jsonp` on classpath)
        return JSON.createObjectBuilder()
                .add("message", " ready")
                .build();
    }
}
