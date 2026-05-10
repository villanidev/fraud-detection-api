package villani.dev.health;

import io.helidon.http.Http;
import io.helidon.http.Status;
import io.helidon.service.registry.Service;
import io.helidon.webserver.http.RestServer;

@SuppressWarnings("deprecation")
@RestServer.Endpoint
@Http.Path("/ready")
@Service.Singleton
public class HealthCheckEndpoint {

    @Http.GET
    @RestServer.Status(Status.NO_CONTENT_204_CODE)
    public void health() {
        //nao retorna nada além do status code
    }
}
