package villani.dev.health;

import io.helidon.common.media.type.MediaTypes;
import io.helidon.http.Http;
import io.helidon.service.registry.Service;
import io.helidon.webserver.http.RestServer;
import villani.dev.fraud.FraudCheckService;

@SuppressWarnings("deprecation")
@RestServer.Endpoint
@Http.Path("/stats")
@Service.Singleton
public class StatsEndpoint {

    private final FraudCheckService fraudCheckService;

    @Service.Inject
    StatsEndpoint(FraudCheckService fraudCheckService) {
        this.fraudCheckService = fraudCheckService;
    }

    @Http.GET
    @Http.Produces(MediaTypes.APPLICATION_JSON_VALUE)
    public String getStats() {
        long total    = fraudCheckService.getTotalRequests();
        long grayZone = fraudCheckService.getGrayZoneRequests();
        double grayPct = total == 0 ? 0.0 : grayZone * 100.0 / total;
        return String.format(
            "{\"total\":%d,\"gray_zone\":%d,\"gray_zone_pct\":%.2f}",
            total, grayZone, grayPct);
    }
}
