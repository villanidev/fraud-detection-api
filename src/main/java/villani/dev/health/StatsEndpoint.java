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
    private final PerformanceStats  performanceStats;

    @Service.Inject
    StatsEndpoint(FraudCheckService fraudCheckService, PerformanceStats performanceStats) {
        this.fraudCheckService = fraudCheckService;
        this.performanceStats  = performanceStats;
    }

    @Http.GET
    @Http.Produces(MediaTypes.APPLICATION_JSON_VALUE)
    public String getStats() {
        long total    = fraudCheckService.getTotalRequests();
        long grayZone = fraudCheckService.getGrayZoneRequests();
        double grayPct = total == 0 ? 0.0 : grayZone * 100.0 / total;

        String timing = performanceStats.toJson();
        // Merge gray-zone info into the timing JSON
        String grayJson = String.format(
            ",\"gray_zone\":{\"count\":%d,\"pct\":%.2f}", grayZone, grayPct);
        return timing.substring(0, timing.length() - 1) + grayJson + "}";
    }
}
