package villani.dev.fraud;

import io.helidon.common.media.type.MediaTypes;
import io.helidon.http.Http;
import io.helidon.service.registry.Service;
import io.helidon.webserver.http.RestServer;

@SuppressWarnings("deprecation")
@RestServer.Endpoint
@Http.Path("/fraud-score")
@Service.Singleton
public class FraudCheckEndpoint {

    private final FraudCheckService fraudCheckService;

    @Service.Inject
    FraudCheckEndpoint(FraudCheckService fraudCheckService) {
        this.fraudCheckService = fraudCheckService;
    }

    @Http.POST
    @Http.Produces(MediaTypes.APPLICATION_JSON_VALUE)
    public byte[] checkScore(@Http.Entity String body) {
        float[] txArray = TransactionRequestBytes.toRequestArray(body);
        return fraudCheckService.checkScore2(txArray);
    }
}
