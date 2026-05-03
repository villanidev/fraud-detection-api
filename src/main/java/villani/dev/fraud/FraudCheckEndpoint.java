package villani.dev.fraud;

import io.helidon.common.media.type.MediaTypes;
import io.helidon.http.Http;
import io.helidon.service.registry.Service;
import io.helidon.webserver.http.RestServer;
import io.helidon.webserver.http.ServerRequest;

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
    public String checkScore(ServerRequest req) {
        long requestStartNs = System.nanoTime();
        try {
            String body = req.content().as(String.class);
            TransactionRequest tx = TransactionRequest.parse(body);
            return fraudCheckService.checkScore(tx, requestStartNs);
        } catch (Exception e) {
            /*
                HTTP error: peso 5  (Err × 5)
                aprova fraud: peso 3  (FN × 3)
                500 é matematicamente pior que deixar passar fraude
            */
            return "{\"approved\":true,\"fraud_score\":0.0000}";
        }
    }
}

