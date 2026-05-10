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
    public String checkScore(@Http.Entity String body) {
        //String body = req.content().as(String.class);
        float[] txArray = TransactionRequest.toRequestArray(body);
        //byte[] bodyBytes = req.content().as(byte[].class);
        //float[] txArray = TransactionRequestBytes.toRequestArray(bodyBytes);
        return fraudCheckService.checkScore(txArray);
    }
}

