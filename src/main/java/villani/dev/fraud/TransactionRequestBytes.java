package villani.dev.fraud;

import java.nio.charset.StandardCharsets;

public class TransactionRequestBytes {

    // Constante para mapear os índices exatos do seu array de retorno
    private static final int IDX_AMOUNT = 0;
    private static final int IDX_INSTALLMENTS = 1;
    private static final int IDX_AVG_VS_AMOUNT = 2;
    private static final int IDX_HOUR = 3;
    private static final int IDX_DOW = 4;
    private static final int IDX_MIN_SINCE_LAST = 5;
    private static final int IDX_KM_FROM_LAST = 6;
    private static final int IDX_KM_FROM_HOME = 7;
    private static final int IDX_TX_COUNT_24H = 8;
    private static final int IDX_IS_ONLINE = 9;
    private static final int IDX_CARD_PRESENT = 10;
    private static final int IDX_UNKNOWN_MERCHANT = 11;
    private static final int IDX_MCC_CODE = 12;
    private static final int IDX_MERCH_AVG = 13;

    public static float[] toRequestArray(String json) {
        float[] txArray = new float[14];

        // Inicializa valores padrões para o caso de "last transaction" ser nulo
        txArray[IDX_MIN_SINCE_LAST] = -1f;
        txArray[IDX_KM_FROM_LAST] = -1f;

        // Variáveis de controle para o cálculo de frações e chaves secundárias
        float txAmount = 0f;
        float custAvgAmount = 0f;
        long txEpoch = 0L;
        long lastEpoch = 0L;
        float kmFromCurrent = 0f;
        boolean temLastTransaction = false;

        // Strings e IDs de controle para o match de "known_merchants"
        String merchantId = "";

        int length = json.length();
        int i = 0;

        // Parser Linear (Single-Pass)
        while (i < length) {
            char c = json.charAt(i);

            if (c == '"') {
                int startKey = ++i;
                while (i < length && json.charAt(i) != '"') {
                    i++;
                }
                int endKey = i;
                i++; // Pula as aspas de fechamento

                // Avança até o caractere de valor após os dois pontos ':'
                while (i < length && (json.charAt(i) == ':' || json.charAt(i) == ' ' || json.charAt(i) == '\n' || json.charAt(i) == '\r')) {
                    i++;
                }

                // Captura do valor baseado na chave encontrada
                int keyLen = endKey - startKey;

                // Otimização por hash rápido ou tamanho da chave para evitar alocação de Strings de chaves
                if (json.regionMatches(startKey, "amount", 0, keyLen)) {
                    // Como a chave "amount" aparece em múltiplos escopos, diferenciamos pelo contexto atual no JSON
                    // Uma forma simples é verificar se já mapeamos o merchant (evita colisão)
                    float val = parseNextFloat(json, i);
                    if (txAmount == 0f) txAmount = val;
                    else if (merchantId.isEmpty()) txAmount = val; // fallback de segurança
                } else if (json.regionMatches(startKey, "installments", 0, keyLen)) {
                    txArray[IDX_INSTALLMENTS] = parseNextInt(json, i);
                } else if (json.regionMatches(startKey, "requested_at", 0, keyLen) || json.regionMatches(startKey, "requested at", 0, keyLen)) {
                    int startStr = ++i;
                    while (i < length && json.charAt(i) != '"') i++;
                    int endStr = i;

                    txArray[IDX_HOUR] = tsHour(json, startStr);
                    txArray[IDX_DOW] = tsDayOfWeek(json, startStr);
                    txEpoch = tsEpochSeconds(json, startStr);
                } else if (json.regionMatches(startKey, "avg_amount", 0, keyLen) || json.regionMatches(startKey, "avg amount", 0, keyLen)) {
                    float val = parseNextFloat(json, i);
                    if (custAvgAmount == 0f) custAvgAmount = val; // O primeiro pertence ao customer
                    else txArray[IDX_MERCH_AVG] = val; // O segundo ao merchant
                } else if (json.regionMatches(startKey, "tx_count_24h", 0, keyLen) || json.regionMatches(startKey, "tx count 24h", 0, keyLen)) {
                    txArray[IDX_TX_COUNT_24H] = parseNextInt(json, i);
                } else if (json.regionMatches(startKey, "id", 0, keyLen)) {
                    // Captura o ID do Merchant (geralmente começa com MERC)
                    int startStr = ++i;
                    while (i < length && json.charAt(i) != '"') i++;
                    if (json.regionMatches(startStr, "MERC", 0, 4)) {
                        merchantId = json.substring(startStr, i); // Única alocação inevitável para o contains posterior
                    }
                } else if (json.regionMatches(startKey, "mcc", 0, keyLen)) {
                    // Mcc vem mapeado como string no JSON (ex: "5411")
                    i++; // pula aspas
                    int startMcc = i;
                    while (i < length && json.charAt(i) != '"') i++;
                    txArray[IDX_MCC_CODE] = digits(json, startMcc, i);
                } else if (json.regionMatches(startKey, "is_online", 0, keyLen)
                        || json.regionMatches(startKey, "is online", 0, keyLen)) {
                    txArray[IDX_IS_ONLINE] = json.charAt(i) == 't' ? 1f : 0f;
                } else if (json.regionMatches(startKey, "card_present", 0, keyLen)
                        || json.regionMatches(startKey, "card present", 0, keyLen)) {
                    txArray[IDX_CARD_PRESENT] = json.charAt(i) == 't' ? 1f : 0f;
                } else if (json.regionMatches(startKey, "km_from_home", 0, keyLen)
                        || json.regionMatches(startKey, "km from home", 0, keyLen)) {
                    txArray[IDX_KM_FROM_HOME] = parseNextFloat(json, i);
                } else if (json.regionMatches(startKey, "last_transaction", 0, keyLen)
                        || json.regionMatches(startKey, "last transaction", 0, keyLen)) {
                    if (json.charAt(i) != 'n') { // se não for 'null'
                        temLastTransaction = true;
                    }
                } else if (json.regionMatches(startKey, "timestamp", 0, keyLen)) {
                    int startStr = ++i;
                    while (i < length && json.charAt(i) != '"') i++;
                    lastEpoch = tsEpochSeconds(json, startStr);
                } else if (json.regionMatches(startKey, "km_from_current", 0, keyLen)
                        || json.regionMatches(startKey, "km from current", 0, keyLen)) {
                    kmFromCurrent = parseNextFloat(json, i);
                } else if (json.regionMatches(startKey, "known_merchants", 0, keyLen)
                        || json.regionMatches(startKey, "known merchants", 0, keyLen)) {
                    // Processa a lista de conhecidos inline de forma posicional
                    while (i < length && json.charAt(i) != ']') {
                        if (json.charAt(i) == '"') {
                            int startM = ++i;
                            while (i < length && json.charAt(i) != '"') i++;
                            if (!merchantId.isEmpty() && json.regionMatches(startM, merchantId, 0, i - startM)) {
                                txArray[IDX_UNKNOWN_MERCHANT] = 0f; // É conhecido
                            }
                        }
                        i++;
                    }
                    // Se passou pela lista inteira e não achou, define como desconhecido (se já capturou o merchantId)
                    if (txArray[IDX_UNKNOWN_MERCHANT] == 0f && !merchantId.isEmpty()) {
                        // Lógica controlada pelo estado final fora do loop para precisão
                    }
                }
            }
            i++;
        }

        // Pós-processamento dos cálculos estruturados
        txArray[IDX_AMOUNT] = txAmount;
        txArray[IDX_AVG_VS_AMOUNT] = custAvgAmount > 0 ? (txAmount / custAvgAmount) : 0f;

        // Tratamento da flag de merchant desconhecido caso a lista passe antes do ID do merchant no payload
        if (!merchantId.isEmpty() && !json.contains('"' + merchantId + '"')) {
            txArray[IDX_UNKNOWN_MERCHANT] = 1f;
        } else if (!merchantId.isEmpty() && json.indexOf('"' + merchantId + '"') == json.lastIndexOf('"' + merchantId + '"')) {
            // Se o ID só aparece uma vez (que é na definição do merchant), significa que não estava na lista do customer
            txArray[IDX_UNKNOWN_MERCHANT] = 1f;
        }

        if (temLastTransaction && lastEpoch != 0L) {
            txArray[IDX_MIN_SINCE_LAST] = Math.max(0, (txEpoch - lastEpoch) / 60L);
            txArray[IDX_KM_FROM_LAST] = kmFromCurrent;
        }

        return txArray;
    }

    // --- Helpers de Parsing de alta performance sem alocação ---

    private static float parseNextFloat(String j, int start) {
        int end = start;
        while (end < j.length() && j.charAt(end) != ',' && j.charAt(end) != '}' && j.charAt(end) != '\n' && j.charAt(end) != '\r') {
            end++;
        }
        return Float.parseFloat(j.substring(start, end).trim());
    }

    private static int parseNextInt(String j, int start) {
        return (int) parseNextFloat(j, start);
    }

    private static int tsHour(String s, int offset) {
        return (s.charAt(offset + 11) - '0') * 10 + (s.charAt(offset + 12) - '0');
    }

    private static int tsDayOfWeek(String s, int offset) {
        int y = digits(s, offset, offset + 4);
        int m = digits(s, offset + 5, offset + 7);
        int d = digits(s, offset + 8, offset + 10);
        final int[] t = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
        if (m < 3) y--;
        int dow = (y + y / 4 - y / 100 + y / 400 + t[m - 1] + d) % 7;

        return switch (dow) {
            case 1 -> 0; // Seg
            case 2 -> 1; // Ter
            case 3 -> 2; // Qua
            case 4 -> 3; // Qui
            case 5 -> 4; // Sex
            case 6 -> 5; // Sáb
            case 0 -> 6; // Dom
            default -> 0;
        };
    }

    static long tsEpochSeconds(String s, int offset) {
        int y = digits(s, offset, offset + 4);
        int m = digits(s, offset + 5, offset + 7);
        int d = digits(s, offset + 8, offset + 10);
        int h = digits(s, offset + 11, offset + 13);
        int min = digits(s, offset + 14, offset + 16);
        int sec = digits(s, offset + 17, offset + 19);
        if (m <= 2) {
            y--;
            m += 9;
        } else {
            m -= 3;
        }

        long era = (y >= 0 ? y : y - 399) / 400;
        int yoe = (int) (y - era * 400);
        int doy = (153 * m + 2) / 5 + d - 1;
        int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        long days = era * 146097L + doe - 719468L;
        return days * 86400L + h * 3600L + min * 60L + sec;
    }

    private static int digits(String s, int start, int end) {
        int v = 0;
        for (int i = start; i < end; i++) {
            v = v * 10 + (s.charAt(i) - '0');
        }
        return v;
    }
}

