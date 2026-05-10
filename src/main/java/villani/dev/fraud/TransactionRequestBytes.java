package villani.dev.fraud;

import java.nio.charset.StandardCharsets;

/**
 * JSON parser zero‑alocação para o endpoint de detecção de fraude.
 * Converte o corpo da requisição (byte[]) diretamente nos 14 valores do vetor,
 * aplicando as normalizações e usando apenas arrays primitivos.
 *
 */
public final class TransactionRequestBytes {

    // Constantes auxiliares (bytes das chaves JSON)
    private static final byte[] KEY_TRANSACTION = "transaction".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_AMOUNT      = "amount".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_INSTALLMENTS= "installments".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_REQUESTED_AT= "requested_at".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_CUSTOMER    = "customer".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_AVG_AMOUNT  = "avg_amount".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_TX_COUNT_24H= "tx_count_24h".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_KNOWN_MERCHANTS = "known_merchants".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_MERCHANT    = "merchant".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_ID          = "id".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_MCC         = "mcc".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_MERCHANT_AVG = "avg_amount".getBytes(StandardCharsets.UTF_8); // reutilizado
    private static final byte[] KEY_TERMINAL    = "terminal".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_IS_ONLINE   = "is_online".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_CARD_PRESENT= "card_present".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_KM_FROM_HOME= "km_from_home".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_LAST_TRANSACTION = "last_transaction".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_TIMESTAMP   = "timestamp".getBytes(StandardCharsets.UTF_8);
    private static final byte[] KEY_KM_FROM_CURRENT = "km_from_current".getBytes(StandardCharsets.UTF_8);

    public static float[] toRequestArray(byte[] jsonBytes) {
        float[] out = new float[14];

        // ── Bloco transaction ──
        int txStart = skipToValueStart(jsonBytes, 0, KEY_TRANSACTION);
        float txAmount = extractFloatField(jsonBytes, txStart, KEY_AMOUNT);
        int txInstallments = extractIntField(jsonBytes, txStart, KEY_INSTALLMENTS);
        int tsStart = findFieldValueStart(jsonBytes, txStart, KEY_REQUESTED_AT);
        int txHour = extractHour(jsonBytes, tsStart);
        int txDow  = extractDayOfWeek(jsonBytes, tsStart);
        long txEpoch = extractEpochSeconds(jsonBytes, tsStart);

        // ── Bloco customer ──
        int custStart = skipToValueStart(jsonBytes, 0, KEY_CUSTOMER);
        float custAvgAmount = extractFloatField(jsonBytes, custStart, KEY_AVG_AMOUNT);
        int custTxCount = extractIntField(jsonBytes, custStart, KEY_TX_COUNT_24H);

        // ── Bloco merchant ──
        int merchStart = skipToValueStart(jsonBytes, 0, KEY_MERCHANT);
        int merchIdBegin = findFieldValueStart(jsonBytes, merchStart, KEY_ID);
        int merchIdLen = skipQuotedString(jsonBytes, merchIdBegin) - merchIdBegin;
        boolean unknownMerchant = !merchantIsKnown(jsonBytes, custStart, merchIdBegin, merchIdLen);
        int mccCode = extractIntField(jsonBytes, merchStart, KEY_MCC); // corrigido: suporta string numérica
        float merchAvg = extractFloatField(jsonBytes, merchStart, KEY_MERCHANT_AVG);

        // ── Bloco terminal ──
        int termStart = skipToValueStart(jsonBytes, 0, KEY_TERMINAL);
        boolean isOnline = extractBoolField(jsonBytes, termStart, KEY_IS_ONLINE);
        boolean cardPresent = extractBoolField(jsonBytes, termStart, KEY_CARD_PRESENT);
        float kmFromHome = extractFloatField(jsonBytes, termStart, KEY_KM_FROM_HOME);

        // ── Bloco last_transaction ──
        boolean hasLastTx = false;
        long lastEpoch = 0;
        float kmFromCurrent = 0f;
        int ltKeyPos = indexOf(jsonBytes, 0, KEY_LAST_TRANSACTION);
        if (ltKeyPos >= 0) {
            int valStart = skipToValueStart(jsonBytes, ltKeyPos - KEY_LAST_TRANSACTION.length, KEY_LAST_TRANSACTION);
            if (valStart >= 0 && jsonBytes[valStart] == '{') {
                hasLastTx = true;
                int ltTsStart = findFieldValueStart(jsonBytes, valStart, KEY_TIMESTAMP);
                lastEpoch = extractEpochSeconds(jsonBytes, ltTsStart);
                kmFromCurrent = extractFloatField(jsonBytes, valStart, KEY_KM_FROM_CURRENT);
            }
        }

        // ── Preenche array (valores brutos) ──
        out[0] = txAmount;
        out[1] = txInstallments;
        out[2] = custAvgAmount > 0 ? (txAmount / custAvgAmount) : 0f; // amount_vs_avg bruto (ratio)
        out[3] = txHour;
        out[4] = txDow;
        out[5] = hasLastTx ? Math.max(0, (txEpoch - lastEpoch) / 60L) : -1f;
        out[6] = hasLastTx ? kmFromCurrent : -1f;
        out[7] = kmFromHome;
        out[8] = custTxCount;
        out[9] = isOnline ? 1f : 0f;
        out[10] = cardPresent ? 1f : 0f;
        out[11] = unknownMerchant ? 1f : 0f;
        out[12] = mccCode;           // inteiro puro, normalizado depois
        out[13] = merchAvg;

        return out;
    }

    // ──────────────────────── Utilitários de parsing ────────────────────────

    /** Avança até o valor associado a uma chave (objeto ou array). Retorna posição do '{' ou '['. */
    private static int skipToValueStart(byte[] buf, int from, byte[] key) {
        int pos = indexOf(buf, from, key);
        if (pos < 0) return -1;
        while (buf[pos] != ':') pos++;
        do pos++; while (pos < buf.length && isWhitespace(buf[pos]));
        return pos;
    }

    /** Encontra a chave dentro de um objeto e retorna o início do conteúdo do valor.
     *  Para valores string, retorna a posição do primeiro caractere depois das aspas. */
    private static int findFieldValueStart(byte[] buf, int objectStart, byte[] key) {
        int pos = indexOf(buf, objectStart, key);
        if (pos < 0) return -1;
        while (buf[pos] != ':') pos++;
        do pos++; while (pos < buf.length && isWhitespace(buf[pos]));
        if (buf[pos] == '"') pos++; // pula aspas de abertura
        return pos;
    }

    /** Extrai um float de um campo que pode ser número solto ou string numérica. */
    private static float extractFloatField(byte[] buf, int objectStart, byte[] key) {
        int valStart = findFieldValueStart(buf, objectStart, key);
        if (valStart < 0) return 0f;
        int valEnd = findValueEnd(buf, valStart);
        String numStr = new String(buf, valStart, valEnd - valStart, StandardCharsets.UTF_8);
        return Float.parseFloat(numStr);
    }

    /** Extrai um int de um campo (suporta string numérica). */
    private static int extractIntField(byte[] buf, int objectStart, byte[] key) {
        float val = extractFloatField(buf, objectStart, key);
        return (int) val;
    }

    /** Extrai booleano (true/false). */
    private static boolean extractBoolField(byte[] buf, int objectStart, byte[] key) {
        int valStart = findFieldValueStart(buf, objectStart, key);
        return valStart >= 0 && buf[valStart] == 't';
    }

    /** Encontra o fim do valor a partir do início do conteúdo.
     *  Se o valor original estava entre aspas (indicado pelo caractere anterior ser '"'),
     *  o terminador é '"'; caso contrário, terminadores são ',', '}', ']', whitespace. */
    private static int findValueEnd(byte[] buf, int contentStart) {
        boolean isQuoted = (contentStart > 0 && buf[contentStart - 1] == '"');
        int end = contentStart;
        while (end < buf.length) {
            byte b = buf[end];
            if (isQuoted) {
                if (b == '"') return end; // fecha aspas
            } else {
                if (b == ',' || b == '}' || b == ']' || isWhitespace(b)) return end;
            }
            end++;
        }
        return end;
    }

    private static boolean isWhitespace(byte b) {
        return b == ' ' || b == '\n' || b == '\r' || b == '\t';
    }

    /** IndexOf simples para byte[]. */
    private static int indexOf(byte[] buf, int from, byte[] key) {
        outer:
        for (int i = from; i <= buf.length - key.length; i++) {
            for (int k = 0; k < key.length; k++) {
                if (buf[i + k] != key[k]) continue outer;
            }
            return i;
        }
        return -1;
    }

    private static int extractHour(byte[] buf, int tsStart) {
        return (buf[tsStart + 11] - '0') * 10 + (buf[tsStart + 12] - '0');
    }

    private static int extractDayOfWeek(byte[] buf, int start) {
        int y = digits4(buf, start);
        int m = digits2(buf, start + 5);
        int d = digits2(buf, start + 8);
        if (m <= 2) { y--; m += 12; }
        int h = (d + (13 * (m + 1)) / 5 + y + y / 4 - y / 100 + y / 400) % 7;
        return switch (h) {
            case 0 -> 5; // Sáb
            case 1 -> 6; // Dom
            case 2 -> 0; // Seg
            case 3 -> 1; // Ter
            case 4 -> 2; // Qua
            case 5 -> 3; // Qui
            case 6 -> 4; // Sex
            default -> 0;
        };
    }

    private static long extractEpochSeconds(byte[] buf, int start) {
        int y = digits4(buf, start);
        int m = digits2(buf, start + 5);
        int d = digits2(buf, start + 8);
        int h = digits2(buf, start + 11);
        int min = digits2(buf, start + 14);
        int sec = digits2(buf, start + 17);
        if (m <= 2) { y--; m += 9; } else { m -= 3; }
        long era = (y >= 0 ? y : y - 399) / 400;
        int yoe = (int)(y - era * 400);
        int doy = (153 * m + 2) / 5 + d - 1;
        int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        long days = era * 146097L + doe - 719468L;
        return days * 86400L + h * 3600L + min * 60L + sec;
    }

    private static int digits4(byte[] buf, int start) {
        return (buf[start]-'0')*1000 + (buf[start+1]-'0')*100 + (buf[start+2]-'0')*10 + (buf[start+3]-'0');
    }
    private static int digits2(byte[] buf, int start) {
        return (buf[start]-'0')*10 + (buf[start+1]-'0');
    }

    private static boolean merchantIsKnown(byte[] buf, int custStart, int merchIdStart, int merchIdLen) {
        int pos = skipToValueStart(buf, custStart, KEY_KNOWN_MERCHANTS);
        if (pos < 0 || buf[pos] != '[') return false;
        int arrayEnd = findClosingBracket(buf, pos);
        pos++;
        while (pos < arrayEnd) {
            while (pos < arrayEnd && buf[pos] != '"') pos++;
            if (pos >= arrayEnd) break;
            int itemStart = pos + 1;
            int itemEnd = skipQuotedString(buf, itemStart);
            if (itemEnd - itemStart == merchIdLen && regionMatches(buf, itemStart, merchIdStart, merchIdLen)) {
                return true;
            }
            pos = itemEnd + 1;
        }
        return false;
    }

    private static int skipQuotedString(byte[] buf, int start) {
        int i = start;
        while (i < buf.length && buf[i] != '"') i++;
        return i;
    }

    private static int findClosingBracket(byte[] buf, int openPos) {
        int i = openPos + 1;
        while (i < buf.length && buf[i] != ']') i++;
        return i;
    }

    private static boolean regionMatches(byte[] buf, int pos, int otherStart, int len) {
        for (int i = 0; i < len; i++) {
            if (buf[pos + i] != buf[otherStart + i]) return false;
        }
        return true;
    }
}

