package villani.dev.fraud;

public class TransactionRequestBytes {

    public static void toRequestArray(byte[] json, int len, float[] tx) {
        // fills tx[0..13] following the same layout as TransactionRequest.toRequestArray
        // zero-allocation, operates only on provided byte[] and primitive locals

        int txStart = sectionStart(json, len, 0, "transaction");
        float txAmount = extractFloat(json, len, txStart, "amount");
        int txInstallments = (int) extractFloat(json, len, txStart, "installments");
        long reqAtRange = extractQuotedRange(json, len, txStart, "requested_at");
        int reqAtStart = (int)(reqAtRange >>> 32);
        int reqAtEnd = (int)(reqAtRange & 0xffffffffL);
        int txHour = tsHour(json, reqAtStart);
        int txDow = tsDayOfWeek(json, reqAtStart);
        long txEpoch = tsEpochSeconds(json, reqAtStart);

        int custStart = sectionStart(json, len, 0, "customer");
        float custAvgAmount = extractFloat(json, len, custStart, "avg_amount");
        int custTxCount = (int) extractFloat(json, len, custStart, "tx_count_24h");

        int merchStart = sectionStart(json, len, 0, "merchant");
        long merchIdRange = extractQuotedRange(json, len, merchStart, "id");
        int merchIdS = (int)(merchIdRange >>> 32);
        int merchIdE = (int)(merchIdRange & 0xffffffffL);
        int mccCode = extractIntStr(json, len, merchStart, "mcc");
        float merchAvg = extractFloat(json, len, merchStart, "avg_amount");

        boolean unknownMerchant = !merchantIsKnown(json, len, custStart, "known_merchants", json, merchIdS, merchIdE);

        int termStart = sectionStart(json, len, 0, "terminal");
        boolean isOnline = extractBool(json, len, termStart, "is_online");
        boolean cardPresent = extractBool(json, len, termStart, "card_present");
        float kmFromHome = extractFloat(json, len, termStart, "km_from_home");

        // last_transaction optional
        int lastIdx = indexOfKey(json, len, 0, "last_transaction");
        long lastEpoch = 0;
        float kmFromCurrent = 0f;
        if (lastIdx >= 0) {
            int colon = indexOf(json, len, lastIdx, ':');
            colon++;
            while (colon < len) {
                byte b = json[colon];
                if (b == ' ' || b == '\n' || b == '\r') { colon++; continue; }
                break;
            }
            if (colon < len && json[colon] == '{') {
                long tsRange = extractQuotedRange(json, len, colon, "timestamp");
                int tsStart = (int)(tsRange >>> 32);
                lastEpoch = tsEpochSeconds(json, tsStart);
                kmFromCurrent = extractFloat(json, len, colon, "km_from_current");
            }
        }

        // populate tx array
        tx[0] = txAmount;
        tx[1] = txInstallments;
        tx[2] = custAvgAmount > 0f ? (txAmount / custAvgAmount) : 0f;
        tx[3] = txHour;
        tx[4] = txDow;
        if (lastEpoch == 0L && kmFromCurrent == 0f) {
            tx[5] = -1f;
            tx[6] = -1f;
        } else {
            long minutes = Math.max(0L, (txEpoch - lastEpoch) / 60L);
            tx[5] = (float) minutes;
            tx[6] = kmFromCurrent;
        }
        tx[7] = kmFromHome;
        tx[8] = custTxCount;
        tx[9] = isOnline ? 1f : 0f;
        tx[10] = cardPresent ? 1f : 0f;
        tx[11] = unknownMerchant ? 1f : 0f;
        tx[12] = mccCode;
        tx[13] = merchAvg;
    }

    // ---------- helpers (byte[] based, zero-allocation) -----------------

    private static int sectionStart(byte[] j, int len, int from, String key) {
        int k = indexOfKey(j, len, from, key);
        if (k < 0) return -1;
        int colon = indexOf(j, len, k, ':');
        return indexOf(j, len, colon, '{');
    }

    private static int indexOfKey(byte[] j, int len, int from, String key) {
        // find '"' + key + '"' starting at from
        int k = from;
        int keyLen = key.length();
        outer: for (; k < len - keyLen - 2; k++) {
            if (j[k] != '"') continue;
            int p = k + 1;
            for (int i = 0; i < keyLen; i++) {
                if (p + i >= len) return -1;
                if (j[p + i] != (byte) key.charAt(i)) continue outer;
            }
            int q = p + keyLen;
            if (q < len && j[q] == '"') return k;
        }
        return -1;
    }

    private static int indexOf(byte[] j, int len, int from, char c) {
        for (int i = Math.max(0, from); i < len; i++) if (j[i] == (byte) c) return i;
        return -1;
    }

    private static long extractQuotedRange(byte[] j, int len, int from, String key) {
        int k = indexOfKey(j, len, from, key);
        if (k < 0) return 0L;
        int colon = indexOf(j, len, k, ':');
        int q1 = indexOf(j, len, colon + 1, '"');
        int q2 = indexOf(j, len, q1 + 1, '"');
        return (((long)(q1 + 1)) << 32) | (q2 & 0xffffffffL);
    }

    private static float extractFloat(byte[] j, int len, int from, String key) {
        int k = indexOfKey(j, len, from, key);
        int start = indexOf(j, len, k, ':');
        start++;
        while (start < len) { byte b = j[start]; if (b == ' ') { start++; continue; } break; }
        int end = start;
        while (end < len) {
            byte c = j[end];
            if (c == ',' || c == '}' || c == '\n' || c == '\r') break;
            end++;
        }
        return parseFloat(j, start, end);
    }

    private static int extractIntStr(byte[] j, int len, int from, String key) {
        int k = indexOfKey(j, len, from, key);
        int q1 = indexOf(j, len, indexOf(j, len, k, ':') + 1, '"') + 1;
        int v = 0;
        for (int p = q1; p < len; p++) {
            byte c = j[p];
            if (c == '"') break;
            v = v * 10 + (c - '0');
        }
        return v;
    }

    private static boolean extractBool(byte[] j, int len, int from, String key) {
        int k = indexOfKey(j, len, from, key);
        int start = indexOf(j, len, k, ':') + 1;
        while (start < len && j[start] == ' ') start++;
        return j[start] == 't';
    }

    private static boolean merchantIsKnown(byte[] j, int len, int from, String key, byte[] whole, int targetS, int targetE) {
        int k = indexOfKey(j, len, from, key);
        if (k < 0) return false;
        int bracket = indexOf(j, len, k, '[');
        int end = indexOf(j, len, bracket, ']');
        int pos = bracket + 1;
        int tLen = targetE - targetS;
        while (pos < end) {
            int q1 = indexOf(whole, len, pos, '"');
            if (q1 < 0 || q1 >= end) break;
            int q2 = indexOf(whole, len, q1 + 1, '"');
            int candidateLen = q2 - q1 - 1;
            if (candidateLen == tLen) {
                boolean eq = true;
                for (int i = 0; i < tLen; i++) if (whole[q1 + 1 + i] != whole[targetS + i]) { eq = false; break; }
                if (eq) return true;
            }
            pos = q2 + 1;
        }
        return false;
    }

    // --------- numeric parsing from bytes ---------------------------------

    private static float parseFloat(byte[] b, int s, int e) {
        // parse optional sign, integer part, optional fraction
        boolean neg = false;
        int i = s;
        if (i < e && b[i] == '-') { neg = true; i++; }
        long intPart = 0;
        while (i < e) {
            byte c = b[i];
            if (c >= '0' && c <= '9') { intPart = intPart * 10 + (c - '0'); i++; continue; }
            break;
        }
        double value = intPart;
        if (i < e && b[i] == '.') {
            i++;
            double place = 0.1;
            while (i < e) {
                byte c = b[i];
                if (c >= '0' && c <= '9') { value += (c - '0') * place; place *= 0.1; i++; continue; }
                break;
            }
        }
        return (float) (neg ? -value : value);
    }

    // timestamp helpers operate directly on bytes at given start (expects format 2026-03-11T18:45:53Z)
    private static int tsHour(byte[] j, int start) {
        return (j[start + 11] - '0') * 10 + (j[start + 12] - '0');
    }

    private static int tsDayOfWeek(byte[] j, int start) {
        int y = digits(j, start + 0, start + 4);
        int m = digits(j, start + 5, start + 7);
        int d = digits(j, start + 8, start + 10);
        final int[] t = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
        if (m < 3) y--;
        int dow = (y + y / 4 - y / 100 + y / 400 + t[m - 1] + d) % 7;
        return switch (dow) {
            case 1 -> 0;
            case 2 -> 1;
            case 3 -> 2;
            case 4 -> 3;
            case 5 -> 4;
            case 6 -> 5;
            case 0 -> 6;
            default -> 0;
        };
    }

    private static long tsEpochSeconds(byte[] j, int start) {
        int y = digits(j, start + 0, start + 4);
        int m = digits(j, start + 5, start + 7);
        int d = digits(j, start + 8, start + 10);
        int h = digits(j, start + 11, start + 13);
        int min = digits(j, start + 14, start + 16);
        int sec = digits(j, start + 17, start + 19);
        if (m <= 2) { y--; m += 9; } else { m -= 3; }
        long era = (y >= 0 ? y : y - 399) / 400;
        int yoe = (int)(y - era * 400);
        int doy = (153 * m + 2) / 5 + d - 1;
        int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        long days = era * 146097L + doe - 719468L;
        return days * 86400L + h * 3600L + min * 60L + sec;
    }

    private static int digits(byte[] j, int s, int e) {
        int v = 0;
        for (int i = s; i < e; i++) v = v * 10 + (j[i] - '0');
        return v;
    }
}


