/**
 * Local load test — simulates the Rinha de Backend 2026 evaluation.
 *
 * Faithful to the real test in:
 *   - resource constraints (run via docker-compose with limits)
 *   - incremental VU ramp-up
 *   - payload structure identical to the real test
 *   - p99 thresholds matching scoring formula
 *
 * Usage:
 *   k6 run test/k6-local.js
 *   k6 run --env VUS=50 --env DURATION=120s test/k6-local.js
 *
 * Parameters (env vars):
 *   VUS      — peak virtual users (default: 50)
 *   DURATION — test duration in seconds (default: 90s)
 *   HOST     — target host (default: http://localhost:9999)
 */
import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";
import { SharedArray } from "k6/data";

// ── Scoring constants (from EVALUATION.md) ─────────────────────────────────
const K_LATENCY = 1000;
const T_MAX = 1000; // ms
const P99_MIN = 1; // ms
const P99_MAX = 2000; // ms — above this: cutoff = -3000
const FAILURE_CUTOFF = 0.15; // 15%
const W_FP = 1,
  W_FN = 3,
  W_ERR = 5;
const K_DET = 1000,
  EPS_MIN = 0.001,
  BETA = 300;

// ── Config ──────────────────────────────────────────────────────────────────
const HOST = __ENV.HOST || "http://localhost:9999";
//const VUS = parseInt(__ENV.VUS || "50");
//const DURATION = __ENV.DURATION || "60s";
//const TARGET_RPS = __ENV.RPS || 200; // Alvo de Requisições por Segundo
const TOTAL_REQS = 54100;
const RPS = 600; // Ajuste para a agressividade que você quer testar
const DURATION = `${Math.ceil(TOTAL_REQS / RPS)}s`; // Calcula duração com base no RPS e total de requisições

// ── Custom metrics ───────────────────────────────────────────────────────────
const httpErrors = new Counter("http_errors");
const fraudCount = new Counter("fraud_responses");
const legitCount = new Counter("legit_responses");

// ── k6 options ───────────────────────────────────────────────────────────────
/*export const options = {
  stages: [
    { duration: "10s", target: 5 }, // warm-up
    { duration: "20s", target: VUS }, // ramp up
    { duration: DURATION, target: VUS }, // sustained load
    { duration: "10s", target: 0 }, // ramp down
  ],
  thresholds: {
    // Mirrors the scoring thresholds from EVALUATION.md
    "http_req_duration{expected_response:true}": [
      "p(99)<2000", // avoid cutoff (-3000)
      "p(99)<200", // worth 1700 points
      "p(99)<100", // worth 2000 points
      "p(99)<50", // worth 2301 points
      "p(99)<10", // worth 3000 points (saturated range)
    ],
    http_req_failed: ["rate<0.15"], // avoid detection cutoff
  },
  summaryTrendStats: ["min", "avg", "med", "p(90)", "p(95)", "p(99)", "max"],
};*/

/*export const options = {
  scenarios: {
    sustained_load: {
      // Mantém uma taxa constante de requisições por segundo (RPS)
      executor: "constant-arrival-rate",
      rate: RPS,
      timeUnit: "1s",
      duration: DURATION, // Calcula duração com base no RPS e total de requisições
      preAllocatedVUs: 100,
      maxVUs: 250,
    },
  },
  thresholds: {
    // Thresholds agressivos baseados no seu EVALUATION.md
    "http_req_duration{expected_response:true}": [
      "p(99)<2000", // Cutoff de erro crítico
      "p(99)<50", // Meta para pontuação alta
      "p(99)<10", // Meta para pontuação máxima
    ],
    http_req_failed: ["rate<0.15"],
  },
  summaryTrendStats: ["min", "avg", "med", "p(90)", "p(95)", "p(99)", "max"],
};*/

export const options = {
  summaryTrendStats: ["p(99)"],
  systemTags: ["status", "method"],
  dns: {
    ttl: "5m",
    select: "roundRobin",
  },
  scenarios: {
    default: {
      executor: "ramping-arrival-rate",
      startRate: 1,
      timeUnit: "1s",
      preAllocatedVUs: 100,
      maxVUs: 250,
      gracefulStop: "10s",
      stages: [{ duration: "120s", target: 900 }],
    },
  },
};

// ── MCC codes present in the actual dataset ───────────────────────────────────
const MCC_CODES = [
  "5411",
  "5812",
  "5912",
  "5944",
  "7801",
  "7802",
  "7995",
  "4511",
  "5311",
  "5999",
];

// ── Merchant pool (realistic IDs) ─────────────────────────────────────────────
const MERCHANTS = Array.from(
  { length: 80 },
  (_, i) => `MERC-${String(i + 1).padStart(3, "0")}`,
);

// ── ISO-8601 date generator ───────────────────────────────────────────────────
function isoDate(year, month, day, hour, min, sec) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${year}-${pad(month)}-${pad(day)}T${pad(hour)}:${pad(min)}:${pad(sec)}Z`;
}

function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}
function randFloat(min, max, decimals = 2) {
  return parseFloat((Math.random() * (max - min) + min).toFixed(decimals));
}
function randChoice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

/**
 * Generates a randomized but realistic transaction payload.
 * Mix of fraud-like and legit-like patterns to exercise both code paths.
 */
function generatePayload(idx) {
  const isFraudLike = Math.random() < 0.45; // ~45% fraud rate matches dataset

  // Dates in 2026 (matches example-payloads.json)
  const month = randInt(1, 5);
  const day = randInt(1, 28);
  const hour = randInt(0, 23);
  const reqAt = isoDate(2026, month, day, hour, randInt(0, 59), randInt(0, 59));

  const custAvg = randFloat(50, 2000);
  // Fraud pattern: amount >> avg; legit: amount ≈ avg
  const amount = isFraudLike
    ? randFloat(custAvg * 3, Math.min(custAvg * 8, 10000))
    : randFloat(custAvg * 0.5, custAvg * 1.5);

  const merchantId = randChoice(MERCHANTS);
  const knownCount = randInt(1, 6);
  const knownMerch = Array.from({ length: knownCount }, () =>
    randChoice(MERCHANTS),
  );
  const mcc = isFraudLike
    ? randChoice(["7801", "7802", "7995", "5944"])
    : randChoice(MCC_CODES);
  const merchantAvg = randFloat(30, 5000);

  // last_transaction: absent ~25% of the time (matches the -1 sentinel in embeddings)
  const hasLastTx = Math.random() > 0.25;
  const lastTx = hasLastTx
    ? {
        timestamp: isoDate(
          2026,
          month,
          day,
          Math.max(0, hour - randInt(1, 6)),
          randInt(0, 59),
          0,
        ),
        km_from_current: randFloat(0, isFraudLike ? 500 : 50, 10),
      }
    : null;

  return {
    id: `tx-local-${idx}-${Date.now()}`,
    transaction: {
      amount: parseFloat(amount.toFixed(2)),
      installments: randInt(1, 12),
      requested_at: reqAt,
    },
    customer: {
      avg_amount: parseFloat(custAvg.toFixed(2)),
      tx_count_24h: isFraudLike ? randInt(5, 20) : randInt(1, 5),
      known_merchants: knownMerch,
    },
    merchant: {
      id: merchantId,
      mcc: mcc,
      avg_amount: parseFloat(merchantAvg.toFixed(2)),
    },
    terminal: {
      is_online: isFraudLike ? Math.random() > 0.3 : Math.random() > 0.7,
      card_present: !isFraudLike || Math.random() > 0.6,
      km_from_home: randFloat(0, isFraudLike ? 800 : 30, 10),
    },
    last_transaction: lastTx,
  };
}

// ── Main test function ────────────────────────────────────────────────────────
let requestIdx = 0;

export default function () {
  const payload = JSON.stringify(generatePayload(requestIdx++));

  const res = http.post(`${HOST}/fraud-score`, payload, {
    headers: { "Content-Type": "application/json" },
    timeout: "5s",
  });

  const ok = check(res, {
    "status 200": (r) => r.status === 200,
    "has approved field": (r) => r.body && r.body.includes('"approved"'),
    "has fraud_score field": (r) => r.body && r.body.includes('"fraud_score"'),
  });

  if (!ok || res.status !== 200) {
    httpErrors.add(1);
  } else {
    try {
      const body = JSON.parse(res.body);
      if (body.approved === false) fraudCount.add(1);
      else legitCount.add(1);
    } catch (_) {
      httpErrors.add(1);
    }
  }
}

// ── Summary with scoring calculation ─────────────────────────────────────────
export function handleSummary(data) {
  const p99ms = data.metrics["http_req_duration"].values["p(99)"];
  const errorRate = data.metrics["http_req_failed"].values["rate"];
  const totalReqs = data.metrics["http_reqs"].values["count"];
  const errCount = Math.round(
    data.metrics["http_errors"]?.values["count"] || 0,
  );
  const fraudTotal = data.metrics["fraud_responses"]?.values["count"] || 0;
  const legitTotal = data.metrics["legit_responses"]?.values["count"] || 0;

  // p99 score
  let p99Score;
  if (p99ms > P99_MAX) {
    p99Score = -3000;
  } else {
    p99Score = Math.min(
      3000,
      K_LATENCY * Math.log10(T_MAX / Math.max(p99ms, P99_MIN)),
    );
  }

  // Detection score (simplified — we don't have ground truth for generated payloads)
  // We can only measure HTTP errors; FP/FN unknown without labeled dataset
  const epsilon = errCount / Math.max(totalReqs, 1);
  let detScore;
  if (errorRate > FAILURE_CUTOFF) {
    detScore = -3000;
  } else if (errCount === 0) {
    detScore = "~3000 (0 errors — requires labeled dataset to compute exactly)";
  } else {
    const E = errCount * W_ERR; // worst case: all errors are HTTP errors
    const eps = Math.max(E / totalReqs, EPS_MIN);
    detScore = K_DET * Math.log10(1 / eps) - BETA * Math.log10(1 + E);
  }

  const lines = [
    "",
    "══════════════════════════════════════════════════════",
    "  RINHA DE BACKEND 2026 — Local Simulation Results",
    "══════════════════════════════════════════════════════",
    `  Total requests  : ${totalReqs}`,
    `  Fraud denied    : ${fraudTotal}`,
    `  Legit approved  : ${legitTotal}`,
    `  HTTP errors     : ${errCount}`,
    `  Error rate      : ${(errorRate * 100).toFixed(2)}%  (cutoff: >15%)`,
    "",
    `  p50   : ${data.metrics["http_req_duration"].values["med"].toFixed(2)}ms`,
    `  p90   : ${data.metrics["http_req_duration"].values["p(90)"].toFixed(2)}ms`,
    `  p95   : ${data.metrics["http_req_duration"].values["p(95)"].toFixed(2)}ms`,
    `  p99   : ${p99ms.toFixed(2)}ms  ← scoring input`,
    `  max   : ${data.metrics["http_req_duration"].values["max"].toFixed(2)}ms`,
    "",
    "  ── Scoring ──────────────────────────────────────────",
    `  p99_score       : ${typeof p99Score === "number" ? p99Score.toFixed(2) : p99Score}`,
    `  detection_score : ${typeof detScore === "number" ? detScore.toFixed(2) : detScore}`,
    `  final_score     : ${typeof p99Score === "number" && typeof detScore === "number" ? (p99Score + detScore).toFixed(2) : "~" + (p99Score + 3000).toFixed(0) + " (estimated, no ground truth)"}`,
    "",
    "  NOTE: detection_score is ESTIMATED (no labeled test data).",
    "  For exact detection accuracy, run integration tests.",
    "══════════════════════════════════════════════════════",
    "",
  ];

  console.log(lines.join("\n"));

  // Also write a JSON results file
  return {
    "test/results-local.json": JSON.stringify(
      {
        p99_ms: p99ms,
        p99_score: p99Score,
        error_rate: errorRate,
        total_requests: totalReqs,
        http_errors: errCount,
      },
      null,
      2,
    ),
    stdout: "\n",
  };
}
