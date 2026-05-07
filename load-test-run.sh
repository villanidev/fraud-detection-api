#!/usr/bin/env bash

export K6_NO_USAGE_REPORT=true

k6 run load-test/test.js > /dev/null 2>&1
cat load-test/results.json | jq
