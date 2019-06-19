#!/bin/bash
set -euo pipefail

bin/extract samples/problem.parsed samples/problem.5.ALS_NORM.factorized --method qr
bin/extract samples/problem.parsed samples/problem.1.ALS_NORM.factorized --method saliency
