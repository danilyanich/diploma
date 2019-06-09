#!/bin/bash
set -euo pipefail

FILES=$(ls samples/*.parsed)
METHODS=(MU ALS_NORM ALS_NNLS ALS_LSQR)
PRECISION='10e-2'
RANK='5'

for file in $FILES; do
  for method in "${METHODS[@]}"; do
    echo $file $method
    bin/factorize $file --eps $PRECISION --rank $RANK --method $method
  done;
done;
