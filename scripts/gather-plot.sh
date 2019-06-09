#!/bin/bash
set -euo pipefail

FILES='samples/*.factorized'

echo $FILES

bin/plot $FILES --out samples/data.csv
