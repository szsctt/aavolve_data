#!/bin/bash
set -euo pipefail

# encode sequences
python scripts/modelling/encode.py \
    --input out/corrected/counts/ \
    --output out/modelling/encoded/ \
    --n-test 50 \
    --weight-threshold 1 \
    --encoding-type onehot