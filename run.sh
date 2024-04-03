#!/bin/bash
set -euo pipefail

# run AAVolve 
# snakemake --use-singularity --cores 32 --config samples=config/samples.csv

# run analysis of AAV2 data
bash scripts/aav2_benchmarking/aav2_benchmarking.sh
