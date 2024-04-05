#!/bin/bash
set -euo pipefail

# inputs and outputs
AAV2="data/references/wtAAV2.fa"
AAV2_LEN=2208
READS="out/c3poa/aav2_np-cc/split/R2C2_Consensus.fasta.gz"
OUTDIR=out/aav2_benchmarking
mkdir -p $OUTDIR

# singlarity containers
SINGDIR='singularity'
mkdir -p $SINGDIR
PYIMG="docker://szsctt/lr_pybio:py310"
PYSIF="${SINGDIR}/pybio.sif"
MMIMG="docker://quay.io/biocontainers/minimap2:2.28--he4a0461_0"
MMSIF="${SINGDIR}/minimap2.sif"
SAMIMG="docker://quay.io/biocontainers/samtools:1.19.2--h50ea8bc_1"
SAMSIF="${SINGDIR}/samtools.sif"
IGVIMG="docker://quay.io/biocontainers/igv:2.17.3--hdfd78af_0"
IGVSIF="${SINGDIR}/igv.sif"
RIMG="docker://szsctt/rbio:v1.7"
RSIF="${SINGDIR}/rbio.sif"

# pull singularity containers
if [ ! -e $PYSIF ]; then
    singularity pull $PYSIF $PYIMG
fi

if [ ! -e $MMSIF ]; then
    singularity pull $MMSIF $MMIMG
fi

if [ ! -e $SAMSIF ]; then
    singularity pull $SAMSIF $SAMIMG
fi

if [ ! -e $IGVSIF ]; then
    singularity pull $IGVSIF $IGVIMG
fi

if [ ! -e $RSIF ]; then
    singularity pull $RSIF $RIMG
fi

# separate consensus reads by number of repeats
SPLITREADS="${OUTDIR}/consensus_reads"
mkdir -p $SPLITREADS
singularity exec \
    $PYSIF \
    python scripts/aav2_benchmarking/filter_consensus_repeats.py \
    --input $READS \
    --output $SPLITREADS

# align consensus reads to reference
ALN="${OUTDIR}/consensus_aln"
VAR="${OUTDIR}/consensus_var"
mkdir -p $ALN $VAR
for f in $(ls $SPLITREADS); do
    BASE=$(basename $f .fa)

    # align reads to reference
    singularity exec $MMSIF \
        minimap2 -ax map-ont --MD $AAV2 $SPLITREADS/$f | \
    singularity exec $SAMSIF \
        samtools sort -o $ALN/${BASE}.bam -
    singularity exec $SAMSIF \
        samtools index $ALN/${BASE}.bam

    # get reads that cover the full length of the reference
    singularity exec $SAMSIF \
        samtools view -h $ALN/$BASE.bam AAV2:1-2 | \
    singularity exec $SAMSIF \
        samtools sort -o $ALN/$BASE.tmp.bam -
    singularity exec $SAMSIF \
        samtools index $ALN/${BASE}.tmp.bam
    singularity exec $SAMSIF \
        samtools view -h $ALN/${BASE}.tmp.bam AAV2:2207-2208 | \
    singularity exec $SAMSIF \
        samtools sort -o $ALN/${BASE}.fl.bam -
    singularity exec $SAMSIF \
        samtools index $ALN/${BASE}.fl.bam
    rm $ALN/${BASE}.tmp.bam $ALN/${BASE}.tmp.bam.bai

    # get variants
    singularity exec $PYSIF \
        python -m aavolve.extract_features_from_sam \
         -i $ALN/${BASE}.fl.bam \
         -r $AAV2 \
         -b 0 \
         -a -1 \
         -o $VAR/${BASE}.vars.tsv \
         -O $VAR/${BASE}.reads.tsv

    # caclculate rates of indels and substitutions
    singularity exec $PYSIF \
        python scripts/aav2_benchmarking/counts_vars.py \
        --variants $VAR/${BASE}.vars.tsv \
        --read-ids $VAR/${BASE}.reads.tsv \
        --reference-length  $AAV2_LEN \
        --output $VAR/${BASE}.rates.tsv

done

# create IGV image
singularity exec \
    $IGVSIF \
    igv -b scripts/aav2_benchmarking/num_repeats_igv.bat


# combine error rates for each number of repeats
rm $VAR/all.rates.tsv
awk 'NR==1 || FNR>1 ' $VAR/*.rates.tsv > $VAR/all.rates.tsv


# make plots
singularity exec $RSIF \
    Rscript scripts/aav2_benchmarking/plot_accuracy.R