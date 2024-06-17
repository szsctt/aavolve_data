#!/bin/bash
set -euo pipefail


# inputs and outputs
AAV2="data/references/wtAAV2_N496D.fa"
AAV2_LEN=2208
OUTDIR=out/sh2389
mkdir -p $OUTDIR

# singularity containers
SINGDIR='singularity'
mkdir -p $SINGDIR
SAMIMG="docker://quay.io/biocontainers/samtools:1.19.2--h50ea8bc_1"
SAMSIF="${SINGDIR}/samtools.sif"
IGVIMG="docker://quay.io/biocontainers/igv:2.17.3--hdfd78af_0"
IGVSIF="${SINGDIR}/igv.sif"
PYIMG="docker://szsctt/lr_py:latest"
PYSIF="${SINGDIR}/lr_py.sif"

# pull containers
if [ ! -e $SAMSIF ]; then
    singularity pull $SAMSIF $SAMIMG
fi

if [ ! -e $IGVSIF ]; then
    singularity pull $IGVSIF $IGVIMG
fi

if [ ! -e $PYSIF ]; then
    singularity pull $PYSIF $PYIMG
fi

# filter alignments for reads that appear in pivoted data only
PIVOTDIR="out/variants/pivot"
ALNDIR="out/aligned"
FILTDIR="${OUTDIR}/filtered"
mkdir -p $FILTDIR
declare -a FILES=("r0_np-cc" "r1_np-cc" "r5_np-cc")
for file in "${FILES[@]}"; do 
    echo "Filtering alignments for $file"
    PIVOT="${PIVOTDIR}/${file}_seq.tsv.gz"
    ALN="${ALNDIR}/${file}.bam"
    READS="${FILTDIR}/${file}_reads.txt"
    FILT="${FILTDIR}/${file}.bam"
    FILTTMP="${FILTDIR}/${file}_tmp.sam"
    # get reads from pivot
    zcat $PIVOT | cut -f1 | sort | uniq > $READS
    
    # filter, sort, index alignment
    singularity exec $SAMSIF samtools view -H $ALN > $FILTTMP
    singularity exec $SAMSIF samtools view $ALN | grep -Ff $READS >> $FILTTMP
    singularity exec $SAMSIF samtools sort -o $FILT $FILTTMP
    singularity exec $SAMSIF samtools index $FILT
    rm $FILTTMP
done

# create IGV image
mkdir -p ${OUTDIR}/igv
singularity exec \
    $IGVSIF \
    igv -b scripts/sh2389_selection/FRG_selection_igv.bat

# caculate maximum possible library size
singularity exec $PYSIF \
    python scripts/sh2389_selection/calc_theoretical_lib_size.py \
      --parents out/variants/parents/aav2389.tsv.gz \
      --group-vars --group-dist 1

# make plots
singularity exec $RSIF \
    Rscript scripts/sh2389_selection/plots.R

singularity exec $RSIF \
    Rscript scripts/sh2389_selection/comp_pacbio.R
