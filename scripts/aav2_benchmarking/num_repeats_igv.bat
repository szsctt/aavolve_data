new
snapshotDirectory out/aav2_benchmarking/num_repeats_igv/
maxPanelHeight 1500
genome data/references/wtAAV2.fa
goto AAV2:1-2208
load out/aav2_benchmarking/consensus_aln/0_repeats.fl.bam
load out/aav2_benchmarking/consensus_aln/1_repeats.fl.bam
load out/aav2_benchmarking/consensus_aln/3_repeats.fl.bam
load out/aav2_benchmarking/consensus_aln/5_repeats.fl.bam
load out/aav2_benchmarking/consensus_aln/10_repeats.fl.bam
setTrackHeight 0 '0_repeats.fl.bam Coverage'
setTrackHeight 0 '1_repeats.fl.bam Coverage'
setTrackHeight 0 '3_repeats.fl.bam Coverage'
setTrackHeight 0 '5_repeats.fl.bam Coverage'
setTrackHeight 0 '10_repeats.fl.bam Coverage'
#setTrackHeight 0 'Sequence'
#squish
snapshot np-aav2-cc_num_repeats.png
snapshot np-aav2-cc_num_repeats.svg
exit

