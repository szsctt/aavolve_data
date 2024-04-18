new
snapshotDirectory out/sh2389/igv/
maxPanelHeight 1500
genome data/references/wtAAV2_N496D.fa
goto AAV2:1-2208
load out/aligned/aav2389.bam
load out/sh2389/filtered/r0_np-cc.bam
load out/sh2389/filtered/r1_np-cc.bam
load out/sh2389/filtered/r5_np-cc.bam
setTrackHeight 0 'aav2389.bam Coverage'
setTrackHeight 0 'r0_np-cc.bam Coverage'
setTrackHeight 0 'r1_np-cc.bam Coverage'
setTrackHeight 0 'r5_np-cc.bam Coverage'
#setTrackHeight 0 'Sequence'
#squish
snapshot sh2389_rounds.png
snapshot sh2389_rounds.svg 
#exit

