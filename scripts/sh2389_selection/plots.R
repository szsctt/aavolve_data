# 1 column = 86.50 mm; 2 columns = 178 mm

library(tidyverse)
library(cowplot)
here::i_am("scripts/sh2389_selection/plots.R")
source("scripts/sh2389_selection/helpers.R")
out_dir <- here::here("out/sh2389/plots")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ggplot2::theme_set(ggplot2::theme_minimal(base_size = 8))


r0_vis <- load_library("r0_np-cc")
r0_vis$p
r1_vis <- load_library("r1_np-cc")
r1_vis$p
r5_vis <- load_library("r5_np-cc")
r5_vis$p

# in r5 library, how many breakpoints over 0.5?
print(glue::glue("Breakpoints in r5 with fraction > 0.5:"))
r5_vis$breakpoints %>%
  filter(fraction > 0.5)

# import counts of amino acid sequences
aa_files <- list.files(here::here("out", "corrected", "counts"), pattern = "aa-seq-counts.tsv.gz", full.names=TRUE)
# load only np-cc data
aa_files <- aa_files[stringr::str_detect(aa_files, "np-cc")]

# read counts
aa <- readr::read_tsv(aa_files, id = "filename") %>% 
  mutate(sample = basename(filename)) %>% 
  mutate(sample = stringr::str_replace(sample, "_aa-seq-counts.tsv.gz", "")) %>% 
  mutate(stage = stringr::str_extract(sample, "r\\d")) %>% 
  mutate(gel_extract = stringr::str_detect(sample, "gel-extract")) 

# combine replicates from the same round
aa <- aa %>% 
  group_by(stage, sequence) %>% 
  summarise(count = sum(count)) %>%
  group_by(stage) %>%
  mutate(frac = count / sum(count)) 

# plot ECDF of counts
p1 <- aa %>%
  mutate(stage = stringr::str_replace(stage, "r", "Round ")) %>%
  ggplot(aes(x = frac, color = stage)) +
  stat_ecdf(geom="step") +
  scale_x_log10() +
  labs(x = "Read\nfraction", y = "ECDF") +
  theme(legend.position = "bottom")

# pivot wider
aa_wide <- aa %>% 
  group_by(stage) %>%
  mutate(frac = count / sum(count)) %>%
  ungroup() %>%
  select(-count) %>%
  pivot_wider(names_from = stage, values_from = frac) %>%
  mutate(r1_div_r0 = r1/r0) %>% 
  mutate(r5_div_r0 = r5/r0) 

scaleFun <- function(x) sprintf("%.2f", x)

# plot r0 -> r1
plotChange <- function(stage) {
  if (stage == "r1") {
    div_col <- "r1_div_r0"
  } else if (stage == "r5") {
    div_col <- "r5_div_r0"
  } else {
    stop("Invalid stage")
  }

  p <- aa_wide %>%
    filter(!is.na(!!sym(div_col))) %>%
    slice_sample(n = 5000) %>%
    select(sequence, r0, !!sym(stage), !!sym(div_col)) %>%
    pivot_longer(one_of(c("r0", stage)), names_to = "stage", values_to = "frac") %>%
    ggplot(aes(x = stage, y = frac, group = sequence, color = !!sym(div_col))) +
    geom_line(alpha = 0.05) +
    geom_jitter(height = 0, width = 0.1) +
    scale_y_log10() +
    scale_color_viridis_c(trans = "log", labels = scaleFun) +
    labs(
      x = "Selection round", y = "Read fraction",
      color = "Fold change"
    ) +
    theme(legend.position = "bottom")
  return(p)
}
p2 <- plotChange("r1")
p3 <- plotChange("r5")


# heatmaps
load_heatmap_data <- function(fname) {
d <- here::here("out", "corrected", "dmat", fname) %>% 
  scan() %>% 
  matrix(ncol = 1000, byrow=TRUE)
  print(dim(d))
  return(d)
}

r0_distances <- load_heatmap_data("r0_np-cc_repeat_first_aa-seq.tsv.gz")
r1_distances <- load_heatmap_data("r1_np-cc_first_aa-seq.tsv.gz")
r5_distances <- load_heatmap_data("r5_np-cc_first_aa-seq.tsv.gz")

scale_max <- max(c(max(r0_distances), max(r1_distances), max(r5_distances)))
breaks <- seq(0, scale_max, length.out=20)
cols <- viridis::plasma(20)

make_heatmap <- function(d, legend=TRUE) {
  p <- ggplotify::as.ggplot(
    pheatmap::pheatmap(d, breaks=breaks, color=cols, cluster_cols = F, cluster_rows = F, legend=legend)
  )
  return(p)
}


plots <- purrr::map2(list(r0_distances, r1_distances, r5_distances), 
                    c(FALSE, FALSE, TRUE),
    ~ make_heatmap(.x, .y)
)


## final figure

row1 <- cowplot::ggdraw() +
    cowplot::draw_image(here::here("out/sh2389/igv/sh2389_rounds.png"), width=1, height=1)


row2 <- cowplot::plot_grid(r0_vis$p, r1_vis$p, r5_vis$p, r0_vis$lgd,
                        nrow=1, rel_widths = c(1, 1, 1, 0.1),
                        labels =c("B", "C", "D", ""))

row3 <- cowplot::plot_grid(plots[[1]], plots[[2]], plots[[3]], nrow=1, rel_widths = c(0.9, 0.9, 1), labels = c("E", "F", "G"))

row4 <- cowplot::plot_grid(p1, p2, p3, nrow=1, rel_widths = c(1, 1, 1), labels = c("H", "I", "J"))


p <- cowplot::plot_grid(row1, row2, row3, row4,
                        ncol=1, rel_heights = c(1, 1, 0.7, 1), labels = c("A", "", "", ""))

ggplot2::ggsave(file.path(out_dir, "selection_rounds.pdf"), plot = p, units = "mm", width = 178, scale=1.5)
ggplot2::ggsave(file.path(out_dir, "selection_rounds.png"), plot = p,  units = "mm", width = 178, dpi=300, scale=1.5)

# save individual figures

ggplot2::ggsave(file.path(out_dir, "selection_rounds_r0.pdf"), plot = r0_vis$p, units = "mm", width = 55, height = 55, scale=1.5)
ggplot2::ggsave(file.path(out_dir, "selection_rounds_r0.png"), plot = r0_vis$p, units = "mm", width = 55, height = 55, scale=1.5, dpi = 300)

ggplot2::ggsave(file.path(out_dir, "selection_rounds_r1.pdf"), plot = r1_vis$p, units = "mm", width = 55, height = 55, scale=1.5)
ggplot2::ggsave(file.path(out_dir, "selection_rounds_r1.png"), plot = r1_vis$p, units = "mm", width = 55, height = 55, scale=1.5, dpi = 300)

ggplot2::ggsave(file.path(out_dir, "selection_rounds_r5.pdf"), plot = r5_vis$p, units = "mm", width = 55, height = 55, scale=1.5)
ggplot2::ggsave(file.path(out_dir, "selection_rounds_r5.png"), plot = r5_vis$p, units = "mm", width = 55, height = 55, scale=1.5, dpi = 300)

ggplot2::ggsave(file.path(out_dir, "selection_rounds_parents_legend.pdf"), plot = r5_vis$lgd, units = "mm", width = 55, height = 55, scale=1.5)
ggplot2::ggsave(file.path(out_dir, "selection_rounds_parents_legend.png"), plot = r5_vis$lgd, units = "mm", width = 55, height = 55, scale=1.5, dpi = 300)

ggplot2::ggsave(file.path(out_dir, "selection_rounds_r0_heatmap.pdf"), plot = plots[[1]], units = "mm", width = 50, height = 55, scale=1.5)
ggplot2::ggsave(file.path(out_dir, "selection_rounds_r0_heatmap.png"), plot = plots[[1]], units = "mm", width = 55, height = 55, scale=1.5, dpi = 300)

ggplot2::ggsave(file.path(out_dir, "selection_rounds_r1_heatmap.pdf"), plot = plots[[2]], units = "mm", width = 50, height = 55, scale=1.5)
ggplot2::ggsave(file.path(out_dir, "selection_rounds_r1_heatmap.png"), plot = plots[[2]], units = "mm", width = 55, height = 55, scale=1.5, dpi = 300)

ggplot2::ggsave(file.path(out_dir, "selection_rounds_r5_heatmap_legend.pdf"), plot = plots[[3]], units = "mm", width = 55, height = 55, scale=1.5)
ggplot2::ggsave(file.path(out_dir, "selection_rounds_r5_heatmap_legend.png"), plot = plots[[3]], units = "mm", width = 55, height = 55, scale=1.5, dpi = 300)

# r5 heatmap without legend
r5_heatmap <- make_heatmap(r5_distances, FALSE)
ggplot2::ggsave(file.path(out_dir, "selection_rounds_r5_heatmap.pdf"), plot = r5_heatmap , units = "mm", width = 55, height = 55, scale=1.5)
ggplot2::ggsave(file.path(out_dir, "selection_rounds_r5_heatmap.png"), plot = r5_heatmap , units = "mm", width = 55, height = 55, scale=1.5, dpi = 300)

ggplot2::ggsave(file.path(out_dir, "selection_rounds_ecdf.pdf"), plot = p1, units = "mm", width = 55, height = 55, scale=1.5)
ggplot2::ggsave(file.path(out_dir, "selection_rounds_ecdf.png"), plot = p1, units = "mm", width = 55, height = 55, scale=1.5, dpi = 300)

ggplot2::ggsave(file.path(out_dir, "selection_rounds_r0_to_r1_frac.pdf"), plot = p2, units = "mm", width = 55, height = 55, scale=1.5)
ggplot2::ggsave(file.path(out_dir, "selection_rounds_r0_to_r1_frac.png"), plot = p2, units = "mm", width = 55, height = 55, scale=1.5, dpi = 300)

ggplot2::ggsave(file.path(out_dir, "selection_rounds_r0_to_r5_frac.pdf"), plot = p3, units = "mm", width = 55, height = 55, scale=1.5)
ggplot2::ggsave(file.path(out_dir, "selection_rounds_r0_to_r5_frac.png"), plot = p3, units = "mm", width = 55, height = 55, scale=1.5, dpi = 300)


#### read counts at each stage ####


# read in data
rc <- read_tsv(list.files(here::here("out", "qc"), pattern = "*read-counts.tsv", full.names = TRUE), 
                    col_names = c("counted_file", "type", "count"), id = "count_file", show_col_types = F) 

rc <- rc %>% 
  rowwise() %>% 
  mutate(sample = stringr::str_extract(basename(count_file), ".+(?=_read-counts.tsv)")) %>%
  dplyr::mutate(stage = case_when(
    stringr::str_detect(counted_file, "data") & type == "fastq" ~ "raw",
    stringr::str_detect(sample, "sanger") & type == "fasta" ~ "raw",
    stringr::str_detect(counted_file, "R2C2_Cons") ~ "consensus",
    stringr::str_detect(counted_file, "c3poa_filt") ~ "filtered consensus",
    type == "variant_tsv" ~ "variant",
    type == "pivoted_tsv" ~ "pivoted",
    type == "distinct_read_counts" & stringr::str_detect(counted_file, "aa-seq") ~ "distinct aa",
    type == "distinct_read_counts" ~ "distinct nt",
  )) %>% 
  mutate(stage = factor(stage, levels = c("raw", "consensus", "filtered consensus", "variant", "pivoted", "distinct aa", "distinct nt"))) 


# write to file
rc %>% 
  select(sample, stage, count) %>%
  pivot_wider(names_from = stage, values_from = count) %>%
  write_tsv(file.path(out_dir, "read_counts.tsv"))


# also aggregate over same sample and sequncing type for tex
rc_agg <- rc %>%
  mutate(seq_type = stringr::str_extract(sample, "sanger|pb|np-cc|np")) %>%
  mutate(sample_stage = stringr::str_extract(sample, "aav2|r0|r1|r5")) %>%
  group_by(seq_type, sample_stage, stage) %>%
  summarise(count = sum(count)) %>%
  ungroup() 

rc_agg  %>%
  pivot_wider(names_from = stage, values_from = count) %>%
  write_tsv(file.path(out_dir, "read_counts_agg.tsv"))

# also caculate amount of reads at pivoted stage relative to raw
rc_agg %>% 
  filter(stage == "raw" | stage == "pivoted") %>%
  pivot_wider(names_from = stage, values_from = count) %>%
  mutate(pivoted_div_raw = pivoted / raw) %>%
  write_tsv(file.path(out_dir, "read_counts_agg_pivoted.tsv"))

# make table with np-cc results for r0, r1 and r5 for supplementary table
rc_agg %>% 
  filter(seq_type == "np-cc") %>%
  filter(sample_stage %in% c("r0", "r1", "r5")) %>%
  select(sample_stage, stage, count) %>%
  pivot_wider(names_from = stage, values_from = count) %>%
  rename(all_of(c(`Selection round`="sample_stage", 
                  Raw='raw', 
                  Consensus='consensus',
                  `Filtered consensus`='filtered consensus',
                  `Filtered by reference coverage`='variant',
                  `Reads with identified parents`='pivoted',
                  `Distinct amino acids`='distinct aa',
                  `Distinct nucleotides`='distinct nt'))) %>%
  write_tsv(file.path(out_dir, "supp_read_counts_np-cc.tsv"))
