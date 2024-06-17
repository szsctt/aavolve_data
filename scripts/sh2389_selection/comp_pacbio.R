# make figure for comparing Nanopore, PacBio, and sanger sequecning

library(tidyverse)
library(cowplot)
here::i_am("scripts/sh2389_selection/comp_pacbio.R")
source("scripts/sh2389_selection/helpers.R")

ggplot2::theme_set(ggplot2::theme_minimal(base_size = 8))

out_dir <- here::here("out/sh2389/plots_r5_pb-vs-np")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# load r5 PacBio, Sanger and nanopore data

r5_pb <- load_library("r5_pb")
r5_sg <- load_library("r5_sanger")
r5_np_cc <- load_library("r5_np-cc")
r5_np <- load_library("r5_np")


ggsave(file.path(out_dir, "r5_pb.pdf"), r5_pb$p, width = 100, height = 90, units = "mm", scale = 1.5)
ggsave(file.path(out_dir, "r5_sg.pdf"), r5_sg$p, width = 100, height = 90, units = "mm", scale = 1.5)
ggsave(file.path(out_dir, "r5_np-cc.pdf"), r5_np_cc$p, width = 100, height = 90, units = "mm", scale = 1.5)
ggsave(file.path(out_dir, "r5_np.pdf"), r5_np$p, width = 100, height = 90, units = "mm", scale = 1.5)

# counts of reads at each stage of processing

rc <- read_tsv(list.files(here::here("out", "qc"), pattern = "*read-counts.tsv", full.names = TRUE), 
                    col_names = c("counted_file", "type", "count"), id = "count_file", show_col_types = F) %>%
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
  dplyr::filter(stringr::str_detect(count_file, "r5"))

# for pacbio data, input is consensus, not raw
rc <- rc %>%
    mutate(stage = case_when(
        sample == "r5_pb" & stage == "raw" ~ "consensus",
        TRUE ~ stage
    )) 

# present data as pivoted dataframe
rc %>% 
  select(sample, stage, count) %>% 
  mutate(stage = factor(stage, levels = c("raw", "consensus", "filtered consensus", "variant", "pivoted", "distinct aa", "distinct nt"))) %>%
  arrange(stage) %>%
  pivot_wider(names_from = stage, values_from = count) %>%
    write_tsv(file.path(out_dir, "read_counts.tsv"))


# load sequence counts at amino acid level
read_counts <- function(base) {
    aa_files <- list.files(here::here("out", "corrected", "counts"), 
                    pattern = "aa-seq-counts.tsv.gz", full.names=TRUE)
    aa_files <- aa_files[stringr::str_detect(aa_files, base)]
    df <- readr::read_tsv(aa_files, id = "filename") %>% 
        mutate(sample = basename(filename))  
}
r5_np_cc_counts <- read_counts("r5_np-cc") %>%
    mutate(tech = "R2C2")
r5_pb_counts <- read_counts("r5_pb") %>%
    mutate(tech = "PacBio")
r5_sanger_counts <- read_counts("r5_sanger") %>%
    mutate(tech = "Sanger")
r5_np_counts <- read_counts("r5_np_") %>%
    mutate(tech = "Nanopore")
counts <- bind_rows(r5_np_cc_counts, r5_pb_counts, r5_sanger_counts, r5_np_counts)

# number of distinct reads at amino acid level for each dataset
counts %>% 
    group_by(tech) %>%
    summarise(n_distinct = n()) %>%
    write_tsv(file.path(out_dir, "distinct_aa_counts.tsv"))


# venn diagram to show number of shared reads between datasets - all reads
a <- list(
    `Nanopore R2C2` = r5_np_cc_counts$sequence,
    #`Nanopore` = r5_np_counts$sequence,
    `PacBio` = r5_pb_counts$sequence,
    `Sanger` = r5_sanger_counts$sequence
)

p1 <- ggvenn::ggvenn(a, show_percentage = F)
ggsave(file.path(out_dir, "venn_all_reads.pdf"), p1, width = 100, height = 100, units = "mm", scale = 1.5)

# venn diagrame to show number of shared reads btween datasets - top 10% of reads
top_seq <- function(df, frac) {
    df %>% 
        slice_max(order_by = count, prop = frac, with_ties = TRUE) %>%
        pull(sequence) %>%
        return()
}
prop <- 0.2
a <- list(
    `Nanopore R2C2` = top_seq(r5_np_cc_counts, prop),
    #`Nanopore` = top_seq(r5_np_counts, prop),
    `PacBio` = top_seq(r5_pb_counts, prop),
    `Sanger` = top_seq(r5_sanger_counts, prop)
)
p2 <- ggvenn::ggvenn(a, show_percentage = F)
ggsave(file.path(out_dir, glue::glue("venn_top {prop}_reads.pdf")), p2, width = 100, height = 100, units = "mm", scale = 1.5)

# rank sequences in each dataset and plot ranks for each dataset vs each other
counts <- counts %>%
    group_by(tech) %>%
    mutate(rank = dense_rank(desc(count))) %>%
    ungroup() 

p3<- counts %>%
    select(tech, rank, sequence) %>%
    pivot_wider(names_from = tech, values_from = rank) %>%
    select(-sequence) %>%
    GGally::ggpairs()

ggsave(file.path(out_dir, "ranked_sequences.pdf"), p3, width = 200, height = 200, units = "mm", scale = 1.5)

# plot pacBio ranks vs nanopore R2C2 ranks
c_pb_np <- counts %>%
    filter(tech %in% c("PacBio", "R2C2")) %>%
    select(tech, rank, sequence) %>%
    pivot_wider(names_from = tech, values_from = rank) %>%
    filter(!is.na(PacBio) & !is.na(R2C2))
    
p4 <- c_pb_np %>%
    ggplot(aes(x = R2C2, y = PacBio)) +
    geom_point() +
    labs(x = "Nanopore R2C2 rank", y = "PacBio rank") 

ggsave(file.path(out_dir, "ranked_sequences_pb_vs_np-cc.pdf"), p4, width = 50, height = 50, units = "mm", scale = 1.5)


print(glue::glue("Correlation coeffecient between rank in PacBio and rank in R2C2: {cor(c_pb_np$R2C2, c_pb_np$PacBio)}"))
