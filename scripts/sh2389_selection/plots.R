# 1 column = 86.50 mm; 2 columns = 178 mm

library(tidyverse)
library(cowplot)

out_dir <- here::here("out/sh2389/plots")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ggplot2::theme_set(ggplot2::theme_minimal(base_size = 8))

pos_num <- function(v) {
    return(stringr::str_extract(v, "^\\d+"))
}

var_to_pos <- function(v) {
    pos <- pos_num(v) %>% 
        as.numeric()
    pos_fct <- dplyr::case_when(
        stringr::str_detect(v, "sub") ~ pos_num(v),
        stringr::str_detect(v, "ins") ~ glue::glue("{pos_num(v)} "),
        stringr::str_detect(v, "del") ~ glue::glue("{pos_num(v)}  ")
    )
    pos_fct <- forcats::fct_reorder(pos_fct, pos)
    return(pos_fct)
}

get_n_rows <- function(filename) {
    n_rows <- system(paste("zcat", filename, "| wc -l"), intern = TRUE)
    return(as.numeric(n_rows)-1)

}

load_library <- function(base, n_reads=5000) {
  
    aav_names <- c("AAV2_N496D", "AAV3b", "AAV8", "AAV9", "ambiguous")
    pal <- scales::hue_pal()(5)
    names(pal) <- aav_names
    
    # read assigned parents and counts
    assigned_file <- file.path("out", "parents", "counts", glue::glue("{base}_parent-counts.tsv.gz"))
    assigned <- readr::read_tsv(file = assigned_file, n_max = n_reads) %>%
    # reverse order and assign ids
        dplyr::slice(n():1) %>%
        dplyr::mutate(id = dplyr::row_number())

    # total number of reads
    n_reads <- get_n_rows(file.path("out", "parents", "assigned", glue::glue("{base}_assigned-parents.tsv.gz")))

    # read breakpoints
    breakpoints <- readr::read_tsv(file = file.path("out", "parents", "breaks", glue::glue("{base}-pervar.tsv.gz")))

    p1 <- assigned %>%
        # pivot longer
        dplyr::select(-count) %>%
        tidyr::pivot_longer(cols = -id, names_to = "variant", values_to = "parent") %>% 
        # get position of each variant
        dplyr::mutate(pos = var_to_pos(variant)) %>%
        # for multiple parents, just put "ambiguous"
        dplyr::mutate(parent = case_when(
            stringr::str_detect(parent, ",") ~ "ambiguous",
            TRUE ~ parent
        )) %>%
        ggplot2::ggplot(ggplot2::aes(x=pos, y=id, fill=parent)) + 
            ggplot2::geom_raster() +
        ggplot2::labs(x="Reference position", y="Read", fill="Parent") +
        ggplot2::scale_fill_manual(values=pal) +
        ggplot2::theme(axis.title = element_blank(), axis.text = element_blank())

    legend <- cowplot::get_legend(p1)
    p1 <- p1 + ggplot2::theme(legend.position="none") 


    # counts
    p2 <- assigned %>% 
        ggplot2::ggplot(ggplot2::aes(x=count, y=id, group=1)) +
        ggplot2::geom_point() +
        ggplot2::geom_line() +
        ggplot2::labs(x="Count", y="Read") +
        ggplot2::scale_x_log10() +
        ggplot2::theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))

    # fraction of breakpoints
    breakpoints <- breakpoints %>% 
        dplyr::mutate(pos = var_to_pos(location)) %>%
        dplyr::mutate(fraction = breakpoints / n_reads) 
    p3 <- breakpoints %>%
        ggplot2::ggplot(ggplot2::aes(x = pos, y=fraction)) +
        ggplot2::geom_col() +
        ggplot2::labs(x="Reference position", y="Read\nfraction") +
        ggplot2::scale_x_discrete(breaks = levels(breakpoints$pos)[c(T, rep(F, 50))]) +
        ggplot2::theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))

    p <- cowplot::plot_grid(p2, p1, NULL, p3, 
                nrow = 2, ncol=2,  rel_widths = c(0.3, 1), rel_heights=c(1, 0.3))
    cowplot::plot_grid(p, legend, nrow=1, rel_widths=c(1, 0.1))

    return(list(n_reads = n_reads, assigned=assigned, 
                    breakpoints=breakpoints, 
                    p1=p1, p2=p2, p3=p3, lgd=legend, p=p))
}


r0_vis <- load_library("r0_np-cc")
r0_vis$p
r1_vis <- load_library("r1_np-cc")
r1_vis$p
r5_vis <- load_library("r5_np-cc")
r5_vis$p

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
