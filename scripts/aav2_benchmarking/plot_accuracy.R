#!/usr/bin/env Rscript

#### setup ####

here::i_am("scripts/aav2_benchmarking/plot_accuracy.R")
library(magrittr)
library(patchwork)

aav2_length <- 2208
out_dir <- here::here("out/aav2_benchmarking/plots")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ggplot2::theme_set(ggplot2::theme_minimal(base_size = 9))

#### average error rates calculated with python script ####

# read in data with average error rates
errors <- readr::read_tsv(here::here("out/aav2_benchmarking/consensus_var/all.rates.tsv"))

# plot number of reads
p1 <- errors %>% 
    dplyr::mutate(read_frac = reads / sum(reads)) %>%
    ggplot2::ggplot(ggplot2::aes(x = repeats, y = read_frac)) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::labs(
        x = "Number of repeats",
        y = "Fraction of reads"
    )

# same plot but with repeat nubmer as factor
p1a <- errors %>% 
    dplyr::mutate(read_frac = reads / sum(reads)) %>%
    dplyr::mutate(repeats = forcats::fct_reorder(factor(repeats), as.numeric(repeats))) %>%
    ggplot2::ggplot(ggplot2::aes(x = repeats, y = read_frac)) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::labs(
        x = "Number of repeats",
        y = "Fraction of reads"
    )

# group all repeats more than 10 and re-caculate rates
errors <- errors %>% 
    dplyr::mutate(group = ifelse(repeats >= 10, "10+", as.character(repeats))) %>%
    dplyr::group_by(group) %>%
    dplyr::summarise(
        reads = sum(reads),
        ins = sum(ins),
        del = sum(del),
        sub = sum(sub),
        total = sum(total_vars)
    ) %>% 
    dplyr::mutate(
        ins_per_read = ins / reads,
        del_per_read = del / reads,
        sub_per_read = sub / reads,
        errors_per_read = total / reads,
        ins_per_base = ins / (reads * aav2_length),
        del_per_base = del / (reads * aav2_length),
        sub_per_base = sub / (reads * aav2_length),
        errors_per_base = total / (reads * aav2_length),
        accuracy_per_base = 1 - errors_per_base
    ) %>%
    dplyr::mutate(group = factor(group, levels = c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10+")))


# plot fraction of reads in each group
p1b <- errors %>% 
    ggplot2::ggplot(ggplot2::aes(x = group, y = reads / sum(reads), group=1)) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::labs(
        x = "Number of repeats",
        y = "Fraction of reads"
    )


# plot per-base deletions, insertions, and substitutions
p2 <- errors %>% 
    dplyr::select(group, del_per_base, ins_per_base, sub_per_base) %>%
    tidyr::pivot_longer(cols = c(del_per_base, ins_per_base, sub_per_base), names_to = "error_type", values_to = "rate") %>%
    dplyr::mutate(error_type = dplyr::case_match(error_type,
        "del_per_base" ~ "Deletions",
        "ins_per_base" ~ "Insertions",
        "sub_per_base" ~ "Substitutions"
    )) %>%
    ggplot2::ggplot(ggplot2::aes(x = group, y = rate, color = error_type, group=error_type)) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::labs(
        x = "Number of repeats",
        y = "Per-base error rate"
    ) +
    # change legend to "Error type"
    ggplot2::scale_color_discrete(name = "Error type")

# plot per-base accuracy
p3 <- errors %>% 
    dplyr::select(group, accuracy_per_base) %>%
    ggplot2::ggplot(ggplot2::aes(x = group, y = accuracy_per_base, group=1)) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::labs(
        x = "Number of repeats",
        y = "Per-base accuracy"
    )


p <- p1 + p3 + p2

ggplot2::ggsave(file.path(out_dir, "average_error_rates.png"), dpi=300)
ggplot2::ggsave(file.path(out_dir, "average_error_rates.pdf"))

#### median error rates ####

# read variant data
files <- list.files(here::here("out/aav2_benchmarking/consensus_var"), full.names = TRUE, pattern="*vars.tsv")
vars <- readr::read_tsv(files, id = "filename") %>%
    # get variant types
    dplyr::mutate(type = dplyr::case_when(
        stringr::str_detect(var, "ins") ~ "ins",
        stringr::str_detect(var, "del") ~ "del",
        stringr::str_detect(var, "[A|C|G|T]\\d+[A|C|G|T]") ~ "sub",
        TRUE ~ NA_character_
    )) %>%
    # remove insertions at last position
    dplyr::filter(!(type == "ins" & pos == as.character(aav2_length))) 

# count nmber of ins, del and sub per read
vars <- vars %>% 
    dplyr::group_by(query_name) %>%
    dplyr::summarise(
        ins = sum(type == "ins"),
        del = sum(type == "del"),
        sub = sum(type == "sub"),
        total = dplyr::n(),
        ins_per_base = sum(type == "ins") / aav2_length,
        del_per_base = sum(type == "del") / aav2_length,
        sub_per_base = sum(type == "sub") / aav2_length,
        accuracy_per_base = 1 - (dplyr::n() / aav2_length)
    )

# read in list of read IDs so we can fill in any reads without any variants
files <- list.files(here::here("out/aav2_benchmarking/consensus_var"), full.names = TRUE, pattern="*reads.tsv")
rids <- readr::read_tsv(files, col_names = "query_name") 

# merge with read IDs
vars <- dplyr::left_join(rids, vars, by = "query_name") %>%
    tidyr::replace_na(list(ins = 0, del = 0, sub = 0, total = 0, ins_per_base = 0, del_per_base = 0, sub_per_base = 0, accuracy_per_base = 1)) %>%
    # get number of repeats for each read
    dplyr::mutate(repeats = stringr::str_split(query_name, "_", simplify=TRUE)[,4]) %>%
    dplyr::mutate(group  = ifelse(as.numeric(repeats) >= 10, "10+", repeats)) %>%
    dplyr::mutate(group = factor(group, levels = c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10+")))


# plot per-base deletions, insertions, and substitutions
p4 <- vars %>% 
    dplyr::select(repeats, del_per_base, ins_per_base, sub_per_base) %>%
    tidyr::pivot_longer(cols = c(del_per_base, ins_per_base, sub_per_base), names_to = "error_type", values_to = "rate") %>%
dplyr::mutate(repeats = forcats::fct_reorder(repeats, as.numeric(repeats))) %>%
    dplyr::mutate(error_type = dplyr::case_match(error_type,
        "del_per_base" ~ "Deletions",
        "ins_per_base" ~ "Insertions",
        "sub_per_base" ~ "Substitutions"
    )) %>%
    ggplot2::ggplot(ggplot2::aes(x = repeats, y = rate, color = error_type)) +
    ggplot2::geom_boxplot() +
    ggplot2::labs(
        x = "Number of repeats",
        y = "Per-base error rate"
    ) +
    # change legend to "Error type"
    ggplot2::scale_color_discrete(name = "Error type") +
    ggplot2::theme(legend.position = "bottom") 

# plot per-base accuracy
p4a <- vars %>% 
    dplyr::mutate(repeats = forcats::fct_reorder(repeats, as.numeric(repeats))) %>%
    ggplot2::ggplot(ggplot2::aes(x = repeats, y = accuracy_per_base)) +
    ggplot2::geom_boxplot() +
    ggplot2::labs(
        x = "Number of repeats",
        y = "Median per-base accuracy"
    )


p <- p1a / p4a / p4
ggplot2::ggsave(file.path(out_dir, "error-rates.png"), dpi=300, height = 7, width=10)
ggplot2::ggsave(file.path(out_dir, "error-rates.pdf"), height=7, width = 10)

# plot fraction of reads with no errors
p5 <- vars %>% 
    dplyr::mutate(no_errors = total == 0) %>%
    dplyr::group_by(group) %>%
    dplyr::summarise(
        no_errors = sum(no_errors),
        reads = dplyr::n()
    ) %>%
    dplyr::mutate(frac_no_errors = no_errors / reads) %>%
    ggplot2::ggplot(ggplot2::aes(x = group, y = frac_no_errors, group=1)) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::labs(
        x = "Number of repeats",
        y = "Fraction of reads with no errors"
    )

# plot median per-base accuracy
p6 <- vars %>% 
    dplyr::group_by(group) %>%
    dplyr::summarise(
        accuracy_per_base = median(accuracy_per_base)
    ) %>%
    ggplot2::ggplot(ggplot2::aes(x = group, y = accuracy_per_base, group=1)) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::labs(
        x = "Number of repeats",
        y = "Median per-base accuracy"
    )

# plot median per-base deletions, insertions, and substitutions
p7 <- vars %>% 
    dplyr::select(group, del_per_base, ins_per_base, sub_per_base) %>%
    tidyr::pivot_longer(cols = c(del_per_base, ins_per_base, sub_per_base), names_to = "error_type", values_to = "rate") %>%
    dplyr::mutate(error_type = dplyr::case_match(error_type,
        "del_per_base" ~ "Deletions",
        "ins_per_base" ~ "Insertions",
        "sub_per_base" ~ "Substitutions"
    )) %>%
    dplyr::group_by(group, error_type) %>%
    dplyr::summarise(
        rate = median(rate)
    ) %>%
    ggplot2::ggplot(ggplot2::aes(x = group, y = rate, color = error_type, group=error_type)) +
    ggplot2::geom_point() +
    ggplot2::geom_line() +
    ggplot2::labs(
        x = "Number of repeats",
        y = "Median per-base error rate"
    ) +
    # change legend to "Error type"
    ggplot2::scale_color_discrete(name = "Error type")


p <- (p1 + p5) / (p6 + p7)

ggplot2::ggsave(file.path(out_dir, "median_error_rates.png"), dpi=300)
ggplot2::ggsave(file.path(out_dir, "median_error_rates.pdf"))


# load image from igv
igv <- cowplot::ggdraw() +
    cowplot::draw_image(here::here("out/aav2_benchmarking/num_repeats_igv/np-aav2-cc_num_repeats.png"), width = 1, height = 1)



layout <- "
AAA
AAA
BCD
"


p <- igv + p1b + p6 + p7 +
    plot_annotation(tag_levels = "A") +
    plot_layout(design = layout)

ggplot2::ggsave(file.path(out_dir, "median_error_rates_with_igv.png"), dpi=300, units="mm", width=86.5, height=80, scale=1.9)
ggplot2::ggsave(file.path(out_dir, "median_error_rates_with_igv.pdf"), units="mm", width=86.5, height=80, scale=1.9)

# write error rates to file
errors %>%
   dplyr:: arrange(group) %>%
   readr::write_tsv(file.path(out_dir, "mean_error_rates.tsv"))

vars %>%
    readr::write_tsv(file.path(out_dir, "per_read_error_rates.tsv"))

vars %>% 
    dplyr::group_by(group) %>%
    dplyr::summarise(
        reads = dplyr::n(),
        median_ins = median(ins),
        median_del = median(del),
        median_sub = median(sub),
        median_total_errors = median(total),
        median_ins_per_base = median(ins_per_base),
        median_del_per_base = median(del_per_base),
        median_sub_per_base = median(sub_per_base),
        median_accuracy_per_base = median(accuracy_per_base)
    ) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(read_fraction = reads / sum(reads)) %>%
    readr::write_tsv(file.path(out_dir, "median_error_rates.tsv"))
