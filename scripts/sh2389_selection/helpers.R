
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
