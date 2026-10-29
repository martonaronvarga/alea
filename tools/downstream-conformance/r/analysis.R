# Named constrained draw tables using the same strict, standard-library reader.
# Optional analysis dependencies: R jsonlite and Python 3; native Alea needs neither.
read_alea <- function(manifest, csv, reader,
                      python = Sys.getenv("ALEA_PYTHON", "python3")) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("install jsonlite")
  bridge <- tempfile(fileext = ".json")
  on.exit(unlink(bridge), add = TRUE)
  args <- c(normalizePath(reader), normalizePath(manifest), normalizePath(csv),
            "--r-json", bridge)
  output <- system2(python, shQuote(args), stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) stop(paste(output, collapse = "\n"))
  data <- jsonlite::fromJSON(bridge, simplifyVector = FALSE, bigint_as_char = TRUE)
  if (!identical(data$bridge_format, "alea-analysis-r") || data$bridge_version != 1L) {
    stop("unsupported R analysis bridge")
  }
  rows <- data$records
  text <- function(key) vapply(rows, function(r) {
    if (is.null(r[[key]])) NA_character_ else r[[key]]
  }, character(1))
  coordinates <- data.frame(chain = text("chain"), phase = text("phase"),
                            draw = text("draw"), error = text("error"),
                            stringsAsFactors = FALSE, check.names = FALSE)
  columns <- unlist(data$manifest$csv$columns, use.names = FALSE)[-(1:5)]
  table <- function(selected, labels) {
    result <- lapply(selected, function(name) {
      values <- lapply(rows, function(r) r$values[[name]])
      if (name %in% c("stat:integration_steps", "stat:tree_depth")) {
        vapply(values, function(v) if (is.null(v)) NA_character_ else v, character(1))
      } else if (name %in% c("stat:accepted", "stat:divergent", "stat:depth_saturated")) {
        vapply(values, function(v) if (is.null(v)) NA else v, logical(1))
      } else {
        vapply(values, function(v) if (is.null(v)) NA_real_ else as.numeric(v), numeric(1))
      }
    })
    names(result) <- labels
    if (!length(result)) return(data.frame(row.names = seq_along(rows)))
    as.data.frame(result, check.names = FALSE, stringsAsFactors = FALSE)
  }
  group <- function(prefix, strip = TRUE) {
    selected <- columns[startsWith(columns, prefix)]
    labels <- if (strip) substring(selected, nchar(prefix) + 1L) else selected
    table(selected, labels)
  }
  available <- lapply(columns, function(name) vapply(rows, function(r) {
    !is.null(r$values[[name]])
  }, logical(1)))
  names(available) <- columns
  structure(list(manifest = data$manifest, coordinates = coordinates,
                 posterior = group("parameter:"), unconstrained = group("unconstrained:"),
                 statistics = group("stat:", FALSE), log_likelihood = group("log_likelihood:"),
                 generated = group("prediction:"),
                 available = as.data.frame(available, check.names = FALSE)),
            class = "alea_analysis")
}

alea_rectangular <- function(x, chains = NULL, phase = "sampling") {
  if (!phase %in% c("warmup", "sampling")) stop("invalid phase")
  plan <- x$manifest$chains
  ids <- vapply(plan, function(c) c$id, character(1))
  if (is.null(chains)) chains <- ids
  if (!is.character(chains) || !length(chains) || anyDuplicated(chains) ||
      !all(chains %in% ids)) stop("select exact chain IDs as character strings")
  indices <- lapply(chains, function(id) {
    p <- plan[[match(id, ids)]]
    if (p$status != "completed") stop("selected chain is incomplete")
    selected <- which(x$coordinates$chain == id & x$coordinates$phase == phase)
    if (!length(selected) || any(!is.na(x$coordinates$error[selected]))) {
      stop("selected grid is empty or contains failures")
    }
    selected
  })
  draws <- lapply(indices, function(i) x$coordinates$draw[i])
  if (!all(vapply(draws, identical, logical(1), draws[[1]]))) {
    stop("unequal draw coordinates")
  }
  list(chain = chains, draw = draws[[1]], rows = indices)
}
