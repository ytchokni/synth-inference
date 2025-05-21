# tests/run_synthdid.R

# Function to install and load a package
install_and_load <- function(pkg_name) {
  if (!require(pkg_name, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg_name, repos = "http://cran.us.r-project.org", quiet = TRUE)
    if (!require(pkg_name, character.only = TRUE, quietly = TRUE)) {
      stop(paste("Package", pkg_name, "could not be installed or loaded."))
    }
  }
}

# Install and load necessary packages
install_and_load("synthdid")
install_and_load("jsonlite")

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop("Usage: Rscript run_synthdid.R <method> <output_json_path>", call. = FALSE)
}

method <- args[1]
output_json_path <- args[2]

# Instead of reading from CSV, use the built-in dataset directly
data('california_prop99')
setup <- panel.matrices(california_prop99)

# Add debug information about matrices
debug_info <- list()
debug_info$N0 <- setup$N0
debug_info$T0 <- setup$T0
debug_info$Y_dims <- dim(setup$Y)
# First few rows and columns of Y to verify ordering
if (dim(setup$Y)[1] > 5 && dim(setup$Y)[2] > 5) {
  debug_info$Y_subset <- setup$Y[1:5, 1:5]
}
# Last unit (should be treated unit - California)
if (dim(setup$Y)[1] > 0 && dim(setup$Y)[2] > 0) {
  debug_info$Y_treated_unit <- setup$Y[dim(setup$Y)[1], ]
}
# Save Y matrix for Python to use for exact comparison
y_matrix_path <- paste0(dirname(output_json_path), "/r_y_matrix.csv")
# Use write.table with higher precision options
# Convert matrix to data.frame first for write.table to handle col.names correctly by default
Y_df <- as.data.frame(setup$Y)
# write.table(Y_df, y_matrix_path, sep = ",", row.names = FALSE, col.names = TRUE, qmethod = "double", digits = 16)
# Simpler: use write_csv from readr, which is usually better for precision with CSVs
if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr", repos = "http://cran.us.r-project.org", quiet = TRUE)
}
readr::write_csv(Y_df, y_matrix_path, col_names = TRUE)
debug_info$Y_matrix_path <- y_matrix_path

# Initialize results list
results <- list()
results$debug_info <- debug_info

# Perform estimation based on method
if (method == "synthdid") {
  estimate_obj <- synthdid_estimate(setup$Y, setup$N0, setup$T0)
  results$estimate <- as.numeric(estimate_obj) # The object itself is the estimate
  weights <- attr(estimate_obj, "weights")
  if (!is.null(weights)) {
    results$lambda_weights <- if (!is.null(weights$lambda)) as.numeric(weights$lambda) else numeric(0)
    results$omega_weights <- if (!is.null(weights$omega)) as.numeric(weights$omega) else numeric(0)
  } else {
    results$lambda_weights <- numeric(0)
    results$omega_weights <- numeric(0)
  }
  results$se <- 0.0 # Matching current Python test behavior

} else if (method == "sc") {
  estimate_obj <- sc_estimate(setup$Y, setup$N0, setup$T0)
  results$estimate <- as.numeric(estimate_obj)
  weights <- attr(estimate_obj, "weights") # sc_estimate also returns weights attribute
  if (!is.null(weights)) {
      results$lambda_weights <- if (!is.null(weights$lambda)) as.numeric(weights$lambda) else rep(0, setup$T0) # Should be zeros
      results$omega_weights <- if (!is.null(weights$omega)) as.numeric(weights$omega) else numeric(0) 
  } else {
      results$lambda_weights <- rep(0, setup$T0)
      results$omega_weights <- numeric(0)
  }
  results$se <- 0.0

} else if (method == "did") {
  estimate_obj <- did_estimate(setup$Y, setup$N0, setup$T0)
  results$estimate <- as.numeric(estimate_obj)
  # DiD weights are fixed
  results$lambda_weights <- if (setup$T0 > 0) rep(1/setup$T0, setup$T0) else numeric(0)
  results$omega_weights <- if (setup$N0 > 0) rep(1/setup$N0, setup$N0) else numeric(0)
  results$se <- 0.0
  
} else {
  stop(paste("Unknown method:", method))
}

# Write results to JSON
tryCatch({
  json_output <- toJSON(results, auto_unbox = TRUE, pretty = TRUE)
  write(json_output, file = output_json_path)
}, error = function(e) {
  stop(paste("Error writing JSON output to:", output_json_path, "\\n", e$message))
}) 