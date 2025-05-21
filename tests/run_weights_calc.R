#!/usr/bin/env Rscript

# Script to calculate weights and intermediate steps for R synthdid algorithm
# Usage: Rscript run_weights_calc.R output_json_path

library(jsonlite)
library(synthdid)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript run_weights_calc.R <output_json_path>")
}
output_file <- args[1]

# Load the California Prop 99 dataset
data('california_prop99')

# Use panel.matrices to create matrices with correct treatment/control ordering
setup = panel.matrices(california_prop99)
Y = setup$Y
N0 = setup$N0
T0 = setup$T0

# Calculate noise level
noise_level = sd(apply(Y[1:N0,1:T0], 1, diff))

# Run the synthdid estimate
tau_hat = synthdid_estimate(Y, N0, T0)

# Get lambda and omega weights
lambda_weights = attr(tau_hat, "weights")$lambda
omega_weights = attr(tau_hat, "weights")$omega

# Run SC estimate
tau_sc = sc_estimate(Y, N0, T0)
sc_omega_weights = attr(tau_sc, "weights")$omega

# Collect results
results = list(
  # Basic setup
  N0 = N0,
  T0 = T0,
  Y_dims = dim(Y),
  Y_subset = as.matrix(Y[1:min(5, nrow(Y)), 1:min(5, ncol(Y))]),
  Y_treated_unit = as.numeric(Y[nrow(Y), ]),
  
  # Noise level
  noise_level = noise_level,
  
  # Main SynthDID estimates and weights
  synthdid_estimate = as.numeric(tau_hat),
  synthdid_estimate_native = as.numeric(tau_hat),
  lambda_weights = as.numeric(lambda_weights),
  omega_weights = as.numeric(omega_weights),
  
  # SC estimate and weights
  sc_estimate = as.numeric(tau_sc),
  sc_omega_weights = as.numeric(sc_omega_weights),
  
  # Add additional components as needed
  debug_info = list(
    lambda_iteration_vals = attr(tau_hat, "weights")$lambda.vals,
    omega_iteration_vals = attr(tau_hat, "weights")$omega.vals
  )
)

# Write results to JSON file
write_json(results, output_file, pretty=TRUE, auto_unbox=TRUE)

# Print confirmation
cat("Results written to", output_file, "\n")

# To export the actual Y matrix for direct comparison, uncomment this section
# write.csv(Y, file=tempfile(fileext=".csv"), row.names=FALSE)
# results$Y_matrix_path = tempfile(fileext=".csv")
# write.csv(Y, file=results$Y_matrix_path, row.names=FALSE) 