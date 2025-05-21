# tests/run_fw_test_r.R

# Load necessary libraries
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "http://cran.us.r-project.org", quiet = TRUE)
}
library(jsonlite)

# Source the R solver if not in a package structure (adjust path if necessary)
# Assuming the R solver functions are in a file accessible from here.
# For synthdid package, it's usually loaded via library(synthdid)
if (!requireNamespace("synthdid", quietly = TRUE)) {
    # This might be needed if running outside a context where synthdid is installed/loaded
    # stop("synthdid package not found. Please install it.")
    # If sourcing directly from local files (dev context):
    # source("../R-package/synthdid/R/solver.R") 
} else {
    library(synthdid) # This should make sc.weight.fw available
}

# Get command line arguments for output path
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: Rscript run_fw_test_r.R <output_json_path>", call. = FALSE)
}
output_json_path <- args[1]

# Define a fixed small Y matrix (N0x(T0+1)) and zeta for testing sc.weight.fw
# Let N0 = 3, T0 = 4. So Y_fw_test is 3x5
Y_fw_test_data <- c(
  10, 12, 11, 13, 15,  # Unit 1
  20, 22, 21, 23, 25,  # Unit 2
  5,  7,  6,  8,  10   # Unit 3 (Target for b if T0+1 is target)
)
Y_fw_test <- matrix(Y_fw_test_data, nrow = 3, ncol = 5, byrow = TRUE)

# For sc.weight.fw, Y is (N0 rows of controls) x (T0 cols of pre-data for controls + 1 col of target for controls)
# In our case, let's say all 3 units are controls, and we are trying to predict the 5th period value (col 5)
# using the first 4 periods (cols 1-4).
# So Y for sc.weight.fw will be Y_fw_test.
# N0 = 3 (all units are controls relative to each other for this test)
# T0 = 4 (first 4 periods are predictors)

zeta_fw_test <- 0.1 
min_decrease_fw_test <- 1e-8 # Smaller decrease for more iterations if needed
max_iter_fw_test <- 500      # Fixed iterations to compare vals array

results <- list()

# Test sc.weight.fw
# intercept = FALSE for simpler direct comparison of core FW logic
# Y should be N0 x (T0+1) where T0 is number of actual pre-periods for lambda
# The last column of Y is the target `b`
# The first T0 columns of Y is the matrix `A`

tryCatch({
    # Ensure sc.weight.fw is available (it's not exported by default if just sourcing solver.R)
    # If synthdid is loaded, it should be synthdid:::sc.weight.fw or just sc.weight.fw if in same env.
    # If sourcing solver.R directly, it's just sc.weight.fw
    
    fw_func <- NULL
    if (exists("sc.weight.fw", mode = "function")) {
        fw_func <- get("sc.weight.fw")
    } else if (requireNamespace("synthdid", quietly = TRUE) && exists("sc.weight.fw", where = asNamespace("synthdid"), mode="function")) {
        # Use synthdid:::sc.weight.fw to access the non-exported function if synthdid is loaded
        fw_func <- get("sc.weight.fw", asNamespace("synthdid"))
    } else {
        # Attempt to source directly if not found via package (for dev environment)
        solver_path_guess1 <- "../R-package/synthdid/R/solver.R" # Relative to tests/ directory
        solver_path_guess2 <- "R-package/synthdid/R/solver.R"   # Relative to project root
        if(file.exists(solver_path_guess1)){
            source(solver_path_guess1)
        } else if(file.exists(solver_path_guess2)){
            source(solver_path_guess2)
        } else {
             stop("Could not find solver.R to source sc.weight.fw")
        }
        if (exists("sc.weight.fw", mode = "function")) {
             fw_func <- get("sc.weight.fw")
        } else {
            stop("Function sc.weight.fw not found after attempting to source solver.R.")
        }
    }

    # lambda_init default is uniform 1/T0. We will use this default.
    r_fw_results <- fw_func(
        Y = Y_fw_test, 
        zeta = zeta_fw_test, 
        intercept = FALSE, 
        lambda = NULL, # Use default init for lambda (uniform 1/T0)
        min.decrease = min_decrease_fw_test, 
        max.iter = max_iter_fw_test
    )
    results$r_lambda <- as.numeric(r_fw_results$lambda)
    results$r_vals <- as.numeric(r_fw_results$vals[!is.na(r_fw_results$vals)]) # Remove trailing NAs from vals

}, error = function(e) {
  results$error <- paste("Error in R sc.weight.fw execution:", e$message)
})

# Write results to JSON
tryCatch({
  json_output <- toJSON(results, auto_unbox = TRUE, pretty = TRUE)
  write(json_output, file = output_json_path)
}, error = function(e) {
  # Fallback if results couldn't be formed, write the error itself
  error_output <- list(error_writing_json = paste("Error writing JSON:", e$message, "Original R error was:", results$error))
  json_error_output <- toJSON(error_output, auto_unbox = TRUE, pretty = TRUE)
  write(json_error_output, file = output_json_path) 
})

# cat("R Frank-Wolfe test script finished.\n") 