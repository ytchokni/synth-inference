#!/usr/bin/env Rscript
# Inspect the structure of the california_prop99 dataset

library(synthdid)

# Load the data
data('california_prop99')

# Print basic structure
cat("Structure of california_prop99:\n")
str(california_prop99)

# Print head
cat("\nHead of california_prop99:\n")
print(head(california_prop99))

# Print class
cat("\nClass of california_prop99:\n")
print(class(california_prop99))

# Check treatment column
cat("\nTable of 'treated' column:\n")
print(table(california_prop99$treated))

# Check State column type and values
cat("\nUnique values in State column:\n")
print(unique(california_prop99$State))
cat("Class of State column:", class(california_prop99$State), "\n")

# Specifically looking at California
cat("\nCalifornia data:\n")
california_data <- california_prop99[california_prop99$State == "California", ]
print(california_data)

# Get the Panel Matrices
cat("\nPanel matrices setup:\n")
setup <- panel.matrices(california_prop99)
print(names(setup))
cat("N0:", setup$N0, "\n")
cat("T0:", setup$T0, "\n")
cat("Y dimensions:", dim(setup$Y), "\n") 