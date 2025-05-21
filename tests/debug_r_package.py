import os
import rpy2.robjects as ro
from rpy2.robjects.packages import importr, PackageNotInstalledError

# Initialize R
r = ro.r

try:
    # Try to import the synthdid package
    synthdid = importr("synthdid")
    print("Successfully imported synthdid package")
    
    # List all functions in the package using R directly
    print("\nAvailable functions in synthdid package:")
    r('print(ls("package:synthdid"))')
    
    # Check if the package has the dataset
    print("\nChecking for california_prop99 dataset:")
    try:
        # Try to load the dataset manually
        r('data(package="synthdid")')
        print("\nAvailable datasets in synthdid package:")
        r('print(data(package="synthdid")$results[, "Item"])')
        
        # Try to load the california_prop99 dataset
        r('data(california_prop99, package="synthdid")')
        r('print(head(california_prop99))')
        print("\nDataset dimensions:")
        r('print(dim(california_prop99))')
    except Exception as e:
        print(f"Error loading california_prop99 dataset: {e}")
    
    # Try the random_low_rank function
    print("\nTesting random_low_rank function:")
    r('set.seed(123)')
    r('setup <- random_low_rank()')
    r('print(names(setup))')
    r('print(dim(setup$Y))')
    
except PackageNotInstalledError:
    print("synthdid package is not installed") 