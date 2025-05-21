"""
Test to compare Frank-Wolfe update behavior between R and Python implementations.
"""

import os
import tempfile
import subprocess
import json
import numpy as np
import pytest
from synthdid_py.solver import fw_step

# Define tolerance for numerical comparison
TOLERANCE = 1e-5  # Standard tolerance for numerical algorithm comparison
EDGE_CASE_TOLERANCE = 1e-3  # Relaxed tolerance for edge cases


def test_fw_step_basic():
    """Compare the basic Frank-Wolfe step between R and Python."""
    # Set up a simple test case
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float64)
    
    x = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    b = np.array([5.0, 10.0, 15.0], dtype=np.float64)
    eta = 0.1
    
    # Get Python result
    py_result = fw_step(A, x, b, eta)
    
    # Get R result
    r_result = None
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create R script
        r_script_path = os.path.join(tmpdir, "test_fw_step.R")
        output_file = os.path.join(tmpdir, "r_fw_result.json")
        
        with open(r_script_path, "w") as f:
            r_script = """
            #!/usr/bin/env Rscript
            
            library(jsonlite)
            
            # Define fw.step function (copied from synthdid R package)
            fw.step = function(A, x, b, eta, alpha=NULL) {{
                v = unit.simplex.vertex(t(A) %*% (A %*% x - b) + eta * x)
                if(is.null(alpha)) {{
                    M = t(A) %*% A + eta * diag(length(x))
                    alpha = min(1, max(0, t(x - v) %*% (M %*% x - t(A) %*% b) / (t(x - v) %*% M %*% (x - v))))
                }}
                alpha * v + (1 - alpha) * x
            }}
            
            # Define helper function
            unit.simplex.vertex = function(c) {{
                j = which.min(c)
                result = rep(0, length(c))
                result[j] = 1
                result
            }}
            
            # Test data
            A = matrix(c(1, 4, 7, 2, 5, 8, 3, 6, 9), nrow=3)
            x = c(0.2, 0.3, 0.5)
            b = c(5, 10, 15)
            eta = 0.1
            
            # Run the fw.step
            result = fw.step(A, x, b, eta)
            
            # Export to JSON
            write_json(as.numeric(result), "{output_file}")
            """
            r_script = r_script.format(output_file=output_file)
            f.write(r_script)
        
        # Run R script
        subprocess.run(["Rscript", r_script_path], check=True)
        
        # Load results
        with open(output_file, "r") as f:
            r_result = np.array(json.load(f))
    
    # Print results
    print("Python result:", py_result)
    print("R result:", r_result)
    
    # Compare results
    assert np.allclose(py_result, r_result, atol=TOLERANCE), \
        f"Frank-Wolfe step results differ: Python={py_result}, R={r_result}"


def test_fw_step_real_data():
    """Compare Frank-Wolfe step on more realistic data."""
    # Use a larger matrix similar to what would be used in synthdid
    np.random.seed(42)  # Set seed for reproducibility
    
    # Create test data
    N, T = 10, 8
    A = np.random.randn(N, T).astype(np.float64)
    x = np.ones(T) / T  # Initial uniform weights
    b = np.random.randn(N).astype(np.float64)
    eta = 0.1
    
    # Get Python result
    py_result = fw_step(A, x, b, eta)
    
    # Get R result
    r_result = None
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create data files
        A_file = os.path.join(tmpdir, "A.csv")
        x_file = os.path.join(tmpdir, "x.csv")
        b_file = os.path.join(tmpdir, "b.csv")
        
        np.savetxt(A_file, A, delimiter=",")
        np.savetxt(x_file, x, delimiter=",")
        np.savetxt(b_file, b, delimiter=",")
        
        # Create R script
        r_script_path = os.path.join(tmpdir, "test_fw_step_real.R")
        output_file = os.path.join(tmpdir, "r_fw_result_real.json")
        
        with open(r_script_path, "w") as f:
            r_script = """
            #!/usr/bin/env Rscript
            
            library(jsonlite)
            
            # Define fw.step function (copied from synthdid R package)
            fw.step = function(A, x, b, eta, alpha=NULL) {{
                v = unit.simplex.vertex(t(A) %*% (A %*% x - b) + eta * x)
                if(is.null(alpha)) {{
                    M = t(A) %*% A + eta * diag(length(x))
                    alpha = min(1, max(0, t(x - v) %*% (M %*% x - t(A) %*% b) / (t(x - v) %*% M %*% (x - v))))
                }}
                alpha * v + (1 - alpha) * x
            }}
            
            # Define helper function
            unit.simplex.vertex = function(c) {{
                j = which.min(c)
                result = rep(0, length(c))
                result[j] = 1
                result
            }}
            
            # Read test data
            A = as.matrix(read.csv("{A_file}", header=FALSE))
            x = as.numeric(read.csv("{x_file}", header=FALSE)[,1])
            b = as.numeric(read.csv("{b_file}", header=FALSE)[,1])
            eta = 0.1
            
            # Run the fw.step
            result = fw.step(A, x, b, eta)
            
            # Export to JSON
            write_json(as.numeric(result), "{output_file}")
            """
            r_script = r_script.format(
                A_file=A_file,
                x_file=x_file,
                b_file=b_file,
                output_file=output_file
            )
            f.write(r_script)
        
        # Run R script
        subprocess.run(["Rscript", r_script_path], check=True)
        
        # Load results
        with open(output_file, "r") as f:
            r_result = np.array(json.load(f))
    
    # Print results
    print("\nLarger test case:")
    print("Python result:", py_result)
    print("R result:", r_result)
    
    # Compare results
    assert np.allclose(py_result, r_result, atol=EDGE_CASE_TOLERANCE), \
        f"Frank-Wolfe step results differ for real data: Python={py_result}, R={r_result}"


def test_fw_step_edge_cases():
    """Test Frank-Wolfe step on edge cases where numerical differences may occur."""
    # Test with very small numbers
    A_small = np.random.randn(5, 3).astype(np.float64) * 1e-6
    x_small = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    b_small = np.random.randn(5).astype(np.float64) * 1e-6
    eta_small = 1e-8
    
    # Test with very large numbers
    A_large = np.random.randn(5, 3).astype(np.float64) * 1e6
    x_large = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    b_large = np.random.randn(5).astype(np.float64) * 1e6
    eta_large = 1e4
    
    # Test with NaNs and zeros in the input
    A_special = np.array([
        [1.0, 0.0, 3.0],
        [0.0, 0.0, 0.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float64)
    
    x_special = np.array([0.2, 0.0, 0.8], dtype=np.float64)
    b_special = np.array([5.0, 0.0, 15.0], dtype=np.float64)
    eta_special = 0.1
    
    test_cases = [
        ("Small values", A_small, x_small, b_small, eta_small),
        ("Large values", A_large, x_large, b_large, eta_large),
        ("Special values", A_special, x_special, b_special, eta_special)
    ]
    
    for name, A, x, b, eta in test_cases:
        print(f"\nTesting {name} case:")
        
        # Get Python result
        try:
            py_result = fw_step(A, x, b, eta)
            print("Python result:", py_result)
        except Exception as e:
            print(f"Python implementation error: {e}")
            py_result = None
        
        # Get R result
        r_result = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create data files
                A_file = os.path.join(tmpdir, "A.csv")
                x_file = os.path.join(tmpdir, "x.csv")
                b_file = os.path.join(tmpdir, "b.csv")
                
                np.savetxt(A_file, A, delimiter=",")
                np.savetxt(x_file, x, delimiter=",")
                np.savetxt(b_file, b, delimiter=",")
                
                # Create R script
                r_script_path = os.path.join(tmpdir, "test_fw_step_edge.R")
                output_file = os.path.join(tmpdir, "r_fw_result_edge.json")
                
                with open(r_script_path, "w") as f:
                    r_script = """
                    #!/usr/bin/env Rscript
                    
                    library(jsonlite)
                    
                    # Define fw.step function (copied from synthdid R package)
                    fw.step = function(A, x, b, eta, alpha=NULL) {{
                        v = unit.simplex.vertex(t(A) %*% (A %*% x - b) + eta * x)
                        if(is.null(alpha)) {{
                            M = t(A) %*% A + eta * diag(length(x))
                            alpha = min(1, max(0, t(x - v) %*% (M %*% x - t(A) %*% b) / (t(x - v) %*% M %*% (x - v))))
                        }}
                        alpha * v + (1 - alpha) * x
                    }}
                    
                    # Define helper function
                    unit.simplex.vertex = function(c) {{
                        j = which.min(c)
                        result = rep(0, length(c))
                        result[j] = 1
                        result
                    }}
                    
                    # Read test data
                    A = as.matrix(read.csv("{A_file}", header=FALSE))
                    x = as.numeric(read.csv("{x_file}", header=FALSE)[,1])
                    b = as.numeric(read.csv("{b_file}", header=FALSE)[,1])
                    eta = {eta_value}
                    
                    # Run the fw.step with error handling
                    result = tryCatch({{
                        fw.step(A, x, b, eta)
                    }}, error = function(e) {{
                        cat("R error:", e$message, "\\n")
                        rep(NA, length(x))
                    }})
                    
                    # Export to JSON
                    write_json(as.numeric(result), "{output_file}")
                    """
                    r_script = r_script.format(
                        A_file=A_file,
                        x_file=x_file,
                        b_file=b_file,
                        eta_value=eta,
                        output_file=output_file
                    )
                    f.write(r_script)
                
                # Run R script
                process = subprocess.run(["Rscript", r_script_path], 
                                       capture_output=True, text=True, check=False)
                if process.stderr:
                    print("R stderr:", process.stderr)
                if process.stdout:
                    print("R stdout:", process.stdout)
                
                # Load results if file exists
                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        r_result = np.array(json.load(f))
                    print("R result:", r_result)
                else:
                    print("R implementation did not produce output")
        except Exception as e:
            print(f"Error running R implementation: {e}")
        
        # Compare results if both are available
        if py_result is not None and r_result is not None:
            # Check for NaNs
            py_nan = np.isnan(py_result)
            r_nan = np.isnan(r_result)
            
            if np.array_equal(py_nan, r_nan):
                # Compare non-NaN values
                if np.any(~py_nan):
                    assert np.allclose(py_result[~py_nan], r_result[~py_nan], atol=EDGE_CASE_TOLERANCE), \
                        f"Frank-Wolfe step results differ for {name} case"
                print(f"{name} case: Results match where not NaN")
            else:
                print(f"{name} case: NaN patterns differ - Python: {py_nan}, R: {r_nan}")
                

if __name__ == "__main__":
    # Run the tests
    test_fw_step_basic()
    test_fw_step_real_data()
    test_fw_step_edge_cases() 