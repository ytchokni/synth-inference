"""
Test to compare how sparsify is applied in the synthdid estimation process.
"""

import os
import tempfile
import subprocess
import json
import numpy as np
import pandas as pd
import pytest
from synthdid_py import synthdid_estimate
from synthdid_py.solver import sparsify_function

# Define tolerance for numerical comparison
TOLERANCE = 1e-3  # Relaxed tolerance for comparing weights after sparsification


def get_r_weights_with_sparsify_analysis():
    """
    Run a specialized R script that returns weights before and after sparsification.
    """
    result = None
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json_path = os.path.join(tmpdir, "r_sparsify_analysis.json")
        
        # Create R script
        r_script_path = os.path.join(tmpdir, "analyze_sparsify.R")
        with open(r_script_path, "w") as f:
            r_script = """
            #!/usr/bin/env Rscript
            
            library(jsonlite)
            library(synthdid)
            
            # Load the California Prop 99 dataset
            data('california_prop99')
            
            # Set up matrices
            setup = panel.matrices(california_prop99)
            Y = setup$Y
            N0 = setup$N0
            T0 = setup$T0
            
            # Generate pre-sparsify weights
            setup_no_sparsify = synthdid_estimate(Y, N0, T0, sparsify=NULL)
            
            # Generate sparsified weights (default)
            setup_with_sparsify = synthdid_estimate(Y, N0, T0)
            
            # Analyze sparsify behavior
            analysis = list(
                # Dimensions and setup info
                dimensions = list(
                    N0 = N0,
                    T0 = T0,
                    Y_shape = dim(Y)
                ),
                # Pre-sparsify weights
                pre_sparsify = list(
                    lambda = as.numeric(attr(setup_no_sparsify, "weights")$lambda),
                    omega = as.numeric(attr(setup_no_sparsify, "weights")$omega)
                ),
                # Sparsified weights
                post_sparsify = list(
                    lambda = as.numeric(attr(setup_with_sparsify, "weights")$lambda),
                    omega = as.numeric(attr(setup_with_sparsify, "weights")$omega)
                ),
                # Estimates
                estimates = list(
                    pre_sparsify = as.numeric(setup_no_sparsify),
                    post_sparsify = as.numeric(setup_with_sparsify)
                )
            )
            
            # Export analysis to JSON
            write_json(analysis, "{output_file}", pretty=TRUE, auto_unbox=TRUE)
            """
            r_script = r_script.format(output_file=output_json_path)
            f.write(r_script)
        
        # Run R script
        process = subprocess.run(
            ["Rscript", r_script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Load results if file exists
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r') as f:
                result = json.load(f)
    
    return result


def set_up_california_prop99_data():
    """Prepare the California Prop 99 data for Python."""
    # Load the dataset from CSV
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_file_path = os.path.join(base_path, 'R-package', 'synthdid', 'data', 'california_prop99.csv')
    
    if not os.path.exists(data_file_path):
        current_file_dir = os.path.dirname(__file__)
        project_root_guess = os.path.abspath(os.path.join(current_file_dir, '..'))
        data_file_path = os.path.join(project_root_guess, 'R-package', 'synthdid', 'data', 'california_prop99.csv')
        if not os.path.exists(data_file_path):
            pytest.skip(f"Data file california_prop99.csv not found at expected location: {data_file_path}")
            return None
    
    df = pd.read_csv(data_file_path, delimiter=';')
    
    # Prepare data similar to how R's panel.matrices works
    unit_col, time_col, outcome_col, treatment_col = "State", "Year", "PacksPerCapita", "treated"
    
    # Get treated and control states
    treated_flags = df.groupby(unit_col)[treatment_col].max()
    control_states = [state for state, treated in treated_flags.items() if treated == 0]
    treated_states = [state for state, treated in treated_flags.items() if treated == 1]
    
    # Sort states alphabetically to match R's factor ordering
    control_states = sorted(control_states)
    treated_states = sorted(treated_states)
    
    # Ensure California is correctly identified as treated
    if "California" not in treated_states:
        control_states = [state for state in control_states if state != "California"]
        treated_states = ["California"]
    
    # Order states: control states first, then treated states
    ordered_states = list(control_states) + list(treated_states)
    
    # Create Y matrix with the correct ordering
    Y_df_ordered = df.pivot(index=unit_col, columns=time_col, values=outcome_col)
    Y_df_ordered = Y_df_ordered.loc[ordered_states]  # Reorder rows
    Y_df_ordered = Y_df_ordered.sort_index(axis=1)   # Ensure time is ordered
    Y_np_full = Y_df_ordered.values.astype(np.float64)
    
    # Calculate N0 and T0
    N0 = len(control_states)
    T0 = 19  # Known pre-treatment periods for California Prop 99
    
    return {
        'Y': Y_np_full,
        'N0': N0,
        'T0': T0
    }


def test_sparsify_behavior():
    """Compare how sparsify function behaves in R and Python."""
    # Get R weights with analysis
    r_analysis = get_r_weights_with_sparsify_analysis()
    
    # Verify dimensions between R and Python
    assert r_analysis is not None, "Failed to get R analysis results"
    print("R dimensions:", r_analysis['dimensions'])
    
    # Set up Python data
    py_data = set_up_california_prop99_data()
    assert py_data is not None, "Failed to set up Python data"
    
    # Generate pre-sparsify weights in Python
    py_no_sparsify = synthdid_estimate(py_data['Y'], py_data['N0'], py_data['T0'], sparsify=None)
    
    # Generate sparsified weights in Python (default)
    py_with_sparsify = synthdid_estimate(py_data['Y'], py_data['N0'], py_data['T0'])
    
    # Compare pre-sparsify weights
    print("\nPre-sparsify lambda weights:")
    print("R:", r_analysis['pre_sparsify']['lambda'])
    print("Python:", py_no_sparsify.weights['lambda'])
    
    print("\nPre-sparsify omega weights:")
    print("R:", r_analysis['pre_sparsify']['omega'])
    print("Python:", py_no_sparsify.weights['omega'])
    
    # Compare post-sparsify weights
    print("\nPost-sparsify lambda weights:")
    print("R:", r_analysis['post_sparsify']['lambda'])
    print("Python:", py_with_sparsify.weights['lambda'])
    
    print("\nPost-sparsify omega weights:")
    print("R:", r_analysis['post_sparsify']['omega'])
    print("Python:", py_with_sparsify.weights['omega'])
    
    # Compare estimates
    print("\nEstimates:")
    print("R (pre-sparsify):", r_analysis['estimates']['pre_sparsify'])
    print("Python (pre-sparsify):", float(py_no_sparsify.estimate))
    print("R (post-sparsify):", r_analysis['estimates']['post_sparsify'])
    print("Python (post-sparsify):", float(py_with_sparsify.estimate))
    
    # Analyze sparsify function directly
    print("\nTesting direct sparsify application:")
    # Convert R pre-sparsify weights to numpy arrays
    r_lambda_pre = np.array(r_analysis['pre_sparsify']['lambda'])
    r_omega_pre = np.array(r_analysis['pre_sparsify']['omega'])
    
    # Apply Python's sparsify to R's pre-sparsify weights
    py_lambda_sparsified = sparsify_function(r_lambda_pre.astype(np.float64))
    py_omega_sparsified = sparsify_function(r_omega_pre.astype(np.float64))
    
    # R's post-sparsify weights
    r_lambda_post = np.array(r_analysis['post_sparsify']['lambda'])
    r_omega_post = np.array(r_analysis['post_sparsify']['omega'])
    
    # Compare direct sparsify application
    print("R lambda (pre):", r_lambda_pre)
    print("Python sparsified:", py_lambda_sparsified)
    print("R lambda (post):", r_lambda_post)
    
    print("R omega (pre):", r_omega_pre)
    print("Python sparsified:", py_omega_sparsified)
    print("R omega (post):", r_omega_post)
    
    # Analyze sparsification patterns
    r_lambda_zeros = np.isclose(r_lambda_post, 0)
    py_lambda_zeros = np.isclose(py_lambda_sparsified, 0)
    
    r_omega_zeros = np.isclose(r_omega_post, 0)
    py_omega_zeros = np.isclose(py_omega_sparsified, 0)
    
    print("\nZero patterns in lambda weights:")
    print("R zeros:", np.sum(r_lambda_zeros), "/", len(r_lambda_zeros))
    print("Python zeros:", np.sum(py_lambda_zeros), "/", len(py_lambda_zeros))
    print("Pattern match for lambda:", np.mean(r_lambda_zeros == py_lambda_zeros) * 100, "%")
    
    print("\nZero patterns in omega weights:")
    print("R zeros:", np.sum(r_omega_zeros), "/", len(r_omega_zeros))
    print("Python zeros:", np.sum(py_omega_zeros), "/", len(py_omega_zeros))
    print("Pattern match for omega:", np.mean(r_omega_zeros == py_omega_zeros) * 100, "%")
    
    # Compare non-zero weights
    if np.any(~r_lambda_zeros & ~py_lambda_zeros):
        non_zero_lambda_match = np.allclose(
            r_lambda_post[~r_lambda_zeros], 
            py_lambda_sparsified[~py_lambda_zeros], 
            atol=TOLERANCE
        )
        print("\nNon-zero lambda weights match:", non_zero_lambda_match)
    
    if np.any(~r_omega_zeros & ~py_omega_zeros):
        non_zero_omega_match = np.allclose(
            r_omega_post[~r_omega_zeros], 
            py_omega_sparsified[~py_omega_zeros], 
            atol=TOLERANCE
        )
        print("\nNon-zero omega weights match:", non_zero_omega_match)
    
    # Verify that Python implementation matches the expected behavior
    assert abs(float(py_with_sparsify.estimate) - r_analysis['estimates']['post_sparsify']) < TOLERANCE, \
        "Final estimates differ too much between R and Python"


if __name__ == "__main__":
    test_sparsify_behavior() 