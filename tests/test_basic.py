import numpy as np
import pandas as pd
import pytest

# Simplified imports from the package root
from synthdid_py import (
    random_low_rank, panel_matrices,
    synthdid_estimate, sc_estimate, did_estimate, 
    synthdid_effect_curve, synthdid_placebo,
    synthdid_se, vcov
)

def test_random_low_rank():
    """Test random low rank data generation"""
    setup = random_low_rank(n_0=10, n_1=3, T_0=8, T_1=4, tau=1.0)
    
    # Check return structure
    assert 'Y' in setup
    assert 'N0' in setup
    assert 'T0' in setup
    
    # Check matrix dimensions
    N = 10 + 3  # n_0 + n_1
    T = 8 + 4   # T_0 + T_1
    assert setup['Y'].shape == (N, T)
    assert setup['N0'] == 10
    assert setup['T0'] == 8

def test_basic_synthdid_estimate():
    """Test basic synthdid estimation"""
    # Generate test data
    setup = random_low_rank(n_0=10, n_1=3, T_0=8, T_1=4, tau=1.0)
    
    # Run estimator
    est = synthdid_estimate(setup['Y'], setup['N0'], setup['T0'])
    
    # Check that estimate is a scalar
    assert isinstance(est.estimate, float)
    
    # Check that estimate is close to the true effect (which is 1.0)
    assert 0 < est.estimate < 2.0, f"Expected estimate around 1.0, got {est.estimate}"
    
    # Check that weights are present in the output
    assert 'lambda' in est.weights
    assert 'omega' in est.weights
    
    # Check weights sum to 1
    assert abs(sum(est.weights['lambda']) - 1.0) < 1e-10
    assert abs(sum(est.weights['omega']) - 1.0) < 1e-10

def test_sc_estimate():
    """Test synthetic control estimation"""
    # Generate test data
    setup = random_low_rank(n_0=10, n_1=3, T_0=8, T_1=4, tau=1.0)
    
    # Run SC estimator
    est = sc_estimate(setup['Y'], setup['N0'], setup['T0'])
    
    # Check that estimate is a scalar
    assert isinstance(est.estimate, float)
    
    # Check that weights are present in the output
    assert 'lambda' in est.weights
    assert 'omega' in est.weights
    
    # For SC, omega weights sum to 1
    assert abs(sum(est.weights['omega']) - 1.0) < 1e-10

def test_did_estimate():
    """Test difference-in-differences estimation"""
    # Generate test data
    setup = random_low_rank(n_0=10, n_1=3, T_0=8, T_1=4, tau=1.0)
    
    # Run DiD estimator
    est = did_estimate(setup['Y'], setup['N0'], setup['T0'])
    
    # Check that estimate is a scalar
    assert isinstance(est.estimate, float)
    
    # Check that weights are present in the output
    assert 'lambda' in est.weights
    assert 'omega' in est.weights
    
    # Lambda weights should be uniform
    lambda_len = len(est.weights['lambda'])
    assert np.allclose(est.weights['lambda'], 1/lambda_len)
    
    # Omega weights should be uniform 
    omega_len = len(est.weights['omega'])
    assert np.allclose(est.weights['omega'], 1/omega_len)

def test_effect_curve():
    """Test effect curve calculation"""
    # Generate test data
    setup = random_low_rank(n_0=10, n_1=3, T_0=8, T_1=4, tau=2.0)
    
    # Run estimator
    est = synthdid_estimate(setup['Y'], setup['N0'], setup['T0'])
    
    # Calculate effect curve
    effect_curve = synthdid_effect_curve(est)
    
    # Check output shape (should be T1 elements)
    assert len(effect_curve) == 4
    
    # All effects should be close to the true effect (2.0) with a more relaxed tolerance
    # Since this is a random test, some variation is expected
    assert np.all(np.abs(effect_curve - 2.0) < 1.5), f"Expected effects close to 2.0, got {effect_curve}"

def test_placebo():
    """Test placebo tests"""
    # Generate test data
    setup = random_low_rank(n_0=10, n_1=3, T_0=8, T_1=4, tau=1.0)
    
    # Run estimator
    est = synthdid_estimate(setup['Y'], setup['N0'], setup['T0'])
    
    # Run placebo test
    placebo_est = synthdid_placebo(est)
    
    # Verify it returns a SynthDIDEstimate object
    assert hasattr(placebo_est, 'estimate')
    assert hasattr(placebo_est, 'weights')

def test_standard_errors():
    """Test standard error calculation methods"""
    # Generate test data
    setup = random_low_rank(n_0=10, n_1=3, T_0=8, T_1=4, tau=1.0)
    
    # Run estimator
    est = synthdid_estimate(setup['Y'], setup['N0'], setup['T0'])
    
    # Calculate standard errors with different methods
    se_jackknife = synthdid_se(est, method="jackknife")
    se_bootstrap = synthdid_se(est, method="bootstrap", replications=50)
    se_placebo = synthdid_se(est, method="placebo")
    
    # Check output structure
    for se_result in [se_jackknife, se_bootstrap, se_placebo]:
        assert 'se' in se_result
        assert se_result['se'] > 0

if __name__ == "__main__":
    test_random_low_rank()
    test_basic_synthdid_estimate()
    test_sc_estimate()
    test_did_estimate()
    test_effect_curve()
    test_placebo()
    test_standard_errors()
    print("All basic tests passed!") 