import os
import pytest
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, PackageNotInstalledError

# Initialize rpy2
pandas2ri.activate()

@pytest.fixture(scope="module")
def r_synthdid():
    """Load the R synthdid package"""
    try:
        # If synthdid is installed, load it
        synthdid = importr("synthdid")
        return synthdid
    except PackageNotInstalledError:
        # If synthdid is not installed, install it from the local package
        r_base = importr("base")
        utils = importr("utils")
        package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                   "../R-package/synthdid"))
        r_base.Sys_setenv(R_LIBS_USER=r_base.tempdir())
        utils.install_packages(f"file://{package_path}", 
                              repos=ro.NULL, 
                              type="source")
        synthdid = importr("synthdid")
        return synthdid

@pytest.fixture(scope="module")
def california_prop99(r_synthdid):
    """Load the California Prop 99 dataset from R"""
    # Use R commands directly
    ro.r('data(california_prop99, package="synthdid")')
    california_prop99 = ro.conversion.rpy2py(ro.r['california_prop99'])
    return california_prop99

def test_r_synthdid_estimate(r_synthdid, california_prop99):
    """Test basic R synthdid estimation on California Prop 99 data"""
    # Convert pandas dataframe to R dataframe
    california_prop99_r = pandas2ri.py2rpy(california_prop99)
    
    # Create panel matrices using R's panel.matrices function
    panel_matrices = r_synthdid.panel_matrices(california_prop99_r, 
                                              ro.StrVector(["Year"]), 
                                              ro.StrVector(["State"]),
                                              ro.StrVector(["PacksPerCapita"]))
    
    # Extract Y, N0, T0 from the panel matrices
    Y = panel_matrices.rx2("Y")
    N0 = panel_matrices.rx2("N0")
    T0 = panel_matrices.rx2("T0")
    
    # Call R's synthdid_estimate function
    est = r_synthdid.synthdid_estimate(Y, N0, T0)
    
    # Extract the estimate value
    estimate_value = float(ro.conversion.rpy2py(est)[0])
    
    # Assert that the estimate is reasonable (should be negative as per the paper)
    assert -25 < estimate_value < -5, f"Expected estimate between -25 and -5, got {estimate_value}"
    
    # Get standard error
    se = r_synthdid.synthdid_se(est)
    se_value = float(ro.conversion.rpy2py(se)[0])
    
    # Assert that the standard error is positive and reasonable
    assert 0 < se_value < 10, f"Expected standard error between 0 and 10, got {se_value}"
    
    # Return the estimates for later comparison with Python implementation
    return estimate_value, se_value

def test_sc_estimate(r_synthdid, california_prop99):
    """Test synthetic control estimation using R"""
    california_prop99_r = pandas2ri.py2rpy(california_prop99)
    panel_matrices = r_synthdid.panel_matrices(california_prop99_r, 
                                              ro.StrVector(["Year"]), 
                                              ro.StrVector(["State"]),
                                              ro.StrVector(["PacksPerCapita"]))
    Y = panel_matrices.rx2("Y")
    N0 = panel_matrices.rx2("N0")
    T0 = panel_matrices.rx2("T0")
    
    # Call R's sc_estimate function
    est = r_synthdid.sc_estimate(Y, N0, T0)
    
    # Extract the estimate value
    estimate_value = float(ro.conversion.rpy2py(est)[0])
    
    # Assert that the estimate is reasonable
    assert -25 < estimate_value < -5, f"Expected estimate between -25 and -5, got {estimate_value}"
    
    return estimate_value

def test_did_estimate(r_synthdid, california_prop99):
    """Test difference-in-differences estimation using R"""
    california_prop99_r = pandas2ri.py2rpy(california_prop99)
    panel_matrices = r_synthdid.panel_matrices(california_prop99_r, 
                                              ro.StrVector(["Year"]), 
                                              ro.StrVector(["State"]),
                                              ro.StrVector(["PacksPerCapita"]))
    Y = panel_matrices.rx2("Y")
    N0 = panel_matrices.rx2("N0")
    T0 = panel_matrices.rx2("T0")
    
    # Call R's did_estimate function
    est = r_synthdid.did_estimate(Y, N0, T0)
    
    # Extract the estimate value
    estimate_value = float(ro.conversion.rpy2py(est)[0])
    
    # Assert that the estimate is reasonable
    assert -25 < estimate_value < -5, f"Expected estimate between -25 and -5, got {estimate_value}"
    
    return estimate_value 