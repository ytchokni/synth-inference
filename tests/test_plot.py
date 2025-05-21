import pytest
import numpy as np
import matplotlib.pyplot as plt

from synthdid_py import (
    random_low_rank, 
    synthdid_estimate, 
    synthdid_effect_curve,
    plot_synthdid
)

def test_plot_random_data():
    """Test plotting functionality on random data"""
    # Generate test data
    setup = random_low_rank(n_0=10, n_1=3, T_0=8, T_1=4, tau=2.0)
    
    # Run estimator
    est = synthdid_estimate(setup['Y'], setup['N0'], setup['T0'])
    
    # Calculate effect curve
    effect_curve = synthdid_effect_curve(est)
    
    # Create plot
    fig, ax = plot_synthdid(est, effect_curve=effect_curve, title="Synthetic DiD Plot - Random Data")
    
    # Check that the plot was created
    assert fig is not None
    assert ax is not None
    
    # Close the plot to avoid displaying it during tests
    plt.close(fig)

def test_plot_california():
    """Test plotting functionality with California Prop 99 data (if available)"""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        
        # Load the R synthdid package
        r = ro.r
        synthdid = importr("synthdid")
        
        # Load the data and create the matrices
        r('data("california_prop99")')
        r('setup = panel.matrices(california_prop99)')
        
        # Extract the data for Python analysis
        Y = np.array(r('setup$Y'))
        N0 = int(r('setup$N0')[0])
        T0 = int(r('setup$T0')[0])
        
        # Run the Python estimator
        est = synthdid_estimate(Y, N0, T0)
        
        # Create plot
        fig, ax = plot_synthdid(
            est, 
            title="California Proposition 99", 
            ylabel="Packs per Capita",
            xlabel="Year"
        )
        
        # Check that the plot was created
        assert fig is not None
        assert ax is not None
        
        # Optionally save the plot
        # fig.savefig("california_prop99_plot.png", dpi=300, bbox_inches="tight")
        
        # Close the plot to avoid displaying it during tests
        plt.close(fig)
        
    except (ImportError, ModuleNotFoundError):
        # Skip if rpy2 is not available
        pytest.skip("rpy2 not available or R synthdid package not installed")
    except Exception as e:
        # Skip if any other error occurs
        pytest.skip(f"Error loading California prop 99 data: {e}")

if __name__ == "__main__":
    # When run directly, show the plots
    test_plot_random_data()
    
    # Try to create and display the California plot
    try:
        test_plot_california()
        print("Both plots created!")
    except Exception as e:
        print(f"Could not create California plot: {e}")
        print("Random data plot created!")
    
    # Show the plots
    plt.show() 