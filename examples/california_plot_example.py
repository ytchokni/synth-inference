"""
Example script showing how to use the Python implementation of synthdid
to analyze the effect of California Proposition 99 on cigarette consumption.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from synthdid_py import synthdid_estimate, synthdid_effect_curve, plot_synthdid, synthdid_se

def load_california_prop99_data():
    """
    Load the California Proposition 99 data from the CSV file.
    
    Returns:
        tuple: (Y, N0, T0) where Y is the outcome matrix, N0 is the number of 
               control units, and T0 is the number of pre-treatment periods.
    """
    # Location of the real data file
    csv_path = "./R-package/synthdid/data/california_prop99.csv"
    
    print(f"Loading California Prop 99 data from {csv_path}...")
    # Read the CSV with semicolon delimiter
    df = pd.read_csv(csv_path, sep=';')
    
    # Convert to wide format (states × years)
    Y = df.pivot(index='State', columns='Year', values='PacksPerCapita')
    
    # Treatment starts in 1989 (find column index)
    years = sorted(df['Year'].unique())
    T0 = years.index(1989)
    
    # California should be the last row
    states = list(Y.index)
    if 'California' in states and states[-1] != 'California':
        # Reorder to put California last
        other_states = [s for s in states if s != 'California']
        Y = Y.loc[other_states + ['California']]
    
    # All states except California are controls
    N0 = Y.shape[0] - 1
    
    return Y.values, N0, T0

def california_prop99_example():
    """
    Analyze the California Proposition 99 data using only Python implementation
    and plot the results using matplotlib.
    """
    try:
        print("Loading California Prop 99 data...")
        Y, N0, T0 = load_california_prop99_data()
        
        print("Running Python implementation...")
        # Run the Python estimator
        est = synthdid_estimate(Y, N0, T0)
        effect_curve = synthdid_effect_curve(est)
        
        # Calculate standard error using placebo method
        se_result = synthdid_se(est, method="placebo")
        se = float(se_result["se"])
        
        # Print estimates and confidence interval
        print(f"Python estimate: {est.estimate:.4f} (SE: {se:.4f})")
        ci_lower = est.estimate - 1.96 * se
        ci_upper = est.estimate + 1.96 * se
        print(f"95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")
        
        print("Creating plot...")
        # Create and display the plot with custom styling
        fig, ax = plot_synthdid(
            est, 
            effect_curve=effect_curve,
            title="California Proposition 99 Effect on Cigarette Consumption", 
            ylabel="Packs per Capita",
            xlabel="Year",
            control_color="salmon",
            treated_color="turquoise",
            effect_color="salmon",
            show_effect_area=False,  # Turn off the effect area since we want to show lambda weights instead
            show_guides=True,
            show_arrow=True,
            lambda_plot_scale=3  # Add lambda weights visualization
        )
        
        # Customize plot with actual years
        years = range(1970, 2001)  # Actual years for the California dataset
        ax.set_xticks(range(0, 31, 5))
        ax.set_xticklabels([years[i] for i in range(0, 31, 5)])
        
        # Save the plot
        fig.savefig("california_prop99_plot.png", dpi=300, bbox_inches="tight")
        print("Plot saved as 'california_prop99_plot.png'")
        
        return True
    except Exception as e:
        print(f"Error in California Prop 99 example: {e}")
        return False

if __name__ == "__main__":
    california_prop99_example() 