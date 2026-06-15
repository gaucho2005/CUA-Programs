import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy.optimize import curve_fit
from scipy.integrate import quad
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad

# ---------------------------------------------------------
# 1. Smooth Crystal Ball (Gradient-Friendly)
# ---------------------------------------------------------

def crystal_ball_signal(x, alpha, n, mean, sigma, N):
    """
    A mathematically smoothed version of the Crystal Ball function.
    Removing sharp discontinuities helps the optimizer scale the peak properly.
    """
    a = np.abs(alpha)
    A = (n / a)**n * np.exp(-a**2 / 2.0)
    B = n / a - a
    z = (x - mean) / sigma
    
    # Smooth implementation of the piecewise condition
    # Using a soft threshold prevents the fitter from getting stuck
    term_tail = N * A * np.power(B - z, -n)
    term_gauss = N * np.exp(-0.5 * z**2)
    
    # Combine them smoothly based on where z sits relative to -alpha
    is_tail = z < -a
    return np.where(is_tail, term_tail, term_gauss)

def polynomial_background(x, c0, c1, c2):
    # Centering around 91.18 stabilizes the matrix inversion in curve_fit
    return c0 + c1 * (x - 91.18) + c2 * (x - 91.18)**2

def total_model(x, alpha, n, mean, sigma, N, c0, c1, c2):
    return crystal_ball_signal(x, alpha, n, mean, sigma, N) + polynomial_background(x, c0, c1, c2)

# ---------------------------------------------------------
# 2. Aggressive Fitting Routine
# ---------------------------------------------------------

def fit_z_mass(df: pl.DataFrame, mass_col: str = "mass", bins: int = 120, range_min: float = 60, range_max: float = 120):
    masses = df[mass_col].to_numpy()
    
    counts, bin_edges = np.histogram(masses, bins=bins, range=(range_min, range_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]
    
    max_intensity = np.max(counts)
    
    # Setting an aggressive, deliberate starting point
    # [alpha, n, mean, sigma, N, c0, c1, c2]
    p0 = [1.2, 3.0, 90.8, 1.8, max_intensity, counts[0], -0.2, 0.0]
    
    # Using statistical weights forces the peak to be prioritized
    bin_errors = np.sqrt(np.maximum(counts, 1))
    
    # SWITCHED METHOD TO 'lm' (Levenberg-Marquardt)
    # This algorithm ignores bounds entirely and focuses strictly on minimizing Chi-Sq.
    # If the peak is short, 'lm' will scale N dynamically until it hits the target.
    popt, pcov = curve_fit(
        total_model, 
        bin_centers, 
        counts, 
        p0=p0, 
        sigma=bin_errors,
        absolute_sigma=True,
        method='lm',       # <--- Crucial change
        maxfev=10000       # Allows the fitter more iterations to scale up
    )
    
    return popt, pcov, bin_centers, counts, bin_width

# ---------------------------------------------------------
# 3. Integration & Execution
# ---------------------------------------------------------

def integrate_signal_model(popt, lower_bound=0.0, upper_bound=120.0):
    alpha, n, mean, sigma, N = popt[:5]
    integral, _ = quad(crystal_ball_signal, lower_bound, upper_bound, args=(alpha, n, mean, sigma, N))
    return integral
# ---------------------------------------------------------
# 4. Plotting & Labeling Helper
# ---------------------------------------------------------

def plot_fit_results(bin_centers, counts, popt):
    """Generates a clearly labeled plot of the data, total fit, and components."""
    plt.figure(figsize=(10, 6))
    
    # Data
    plt.errorbar(bin_centers, counts, yerr=np.sqrt(counts), fmt='ko', markersize=4, label='Data (Polars df)', zorder=3)
    
    # X-axis for smooth plotting
    x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 500)
    
    # Total Fit
    total_y = total_model(x_smooth, *popt)
    plt.plot(x_smooth, total_y, 'r-', linewidth=2, label='Total Fit (Sig + Bkg)', zorder=2)
    
    # Signal Component
    sig_y = crystal_ball_signal(x_smooth, *popt[:5])
    plt.plot(x_smooth, sig_y, 'b--', linewidth=2, label='Signal (Crystal Ball)', zorder=2)
    
    # Background Component
    bkg_y = polynomial_background(x_smooth, *popt[5:])
    plt.plot(x_smooth, bkg_y, 'g:', linewidth=2, label='Background (2nd Order Poly)', zorder=2)
    
    # Formatting and Labeling
    plt.title(r'$Z \rightarrow ee$ Invariant Mass Fit', fontsize=14)
    plt.xlabel('Invariant Mass [GeV]', fontsize=12)
    plt.ylabel('Events / Bin', fontsize=12)
    
    # Display the fitted mass directly on the plot
    plt.text(0.05, 0.75, f"Fitted Z Mass: {popt[2]:.2f} GeV", transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    title="Freaky Ahh Hist"
    out_folder = Path("/home/ciroj/CUA/testing/Charts")
    outfile = out_folder / f"{title}.png"
    plt.savefig(outfile)
    subprocess.run(["explorer.exe", str(outfile).replace("/", "\\")],shell=False)

# ---------------------------------------------------------
# Example Usage 
# ---------------------------------------------------------
def main(dataframe):
    
    # 1. Mocking some Z->ee data to represent your Polars DataFrame
    np.random.seed(42)
    # Signal: rough approximation of a left-tailed distribution around 91
    signal_data = np.random.normal(91, 2.5, 8000)
    tail_data = np.random.exponential(5, 2000) 
    signal_data = np.concatenate([signal_data, 91 - tail_data]) # Fake the brem tail
    # Background: flat-ish slope
    bkg_data = np.random.uniform(60, 120, 3000) 
    
    # Put it into a Polars DataFrame
    #df = pl.DataFrame({"mass": np.concatenate([signal_data, bkg_data])})
    df=dataframe
    
    # 2. Perform the fit
    popt, pcov, bin_centers, counts, bin_width = fit_z_mass(df, mass_col="mass")
    
    # 3. Integrate the signal
    # Integrating from 0 to 120 as requested
    signal_integral = integrate_signal_model(popt, lower_bound=0, upper_bound=120)
    print(f"Total Integrated Signal Events (0 to 120): {signal_integral:.2f}")
    print(f"Extracted Z Mass Mean: {popt[2]:.2f} GeV")
    # 4. Divide by bin width to get actual number of physical events
    number_of_signal_events = signal_integral / bin_width
    
    print(f"Bin Width: {bin_width} GeV")
    print(f"Raw Curve Area: {signal_integral:.2f}")
    print(f"Actual Number of Signal Events: {number_of_signal_events:.0f}")
    print(f"Extracted Z Mass Mean: {popt[2]:.2f} GeV")
    # 5. Plot and label
    plot_fit_results(bin_centers, counts, popt)
