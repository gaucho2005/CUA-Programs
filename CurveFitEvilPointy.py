import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy.optimize import curve_fit
from scipy.integrate import quad
from pathlib import Path

# ---------------------------------------------------------
# 1. Double-Sided Crystal Ball & Background Models
# ---------------------------------------------------------

def double_sided_crystal_ball(x, alpha_l, n_l, alpha_r, n_r, mean, sigma, N):
    """
    Double-Sided Crystal Ball (DSCB) function.
    Provides independent power-law tails for both the low-mass (left) 
    and high-mass (right) components of the Z peak.
    """
    z = (x - mean) / sigma
    
    # Left Tail Parameters
    al = np.abs(alpha_l)
    Al = (n_l / al)**n_l * np.exp(-al**2 / 2.0)
    Bl = n_l / al - al
    
    # Right Tail Parameters
    ar = np.abs(alpha_r)
    Ar = (n_r / ar)**n_r * np.exp(-ar**2 / 2.0)
    Br = n_r / ar - ar
    
    # Core Gaussian
    core = N * np.exp(-0.5 * z**2)
    
    # Left Tail analytic form
    left_base = Bl - z
    left_tail = N * Al * np.power(np.maximum(left_base, 1e-9), -n_l)
    
    # Right Tail analytic form
    right_base = Br + z
    right_tail = N * Ar * np.power(np.maximum(right_base, 1e-9), -n_r)
    
    # Piecewise selection using np.select
    return np.select(
        [z < -al, z > ar],
        [left_tail, right_tail],
        default=core
    )

def polynomial_background(x, c0, c1, c2):
    # Centering around 90 GeV keeps the fit matrix mathematically stable
    return c0 + c1 * (x - 90.0) + c2 * (x - 90.0)**2

def total_model(x, alpha_l, n_l, alpha_r, n_r, mean, sigma, N, c0, c1, c2):
    """Signal (DSCB) + Background"""
    return double_sided_crystal_ball(x, alpha_l, n_l, alpha_r, n_r, mean, sigma, N) + \
           polynomial_background(x, c0, c1, c2)

# ---------------------------------------------------------
# 2. Optimized Fitting Routine
# ---------------------------------------------------------

def fit_z_mass(df: pl.DataFrame, mass_col: str = "mass", bins: int = 120, range_min: float = 60, range_max: float = 120):
    masses = df[mass_col].to_numpy()
    
    counts, bin_edges = np.histogram(masses, bins=bins, range=(range_min, range_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]
    
    max_intensity = np.max(counts)
    
    # --- FIX 1: Estimate background baseline using the sidebands (outer 10% of bins) ---
    left_sideband = np.mean(counts[:int(bins*0.10)])
    right_sideband = np.mean(counts[-int(bins*0.10):])
    estimated_bkg_baseline = max(1.0, (left_sideband + right_sideband) / 2.0)
    
    # Clean signal peak height is maximum data minus the baseline background
    estimated_sig_height = max(10.0, max_intensity - estimated_bkg_baseline)
    
    # Initial Guesses
    # [alpha_l, n_l, alpha_r, n_r, mean, sigma, N, c0, c1, c2]
    p0 = [1.5, 3.5, 1.5, 3.5, 88.5, 2.5, estimated_sig_height, estimated_bkg_baseline, 0.0, 0.0]
    
    # --- FIX 2: Force n_l and n_r to be >= 2.5 so the tails cannot mimic flat background ---
    
    bounds = (
        [0.1, 2.5,  0.1, 2.5,  85.0, 0.5, estimated_sig_height * 0.5, 0.0,     -np.inf, -np.inf], # Lower bounds
        [5.0, 20.0, 5.0, 20.0, 93.0, 6.0, estimated_sig_height * 1.5, np.inf,  np.inf,  np.inf]  # Upper bounds
    )
    
    bin_errors = np.sqrt(np.maximum(counts, 1))
    
    popt, pcov = curve_fit(
        total_model, 
        bin_centers, 
        counts, 
        p0=p0, 
        bounds=bounds,
        sigma=bin_errors,
        absolute_sigma=True,
        maxfev=50000
    )
    
    return popt, pcov, bin_centers, counts, bin_width

# ---------------------------------------------------------
# 3. Scalar Integration Function
# ---------------------------------------------------------

def integrate_signal_model(popt, lower_bound, upper_bound): #must set integration bounds with consideration for min/max range
    """Integrates ONLY the Double-Sided Crystal Ball signal function"""
    alpha_l, n_l, alpha_r, n_r, mean, sigma, N = popt[:7]
    
    integral, _ = quad(
        double_sided_crystal_ball, 
        lower_bound, 
        upper_bound, 
        args=(alpha_l, n_l, alpha_r, n_r, mean, sigma, N),
        limit=200
    )
    return integral

# ---------------------------------------------------------
# 4. Plotting Helper
# ---------------------------------------------------------

def plot_fit(bin_centers, counts, popt):
    plt.figure(figsize=(10, 6))
    
    # Plot Data
    plt.errorbar(bin_centers, counts, yerr=np.sqrt(np.maximum(counts, 1)), fmt='ko', markersize=3, label='Actual Data (Polars df)', zorder=3)
    
    x_smooth = np.linspace(60, 120, 1000)
    
    # Plot Components
    plt.plot(x_smooth, total_model(x_smooth, *popt), 'r-', linewidth=2, label='Total Fit (DSCB + Poly2)', zorder=4)
    plt.plot(x_smooth, double_sided_crystal_ball(x_smooth, *popt[:7]), 'b--', linewidth=1.8, label='Signal (Double-Sided CB)', zorder=2)
    plt.plot(x_smooth, polynomial_background(x_smooth, *popt[7:]), 'g:', linewidth=2, label='Background Baseline', zorder=1)
    
    plt.title(r'$Z \rightarrow ee$ High-Statistics Invariant Mass Fit', fontsize=14)
    plt.xlabel('Invariant Mass [GeV]', fontsize=12)
    plt.ylabel('Events / Bin', fontsize=12)
    plt.text(0.05, 0.80, f"Fitted Mean: {popt[4]:.2f} GeV\nFitted Sigma: {popt[5]:.2f} GeV",transform=plt.gca().transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    title="Freaky Ahh Hist"
    out_folder = Path("/home/ciroj/CUA/testing/Charts")
    outfile = out_folder / f"{title}.png"
    plt.savefig(outfile)
    subprocess.run(["explorer.exe", str(outfile).replace("/", "\\")],shell=False)

def main(dataframe):
    # --- You will replace this block with your actual Polars DataFrame ---
     # 1. Mocking some Z->ee data to represent your Polars DataFrame
    np.random.seed(109)
    # Signal: rough approximation of a left-tailed distribution around 91
    signal_data = np.random.normal(91, 2.5, 8000)
    tail_data = np.random.exponential(5, 2000) 
    signal_data = np.concatenate([signal_data, 91 - tail_data]) # Fake the brem tail
    # Background: flat-ish slope
    bkg_data = np.random.uniform(60, 120, 3000) 
    
    # Put it into a Polars DataFrame
    df = dataframe
    #df = pl.DataFrame({"mass": np.concatenate([signal_data, bkg_data])})
    # ---------------------------------------------------------------------

    popt, pcov, bin_centers, counts, bin_width = fit_z_mass(df, mass_col="mass")
    
    raw_integral = integrate_signal_model(popt, lower_bound=60, upper_bound=120)
    number_of_signal_events = raw_integral / bin_width
    if number_of_signal_events>len(df):
        number_of_signal_events=len(df)
    print(f"AAAAAAAAAAAAAA {df['probe_pt']}")
    print(f"Total Signal Events from Integral: {number_of_signal_events:.0f}")
    print(f"Extracted Z Mass Mean: {popt[2]:.2f} GeV")
    print(f"REMOVED EVENTS FROM BACKGROUND: {len(df)-number_of_signal_events} ({(len(df)-number_of_signal_events)/(.01*len(df))}%)")
    #plot_fit(bin_centers, counts, popt)
    return number_of_signal_events

