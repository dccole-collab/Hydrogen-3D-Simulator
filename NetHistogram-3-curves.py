import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from scipy.stats import gaussian_kde # This imports the smoothing function

# ==========================================
# --- Configuration ---
# ==========================================
a_0 = 0.529177 # Bohr radius in Angstroms
DOUBLES_PER_FRAME = 8 # Binary data structure

# Target time slices in seconds
TIMES_PS = [2.0, 5.0, 20.0]
TIMES_S = [t * 1e-12 for t in TIMES_PS]
# ==========================================

def compute_theoretical_pr(r_array):
    """Calculates the QM P(r) for the 1s ground state."""
    return (4.0 / (a_0**3)) * (r_array**2) * np.exp(-2.0 * r_array / a_0)

def load_3_curves(directory):
    """Reads .bin files and extracts r and dt for three different time brackets."""
    r_data = {2.0: [], 5.0: [], 20.0: []}
    dt_data = {2.0: [], 5.0: [], 20.0: []}
    
    file_pattern = os.path.join(directory, "*.bin")
    run_files = glob.glob(file_pattern)
    
    if not run_files:
        print(f"No .bin files found in {directory}!")
        sys.exit(1)

    print(f"Found {len(run_files)} binary trajectory files. Aggregating data...")
    
    for file in run_files:
        raw_data = np.fromfile(file, dtype=np.float64)
        
        # Safely chop off any incomplete final frame
        valid_length = (len(raw_data) // DOUBLES_PER_FRAME) * DOUBLES_PER_FRAME
        if len(raw_data) != valid_length:
            print(f"Notice: Dropped a partially written final frame from {os.path.basename(file)}.")
            
        reshaped_data = raw_data[:valid_length].reshape(-1, DOUBLES_PER_FRAME)

        # Extract full arrays
        t_full = reshaped_data[:, 0]
        r_full_cm = reshaped_data[:, 1]
        r_full_A = r_full_cm * 1e8 # Convert cm to Angstroms
        
        # Loop through our three time limits and slice the arrays
        for limit_ps, limit_s in zip(TIMES_PS, TIMES_S):
            mask = (t_full <= limit_s)
            t = t_full[mask]
            r = r_full_A[mask]
            
            if len(t) < 2: 
                continue
                
            # Calculate the time weights for this specific slice
            dt = np.diff(t)
            dt = np.append(dt, dt[-1])
            
            r_data[limit_ps].extend(r)
            dt_data[limit_ps].extend(dt)
            
        print(f"Processed {os.path.basename(file)}")
        
    return r_data, dt_data

def main():
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        print("Usage: python3 NetHistogram-3-curves.py /path/to/trajectory/folder")
        return

    # 1. Load all data
    r_data, dt_data = load_3_curves(data_dir)

    # 2. Set up the plot
    plt.figure(figsize=(10, 6))
    
    # 3. Calculate and plot the exact QM Ground State line (Red, Solid)
    r_theoretical = np.linspace(0, 2.5, 1000)
    p_theoretical = compute_theoretical_pr(r_theoretical)
    plt.plot(r_theoretical, p_theoretical, color='red', linewidth=2.5, 
             label=r'QM Ground State $P(r)$')

    # 4. Plot smooth KDE curves using the dt time-weights
    r_eval = np.linspace(0, 2.5, 500) # Points to draw the smooth line

    # 2.0 ps curve (Black, Dotted)
    kde_2 = gaussian_kde(r_data[2.0], weights=dt_data[2.0])
    plt.plot(r_eval, kde_2(r_eval), color='black', linestyle=':', linewidth=1.5, 
             label='SED Ensemble Average (t $\leq$ 2.0 ps)')

    # 5.0 ps curve (Black, Dashed)
    kde_7_5 = gaussian_kde(r_data[5.0], weights=dt_data[5.0])
    plt.plot(r_eval, kde_7_5(r_eval), color='black', linestyle='--', linewidth=1.5, 
             label='SED Ensemble Average (t $\leq$ 5.0 ps)')

    # 20.0 ps curve (Black, Solid, Bolder)
    kde_32 = gaussian_kde(r_data[20.0], weights=dt_data[20.0])
    plt.plot(r_eval, kde_32(r_eval), color='black', linestyle='-', linewidth=2.5, 
             label='SED Ensemble Average (t $\leq$ 20.0 ps)')

    # 5. Format and display
    plt.title("Time-Weighted SED Evolution vs. Quantum Mechanics", fontsize=14)
    plt.xlabel(r"Radial Distance $r$ (Å)", fontsize=12)
    plt.ylabel(r"Probability Density $P(r)$", fontsize=12)
    plt.xlim(0, 2.5)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()