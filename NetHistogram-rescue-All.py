import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys # NEW: Allows reading the command line arguments
# Use like this:If TRAJ is the folder with all your trajectory files (in bin format), with TRAJ at location
# /Volumes/SED_Archive/Simulation_Runs/
# USE:
#     python3 NetHistogram-All.py /Volumes/SED_Archive/Simulation_Runs/TRAJ
# to get the net histogram.
#
# ==========================================
# --- Configuration ---
# ==========================================
a_0 = 0.529177 # Bohr radius in Angstroms

# The target time slice in seconds (e.g., 8.0e-12 for 8 ps)
MAX_TIME_S = 25.0e-12 

# How many doubles make up one frame of your binary data?
DOUBLES_PER_FRAME = 8 
# ==========================================

def compute_theoretical_pr(r_array):
    """Calculates the QM P(r) for the 1s ground state."""
    return (4.0 / (a_0**3)) * (r_array**2) * np.exp(-2.0 * r_array / a_0)

def load_and_aggregate_runs(directory, max_time):
    """Reads .bin files, applies time mask, and aggregates r and dt."""
    all_r_values = []
    all_dt_weights = []
    
    file_pattern = os.path.join(directory, "*.bin")
    run_files = glob.glob(file_pattern)
    
    if not run_files:
        print(f"No .bin files found in {directory}!")
        return np.array([]), np.array([])

    print(f"Found {len(run_files)} binary trajectory files. Aggregating data up to {max_time*1e12} ps...")
    
    for file in run_files:
        raw_data = np.fromfile(file, dtype=np.float64)
        
        #try:
        #    reshaped_data = raw_data.reshape(-1, DOUBLES_PER_FRAME)
        #except ValueError:
        #    print(f"Warning: {file} doesn't divide by {DOUBLES_PER_FRAME}.")
        #    continue
          
# Safely chop off any incomplete final frame caused by copying an active file
        valid_length = (len(raw_data) // DOUBLES_PER_FRAME) * DOUBLES_PER_FRAME
        if len(raw_data) != valid_length:
            print(f"Notice: Dropped a partially written final frame from {os.path.basename(file)}.")
            
        reshaped_data = raw_data[:valid_length].reshape(-1, DOUBLES_PER_FRAME)


  
# 2. Extract full arrays directly from your 8 columns
        t_full = reshaped_data[:, 0]
        r_full = reshaped_data[:, 1] # Column 1 is already 'r'
        
# 3. Apply the Time Mask (Slice the data at MAX_TIME_S)
        
        mask = (t_full <= max_time)
        t = t_full[mask]
        r_raw = r_full[mask]
        
# If the file has no data before the max_time, skip i
        if len(t) < 2: 
            continue

# 4. Convert 'r' from centimeters to Angstroms
        r = r_raw * 1e8

        dt = np.diff(t)
        dt = np.append(dt, dt[-1]) 
        
        all_r_values.extend(r)
        all_dt_weights.extend(dt)
        
        print(f"Processed {os.path.basename(file)}: Kept {len(r)} time-weighted steps.")
        
    return np.array(all_r_values), np.array(all_dt_weights)

def main():
    # NEW: Check if the user provided a folder path in the terminal
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        print("Usage: python3 NetHistogram-All.py /path/to/trajectory/folder")
        return

    aggregated_r, aggregated_dt = load_and_aggregate_runs(data_dir, MAX_TIME_S)
    
    if len(aggregated_r) == 0:
        return

    plt.figure(figsize=(10, 6))
    
    plt.hist(aggregated_r, bins=200, range=(0, 2.5), 
             weights=aggregated_dt, density=True, alpha=0.6, color='royalblue', 
             label=f'SED Ensemble Average (t $\leq$ {MAX_TIME_S*1e12} ps)')
    
    r_theoretical = np.linspace(0, 2.5, 1000)
    p_theoretical = compute_theoretical_pr(r_theoretical)
    plt.plot(r_theoretical, p_theoretical, color='red', linewidth=2.5, 
             label=r'QM Ground State $P(r)$')
    
    plt.title(f"Time-Weighted SED Histogram vs. Quantum Mechanics", fontsize=14)
    plt.xlabel(r"Radial Distance $r$ (Å)", fontsize=12)
    plt.ylabel(r"Probability Density $P(r)$", fontsize=12)
    plt.xlim(0, 2.5)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()