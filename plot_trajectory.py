import sys
import numpy as np
import matplotlib.pyplot as plt

# --- NEW: Purge any cached plots from memory ---
plt.close('all')

# 1. Capture the filename from the command line argument
if len(sys.argv) < 2:
    print("Usage: python3 plot_trajectory.py <binary_file>")
    sys.exit(1)

filename = sys.argv[1]

# 2. Read the raw binary data (8 columns now!)
try:
    raw_data = np.fromfile(filename, dtype=np.float64) 
    data = raw_data.reshape(-1, 8)
except Exception as e:
    print(f"Error reading {filename}: {e}")
    sys.exit(1)

# 3. Extract the relevant columns
time = data[:, 0]
# Multiply by 1e8 to convert cm to Angstroms for the plots
r_angstroms = data[:, 1] * 1e8
a_angstroms = data[:, 2] * 1e8
ecc = data[:, 3]
theta = data[:, 4]

# --- NEW: Diagnostic terminal output ---
print(f"Successfully loaded {len(time)} data points.")
if len(time) > 0:
    print(f"Latest simulation time plotted: {time[-1]:.3e} seconds.")

# 4. Create the 4-panel figure
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
fig.suptitle('3D SED Hydrogen: Dynamic Tracking Diagnostic', fontsize=14, fontweight='bold', y=0.98)

# --- PANEL 1: Radius (r) ---
ax1.plot(time, r_angstroms, color='mediumblue', linewidth=0.5)
ax1.set_ylabel('Radius (Å)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)

# --- PANEL 2: Semi-major Axis (a) ---
ax2.plot(time, a_angstroms, color='forestgreen', linewidth=0.5)
ax2.set_ylabel('Semi-major Axis a (Å)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)

# --- PANEL 3: Eccentricity (e) ---
ax3.plot(time, ecc, color='darkorange', linewidth=0.5)
ax3.set_ylabel('Eccentricity e', fontsize=12)
ax3.set_ylim(0, 1.0)
ax3.grid(True, linestyle='--', alpha=0.6)

# --- PANEL 4: Polar Angle (Theta) ---
ax4.plot(time, theta, color='purple', linewidth=0.5)
ax4.set_ylabel('Polar Angle $\\theta$ (rad)', fontsize=12)
ax4.set_xlabel('Time (s)', fontsize=12)
ax4.set_ylim(0, np.pi)
# Reference line for Pi/2 (equator)
ax4.axhline(np.pi / 2, color='black', linestyle='-', alpha=0.3)
ax4.grid(True, linestyle='--', alpha=0.6)

# Adjust layout and render
plt.tight_layout()
plt.show()