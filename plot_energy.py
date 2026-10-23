import sys
import numpy as np
import matplotlib.pyplot as plt

# 1. Capture the filename from the command line argument
if len(sys.argv) < 2:
    print("Usage: python3 plot_energy.py <binary_file>")
    sys.exit(1)

filename = sys.argv[1]

# 2. Read the raw binary data
# The C++ code writes 8 doubles (64-bit floats) per record:
# {t, r, a, ecc, theta, energy, L_mag, Lz}
try:
    # np.float64 explicitly matches C++ 'double'
    raw_data = np.fromfile(filename, dtype=np.float64) 
    
    # Reshape the flat array into N rows and 8 columns
    data = raw_data.reshape(-1, 8)
except Exception as e:
    print(f"Error reading {filename}: {e}")
    sys.exit(1)

# 3. Extract the relevant columns
time = data[:, 0]
energy = data[:, 5]
L_mag = data[:, 6]
Lz = data[:, 7]

# 4. Create the 3-panel figure, sharing the X-axis (Time)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# --- PANEL 1: Total Energy ---
ax1.plot(time, energy, color='crimson', linewidth=0.5)
ax1.set_ylabel('Total Energy (ergs)', fontsize=12)
ax1.set_title('3D SED Hydrogen: Energy & Angular Momentum', fontsize=14, fontweight='bold')
# Reference line for QM Ground State (-13.6 eV in ergs)
ax1.axhline(-2.179e-11, color='black', linestyle='--', alpha=0.6, label='QM Ground State (-13.6 eV)')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.6)

# --- PANEL 2: Magnitude of Angular Momentum |L| ---
ax2.plot(time, L_mag, color='darkgreen', linewidth=0.5)
ax2.set_ylabel('|L| (erg·s)', fontsize=12)
# Reference line for 1 h-bar
ax2.axhline(1.054e-27, color='black', linestyle='--', alpha=0.6, label='1 ħ')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.6)

# --- PANEL 3: Z-Component of Angular Momentum (Lz) ---
ax3.plot(time, Lz, color='indigo', linewidth=0.5)
ax3.set_ylabel('L_z (erg·s)', fontsize=12)
ax3.set_xlabel('Time (s)', fontsize=12)
# Reference line at zero
ax3.axhline(0.0, color='black', linestyle='-', alpha=0.5)
ax3.grid(True, linestyle='--', alpha=0.6)

# Adjust layout and render
plt.tight_layout()
plt.show()