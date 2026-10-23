import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Command-Line Argument Setup
if len(sys.argv) < 2:
    print("ERROR: Missing data file.")
    print("Usage: python3 HISTOGRAM.py <name_of_data_file.csv>")
    sys.exit(1)

filename = sys.argv[1]
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, filename)

# 2. Load the C++ data
print(f"Loading data from: {filename}...")
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"ERROR: Could not find the file '{filename}' in the script directory.")
    sys.exit(1)

r_vals = data['r_angstroms'].values
counts = data['counts'].values

# 3. Normalize the histogram
dr = r_vals[1] - r_vals[0]
total_area = np.sum(counts * dr)

if total_area > 0:
    pdf_classical = counts / total_area
else:
    pdf_classical = counts

# 4. Calculate the Theoretical Quantum Ground State (1s)
a0 = 0.529177 
pdf_quantum = (4.0 / (a0**3)) * (r_vals**2) * np.exp(-2.0 * r_vals / a0)

# 5. Plot the results
plt.figure(figsize=(10, 6))
plt.bar(r_vals, pdf_classical, width=dr, color='royalblue', alpha=0.6, label='Classical SED Simulation')
plt.plot(r_vals, pdf_quantum, color='crimson', linewidth=2.5, label='Theoretical QM 1s State')

plt.title('Radial Probability Density P(r) vs r', fontsize=16, fontweight='bold')
plt.xlabel('Radial Distance r (Angstroms)', fontsize=14)
plt.ylabel('Probability Density P(r)', fontsize=14)

# ---> CHANGE X-AXIS LIMITS HERE <---
plt.xlim(0, 3.0) 

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()