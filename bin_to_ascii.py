import sys
import os
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bin_to_ascii.py <binary_file.bin>")
        sys.exit(1)

    input_file = sys.argv[1]
    base_name, _ = os.path.splitext(input_file)
    output_file = f"{base_name}.txt"

    try:
        raw_data = np.fromfile(input_file, dtype=np.float64)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        sys.exit(1)
    
    cols = 8  
    total_elements = len(raw_data)
    
    # Calculate complete rows and slice off the incomplete trailing floats
    full_rows = total_elements // cols
    valid_elements = full_rows * cols
    
    print(f"Found {total_elements} floats. Truncating to {valid_elements} to make {full_rows} complete {cols}-column rows.")
    
    clean_data = raw_data[:valid_elements]
    data = clean_data.reshape(-1, cols)
    
    # Extract time (col 0) and radius (col 1)
    t = data[:, 0]
    r = data[:, 1]
    
    try:
        out_data = np.column_stack((t, r))
        np.savetxt(output_file, out_data, fmt='%.10e', delimiter='\t')
        print(f"Success! Saved {len(t)} rows to {output_file}")
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()