import pandas as pd
import re

def load_par_file(input_file):
    """Load a .par file into a dictionary."""
    params = {}
    with open(input_file, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):  # Ignore empty lines and comments
                key, value = line.split(None, 1)  # Split on first whitespace
                try:
                    params[key] = float(value)  # Convert numerical values to float
                except ValueError:
                    params[key] = value  # Keep as string if conversion fails
    return params

def load_geo_file(input_file):
    """Returns the relevant coefficients needed for transformation of pixels in ref file to object file"""
    with open(input_file, "r") as file:
        lines = file.readlines()

    # Get the last non-empty line
    last_line = lines[-1].strip()
    
    # Convert it into a list of floats
    values = list(map(float, last_line.split()))
    
    # Assign to relevant variable names
    geoparam = {
        "dx": values[0],
        "dy": values[1],
        "a": values[2],
        "b": values[3],
        "c": values[4],
        "d": values[5],
        "rms": values[6],
        "nmatch": int(values[7])  # Convert last value to int
    }

    return geoparam

def parse_obj_file(input_file): #helper function to parse objectfile
    """Parses the .df file to extract metadata and tabular data."""
    metadata = {}
    data_rows = []

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):  # Metadata lines
                key_value = re.match(r"#\s*(.+?)\s*=\s*(.+)", line)
                if key_value:
                    key, value = key_value.groups()
                    metadata[key.strip()] = value.strip()
            elif line and not line.startswith("#"):  # Data rows
                parts = line.split()
                if len(parts) >= 7:  # Ensure it matches expected column structure
                    data_rows.append(list(map(float, parts)))

    col_names = ['ID', 'x', 'y', 'x_int', 'y_int', 'Flux', 'Peak']
    data = pd.DataFrame(data_rows,columns=col_names)
    return metadata, data

import pandas as pd

def parse_dat_file(input_file):
    table_data = []
        
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()  # Strip whitespace once at the start

            if not line:  # Skip empty lines
                continue

            if line.startswith("#"):
                if "ID xcen ycen" in line:
                    continue
            else:
                table_data.append(line.replace("-nan", "nan"))
    
    # Convert to DataFrame
    df = pd.DataFrame([row.split() for row in table_data], 
                      columns=['ID', 'xcen', 'ycen', 'nflux', 'flux', 'err', 
                               'sky', 'sky_sdev', 'SNR', 'nbadpix', 'fwhm', 'peak'])

    # Convert numerical columns to float
    df = df.astype({'ID': int, 'xcen': float, 'ycen': float, 'nflux': float, 
                    'flux': float, 'err': float, 'sky': float, 'sky_sdev': float, 
                    'SNR': float, 'nbadpix': int, 'fwhm': float, 'peak': float})
    
    return df
