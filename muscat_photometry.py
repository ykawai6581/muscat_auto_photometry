import os
import glob
import requests
import sys

def target_from_filename():
    notebook_files = glob.glob("yg*.ipynb")
    target = notebook_files[0][10:-13] 
    return target

def obsdates_from_filename():
    notebook_files = glob.glob("yg*.ipynb")
    obsdates = [item[-12:-6] for item in notebook_files]
    return obsdates

def query_radec(target):
    url = f"https://exofop.ipac.caltech.edu/tess/target.php?id={target}&json"
    # Send the GET request
    print(f"Querying exofop for {target}...\n")
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        try:
            data = response.json()  # Parse JSON response
            ra = float(data["coordinates"]["ra"])
            dec = float(data["coordinates"]["dec"])
            #print(json.dumps(data, indent=4))  # Pretty-print JSON
            return ra ,dec
        except ValueError:
            print("Response is not in JSON format:", response.text)
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        sys.exit()