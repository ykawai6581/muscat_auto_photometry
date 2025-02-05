import subprocess
import pandas as pd
from io import StringIO

import numpy as np
import os
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import sys
from IPython.display import IFrame

from pathlib import Path
import multiprocessing as mp
from functools import partial
import glob
from concurrent.futures import ProcessPoolExecutor
import time

import LC_funcs as lc
from astropy.table import Table
from astropy.coordinates import Angle
import astropy.units as u
import barycorr
from multiprocessing import Process, Pool, Array, Manager
import math
import matplotlib.patches as patches
import matplotlib.colors as mcolors


import itertools

def get_combinations(x, y, exclude):
    # Generate the range of numbers from x to y
    numbers = list(range(x, y+1))
    
    # Remove the excluded number
    numbers = [num for num in numbers if num != exclude]
    
    # Generate all combinations of the remaining numbers
    combinations = []
    for r in range(1, len(numbers) + 1):  # Start from 1 to include all sizes of combinations
        comb = itertools.combinations(numbers, r)
        combinations.extend([' '.join(map(str, c)) for c in comb])
    
    return combinations

def time_keeper(func):
    """Decorator to time methods."""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)  # Call the original method
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("Done.")
        print(f"Wall time: {minutes} minutes {seconds} seconds")
        return result
    return wrapper

def ra_to_hms(ra_deg):
    ra_hours = ra_deg / 15
    ra_deg_part, ra_min_part = divmod(ra_hours, 1)
    ra_minutes = ra_min_part * 60
    ra_min_part, ra_sec_part = divmod(ra_minutes, 1)
    ra_seconds = ra_sec_part * 60
    return f"{int(ra_deg_part):02d}:{int(ra_min_part):02d}:{ra_seconds:05.2f}"

def dec_to_dms(dec_deg):
    dec_deg_part, dec_min_part = divmod(abs(dec_deg), 1)
    dec_minutes = dec_min_part * 60
    dec_min_part, dec_sec_part = divmod(dec_minutes, 1)
    dec_seconds = dec_sec_part * 60
    sign = "+" if dec_deg >= 0 else "-"
    return f"{sign}{int(dec_deg_part):02d}:{int(dec_min_part):02d}:{dec_seconds:05.2f}"

from muscat_photometry import target_from_filename, obsdates_from_filename, query_radec

os.umask(0o002)
os.nice(19)

ra, dec = query_radec(target_from_filename())

print(f"Running notebook for {target_from_filename()}")
print(f"Ra, Dec: {ra, dec}")
print(f"Available obsdates {obsdates_from_filename()}")

class MuSCAT_PHOTOMETRY:
    def __init__(self,instrument,obsdate,parent=None):
        if not ((instrument is not None and obsdate is not None) or parent is not None):
            raise ValueError("Either both 'instrument' and 'obsdate' or 'parent' must be provided.")

        if parent:
            # Copy attributes from the parent if given
            self.__dict__.update(parent.__dict__)

        instrument_id = {"muscat":1,"muscat2":2,"muscat3":3,"muscat4":4}
        if instrument not in list(instrument_id.keys()):
            print(f"Instrument has to be one of {list(instrument_id.keys())}")
            return

        self.ra, self.dec = query_radec(target_from_filename())

        self.nccd = 3 if instrument == "muscat" else 4
        self.obslog = []
        self.obsdate = obsdate
        self.instrument = instrument
        self.instid = instrument_id[instrument]
        muscat_bands = {
            "muscat" : ["g","r","z"],
            "muscat2" :["g","r","i","z"],
            "muscat3" :["r","i","g","z"],
            "muscat4" :["g","r","i","z"],
        }
        self.bands = muscat_bands[instrument]
        os.chdir('/home/muscat/reduction_afphot/'+instrument)

        for i in range(self.nccd):
            print(f'\n=== CCD{i} ===')
            cmd = f'perl /home/muscat/obslog/show_obslog_summary.pl {instrument} {obsdate} {i}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            obslog_perccd = result.stdout
            print(obslog_perccd)  # Optional: Print to verify
            obslog_perccd = obslog_perccd.lstrip("# ")
            obslog_perccd_df = pd.read_csv(StringIO(obslog_perccd), delim_whitespace=True)

            self.obslog.append(obslog_perccd_df)
        self.obj_names = list(self.obslog[0]['OBJECT'][(self.obslog[0]['OBJECT'] != 'FLAT') & (self.obslog[0]['OBJECT'] != 'DARK')])
        pick_target = input(f"Available object names {[f'{i}|{item}' for i, item in enumerate(self.obj_names)]}")
        print(pick_target)
        self.target = self.obj_names[int(pick_target[0])]
        print(f"Continuing photometry for {self.target}")

    @time_keeper
    def config_flat(self):
        ## Setting configure files for flat

        ## Change the following values
        #======
        self.flat_first_frameIDs = [int(ccd[ccd["OBJECT"] == "FLAT"]["FRAME#1"]) for ccd in self.obslog]  # first frameIDs of flat data
        #flat_first_frameIDs = [1306, 1857, 2414] 
        #======

        for i in range(self.nccd):
            flat_conf_path = f"{self.obsdate}/FLAT/list/flat_ccd{i}.conf"
            #print(flat_conf_path)
            if not os.path.exists(flat_conf_path):
                cmd = f'perl scripts/config_flat.pl {self.obsdate} {i} -set_dir_only'
                subprocess.run(cmd, shell=True, capture_output=True, text=True)

                flat_conf = self.obsdate + f'/FLAT/list/flat_ccd{i}.conf'
                print(flat_conf)
                text = f'flat {self.flat_first_frameIDs[i]} {self.flat_first_frameIDs[i]+49}\nflat_dark {self.flat_first_frameIDs[i]+50} {self.flat_first_frameIDs[i]+54}'
                with open(flat_conf, mode='w') as f:
                    f.write(text)
                result = subprocess.run(['cat', flat_conf], capture_output=True, text=True)
                print(result.stdout)
                print('\n')
            else:
                print(f"config file already exisits under /FLAT/list/flat_ccd{i}.conf")

    @time_keeper
    def config_object(self):
        ## Setting configure files for object
        exposure = [float(ccd["EXPTIME(s)"][ccd["OBJECT"] == self.target]) for ccd in self.obslog]  # exposure times (sec) for object
        for i in range(self.nccd):
            obj_conf_path = f"{self.obsdate}/{self.target}_{i}/list/object_ccd{i}.conf"
            #print(obj_conf_path)
            if not os.path.exists(obj_conf_path):
                exp=exposure[i]
                cmd = f'perl scripts/config_object.pl {self.obsdate} {self.target} {i} -auto_obj -auto_dark {exp}'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                print(result.stdout)
            else:
                print(f"config file already exisits under /{self.target}_{i}/list/ as object_ccd{i}.conf")

    @time_keeper
    def reduce_flat(self):
        ## Reducing FLAT images 
        for i in range(self.nccd):
            flat_path = f"{self.obsdate}/FLAT/flat/flat_ccd{i}.fits"
            #print(flat_path)
            if not os.path.exists(flat_path):
                print(f'>> Reducing FLAT images of CCD{i} ... (it may take tens of seconds)')
                cmd = f"perl scripts/auto_mkflat.pl {self.obsdate} {i} > /dev/null"
                subprocess.run(cmd, shell=True, capture_output=True, text=True)
            else:
                print(f"flat file already exisits under /FLAT/flat/ as flat_ccd{i}.fits")

    @time_keeper
    ## Reducing Object images 
    def run_auto_mkdf(self):
        for i in range(self.nccd):
            df_directory = f'{self.obsdate}/{self.target}_{i}/df'
            frame_range = self.obslog[i][self.obslog[i]["OBJECT"] == self.target]
            first_frame = int(frame_range["FRAME#1"].iloc[0])
            last_frame = int(frame_range["FRAME#2"].iloc[0])
            print(f'CCD{i}: Reducing frames {first_frame}~{last_frame} ...')
            missing_files = [f"MCT{self.instid}{i}_{self.obsdate}{frame:04d}.df.fits" for frame in range(first_frame, last_frame+1) if not os.path.exists(os.path.join(df_directory, f"MCT{self.instid}{i}_{self.obsdate}{frame:04d}.df.fits"))]

            if missing_files:
                cmd = f"perl scripts/auto_mkdf.pl {self.obsdate} {self.target} {i} > /dev/null"
                subprocess.run(cmd, shell=True, capture_output=True, text=True)
                print(f'Completed auto_mkdf.pl for CCD{i}')
            else:
                print(f"df file already exisits under /{self.target}_{i}/df/")

    @time_keeper
    def create_ref(self, ccd=0,refid_delta=0):
        ## Creating a reference image

        ## Change the folloiwng value if necessary
        #======
        #ref_ccd=1      # CCD number for the reference image
        ref_ccd=ccd
        refid= int(self.obslog[ref_ccd][self.obslog[ref_ccd]["OBJECT"] == self.target]["FRAME#1"])#if you are okay with setting the first frame as reference
        refid+=refid_delta
        #======

        ref_exists = all([os.path.exists(f"{self.obsdate}/{self.target}_{i}/list/ref.lst") for i in range(self.nccd)])
        if not ref_exists:
            cmd = f"perl scripts/make_reference.pl {self.obsdate} {self.target} --ccd={ref_ccd} --refid={refid}"
            subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            with open(Path(f"{self.obsdate}/{self.target}_0/list/ref.lst"), 'r') as f:
                ref_file = f.read()
                print(f'Ref file:\n {ref_file} exists.')

    def show_reference(self, rad=10):
        ## Showing reference image

        target_dir = self.obsdate + '/' + self.target
        ref_list_file = target_dir + '/list/ref.lst'
        with open(ref_list_file) as f:
            refframe = f.readline()

        refframe = refframe.replace('\n','')
        print('reference frame:', refframe)

        ref_obj_file = target_dir + '/objects/' + refframe + '.objects'
        refxy = np.genfromtxt(ref_obj_file, delimiter=13, usecols=(1,2))


        ref_fits = target_dir + '/reference/ref-' + refframe + '.fits'
        hdulist = fits.open(ref_fits)
        data = hdulist[0].data
        #dataf = data.astype(np.float64)

        norm = ImageNormalize(data, interval=ZScaleInterval())
        plt.figure(figsize=(10,10))
        ax=plt.subplot(1,1,1)
        plt.imshow(data, origin='lower', norm=norm)
        rad=rad
        for i in range(len(refxy)):
            circ = plt.Circle(refxy[i], rad, color='red', fill=False)
            ax.add_patch(circ)
            plt.text(refxy[i][0]+rad/2., refxy[i][1]+rad/2., str(i+1), fontsize=20, color='yellow')

    ## Performing aperture photometry
    @time_keeper
    def run_apphot(self, nstars=None, rad1=None, rad2=None, drad=None, method="mapping"):

        # Assume the same available radius for all CCDs
        apphot_base = f"{self.obsdate}/{self.target}_0/apphot_{method}"
        available_rad = sorted([float(p.name[3:]) for p in Path(apphot_base).glob("*/")]) if Path(apphot_base).exists() else []

        if available_rad and rad1==None and rad2==None and drad==None:
            self.rad1, self.rad2, self.method = float(available_rad[0]), float(available_rad[-1]), method
            random_frame = self.obslog[0][self.obslog[0]["OBJECT"] == self.target]
            random_frame = int(random_frame["FRAME#1"].iloc[0])
            df, meta = self.read_photometry(ccd=0, rad=self.rad1, frame=random_frame, add_metadata=True)
            self.nstars = meta['nstars'] 
            print(f"Previously attempted photometry with {available_rad}, nstars={self.nstars}")
            return
        else:
            self.rad1, self.rad2, self.drad, self.method, self.nstars = float(rad1), float(rad2), float(drad), method, int(nstars)

        rads = np.arange(self.rad1, self.rad2 + 1, self.drad)
        print(f"Performing photometry for radius: {rads}")

        # Check for missing photometry files
        missing, missing_files_per_ccd = self.check_missing_photometry(rads)

        if not missing:
            print(f"Photometry is already available for radius: {available_rad}")
            return

        # Determine script to use
        script = f"scripts/auto_apphot_{method}.pl"

        # Run photometry for missing files
        self.run_photometry_if_missing(script, nstars, rads, missing_files_per_ccd)

    def check_missing_photometry(self, rads):
        """Checks for missing photometry files and returns a dictionary of missing files per CCD."""
        missing = False
        missing_files_per_ccd = {}

        for i in range(self.nccd):
            appphot_directory = f"{self.obsdate}/{self.target}_{i}/apphot_{self.method}"
            frame_range = self.obslog[i][self.obslog[i]["OBJECT"] == self.target]
            first_frame = int(frame_range["FRAME#1"].iloc[0])
            last_frame = int(frame_range["FRAME#2"].iloc[0])
            
            missing_files = [
                f"{appphot_directory}/rad{rad}/MCT{self.instid}{i}_{self.obsdate}{frame:04d}.dat"
                for rad in rads
                for frame in range(first_frame, last_frame+1)
                if not os.path.exists(f"{appphot_directory}/rad{rad}/MCT{self.instid}{i}_{self.obsdate}{frame:04d}.dat")
            ]

            if missing_files:
                missing = True
                missing_files_per_ccd[i] = missing_files
                print(f"CCD {i}: Missing files for some radii: {missing_files[:5]}{'...' if len(missing_files) > 5 else ''}")

        return missing, missing_files_per_ccd

    def run_photometry_if_missing(self, script, nstars, rads, missing_files_per_ccd):
        """Runs photometry for CCDs where files are missing."""
        for i, missing_files in missing_files_per_ccd.items():
            for rad in rads:
                if any(f"rad{rad}" in f for f in missing_files):  # Only run if files for this radius are missing
                    cmd = f"perl {script} {self.obsdate} {self.target} {i} {nstars} {rad} {rad} {self.drad} > /dev/null"
                    subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    print(f"Completed aperture photometry for CCD={i}, rad={rad}")
                else:
                    print(f"Photometry already available for CCD={i}, rad={rad}")

    def read_photometry(self, ccd, rad, frame, add_metadata=False):
        filepath = f"{self.obsdate}/{self.target}_{ccd}/apphot_{self.method}/rad{str(rad)}/MCT{self.instid}{ccd}_{self.obsdate}{frame:04d}.dat"
        metadata = {}
        table_started = False
        table_data = []
        
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    if 'ID xcen ycen' in line:
                        table_started = True
                        continue
                    if add_metadata and not table_started:
                        line = line.strip('# \n')
                        if '=' in line:
                            key, value = line.split('=')
                            metadata[key.strip()] = value.strip()
                        elif line.strip():
                            parts = line.split()
                            if len(parts) >= 2:
                                key = parts[0]
                                value = ' '.join(parts[1:])
                                metadata[key.strip()] = value.strip()
                else:
                    # Replace "-nan" with "nan" in the data
                    cleaned_line = line.replace("-nan", "nan")
                    table_data.append(cleaned_line.strip())
        
        # Convert to DataFrame
        df = pd.DataFrame([row.split() for row in table_data], 
                        columns=['ID', 'xcen', 'ycen', 'nflux', 'flux', 'err', 
                                'sky', 'sky_sdev', 'SNR', 'nbadpix', 'fwhm', 'peak'])
        
        # Convert numeric columns with proper NaN handling
        numeric_cols = df.columns.difference(['filename', 'ccd'])
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' will convert invalid parsing to NaN
        
        # Add file information
        df['filename'] = Path(filepath).name
        df['ccd'] = ccd
        
        # Convert numeric metadata values
        if add_metadata:
            for key, value in metadata.items():
                try:
                    # Handle -nan in metadata as well
                    if value.strip().lower() == "-nan":
                        metadata[key] = np.nan
                    else:
                        metadata[key] = float(value)
                except (ValueError, AttributeError):
                    # Keep as string if conversion fails
                    pass
                df[key] = metadata[key]
        
        return df, metadata if add_metadata else df#[['ID', 'peak']]

    def process_single_ccd(self, ccd, rad):
        """
        Process photometry data for a single CCD with metadata.
        """
        frame_range = self.obslog[ccd][self.obslog[ccd]["OBJECT"] == self.target]
        first_frame = int(frame_range["FRAME#1"].iloc[0])
        last_frame = int(frame_range["FRAME#2"].iloc[0])
        
        all_frames = []
        
        for frame in range(first_frame, last_frame+1):
            result = self.read_photometry(ccd=ccd, rad=rad, frame=frame, add_metadata=False)
            #print(frame)
            #print(result)
            #print(type(result))
            if result is not None:
                df = result[0]
                df['frame'] = frame
                all_frames.append(df)

        if all_frames:
            combined_df = pd.concat(all_frames, ignore_index=True)
            
            # Add CCD identifier
            combined_df['ccd'] = ccd
            
            return combined_df  
        else:
            return None

        
    def read_photometry_parallel(self, rad, num_processes=4):
        """
        Read photometry data for multiple CCDs in parallel.
        
        Parameters:
        rad (int/float): Radius value
        num_processes (int): Number of parallel processes to use (default=None, uses CPU count)
        
        Returns:
        tuple: (combined_data_df, combined_metadata_df)
        """
        
        # Create partial function with fixed parameters
        process_func = partial(self.process_single_ccd, rad=rad)
        
        # Process CCDs in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_func, list(range(self.nccd)))
        
        # Filter out None results and separate data and metadata
        valid_results = [result for result in results if result is not None]
        
        return valid_results

    @time_keeper
    def check_saturation(self, rad):
        self.saturation_cids = []
        print(f'>> Checking for saturation ... (it may take a few seconds)')
        df = self.read_photometry_parallel(rad=rad)
        # Count the number of rows where peak > 60000 for this star ID
        for i in range(self.nccd):
            saturation_cids_per_ccd = []
            for star_id in range(int(self.nstars)):
                count_above_threshold = (df[i][df[i]["ID"] == star_id]["peak"] > 60000).sum()
                percentage_above_threshold = count_above_threshold / len(df[i][df[i]["ID"] == star_id]) * 100
                
                # If more than 5% of the rows have a peak > 60000, add this star ID to the list
                if percentage_above_threshold > 5:
                    saturation_cids_per_ccd.append(star_id)
            print(f'  >> CCD {i}: Done.')
            self.saturation_cids.append(saturation_cids_per_ccd)

        for i in range(self.nccd):
            print(f"WARNING: Over 5 percent of frames are saturated for cIDS {self.saturation_cids[i]} in CCD {i}")

    def select_comparison(self, tid, nstars=5):
        self.tid = tid
        print(f"{self.target} | TID = {tid}")
        self.check_saturation(self.rad1)
        self.cids_list = []
        for saturation_cid in self.saturation_cids:
            if saturation_cid:
                brightest_star = max(saturation_cid) + 1
            else:
                brightest_star = 1
            cids = get_combinations(brightest_star, brightest_star + nstars - 1, tid)
            self.cids_list.append(cids)

    @time_keeper
    def create_photometry(self, given_cids=None):
        if given_cids:
            self.cids_list = given_cids
        script_path = "/home/muscat/reduction_afphot/tools/afphot/script/mklc_flux_collect_csv.pl"
        print(">> Creating photometry file for")
        print(f"| Target = {self.target} | TID = {self.tid} | r1={self.rad1} r2={self.rad2} dr={self.drad} | (it may take minutes)")
        for i in range(self.nccd):
            print(f'>> CCD{i}')
            for cid in self.cids_list[i]:
                obj_dir = f"/home/muscat/reduction_afphot/{self.instrument}/{self.obsdate}/{self.target}"
                os.chdir(Path(f"/home/muscat/reduction_afphot/{self.instrument}/{self.obsdate}/{self.target}_{i}")) 
                cmd = f"perl {script_path} -apdir apphot_{self.method} -list list/object_ccd{i}.lst -r1 {int(self.rad1)} -r2 {int(self.rad2)} -dr {self.drad} -tid {self.tid} -cids {cid} -obj {self.target} -inst {self.instrument} -band {self.bands[i]} -date {self.obsdate}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True) #this command requires the cids to be separated by space
                #print(cmd)
                #print(result.stdout)

                outfile = f"lcf_{self.instrument}_{self.bands[i]}_{self.target}_{self.obsdate}_t{self.tid}_c{cid.replace(' ','')}_r{int(self.rad1)}-{int(self.rad2)}.csv" # file name radius must be int
                #print(os.getcwd())
                #print(f"Created {outfile}")
                outfile_path = os.path.join(os.getcwd(),f"apphot_{self.method}", outfile)
                #print(outfile_path)

                if os.path.isfile(outfile_path):
                    #outfile2 = f"{instdir}/{date}/{obj}/lcf_{inst}_{bands[i]}_{obj}_{date}_t{tid}_c{suffix}_r{rad1}-{rad2}.csv"
                    subprocess.run(f"mv {outfile_path} {obj_dir}/{outfile}", shell=True)
                    print(f"## >> Created photometry for cIDs:{cid}")
                else:
                    print(f"## >> Failed to create photometry for cIDs:{cid}")

        os.chdir(Path(f"/home/muscat/reduction_afphot/{self.instrument}"))


class MuSCAT_PHOTOMETRY_OPTIMIZATION:
    def __init__(self, muscat_photometry):
        # Copy all attributes from the existing instance
        self.__dict__.update(muscat_photometry.__dict__)
        self.ap = np.arange(self.rad1, self.rad2+self.drad, self.drad)
        self.cids_list_opt = [[cid.replace(" ", "") for cid in cids] for cids in self.cids_list] #optphot takes cids with no space
        print('available aperture radii: ', self.ap)
        self.bands = ["g","r","i","z"]
        self.mask = [[],[],[],[]]

        self.min_rms_idx_list = []
        self.phot=[]
        phot_dir = f"/home/muscat/reduction_afphot/{self.instrument}/{self.obsdate}/{self.target}"

        for i in range(self.nccd):
            self.phot.append([])
            for cid in self.cids_list_opt[i]:#self.cids_list_opt is only needed to access the files here
                infile = f'{phot_dir}/lcf_{self.instrument}_{self.bands[i]}_{self.target}_{self.obsdate}_t{self.tid}_c{cid}_r{str(int(self.rad1))}-{str(int(self.rad2))}.csv'
                self.phot[i].append(Table.read(infile))

    def add_mask_per_ccd(self, key, ccd, lower=None, upper=None):
        """Applies a mask to all elements in the dataset for a given CCD."""
        mask_ccd = []  # Initialize an empty list for masks

        for j in range(len(self.phot[ccd])):  # Loop over j (e.g., different sources)
            condition = np.ones_like(self.phot[ccd][j][key], dtype=bool)  # Start with all True
            
            if lower is not None:
                condition &= (self.phot[ccd][j][key] > lower[ccd])  # Apply lower bound
                
            if upper is not None:
                condition &= (self.phot[ccd][j][key] < upper[ccd])  # Apply upper bound
            
            mask_ccd.append(condition)  # Store mask for this j
        self.mask[ccd].append(mask_ccd)
        print(f"Added mask to {key} for CCD{ccd}")

    def add_mask(self, key, lower=None, upper=None):
        """Applies a mask to all elements in self.phot, handling lists of upper/lower bounds."""
        
        for i in range(len(self.phot)):  # Loop over CCDs
            self.mask[i] = []  # Ensure mask[i] is a clean list
            
            for j in range(len(self.phot[i])):  # Loop over sources, stars, or apertures
                condition = np.ones_like(self.phot[i][j][key], dtype=bool)  # Start with all True

                if lower is not None:
                    condition &= self.phot[i][j][key] > (lower[i] if isinstance(lower, list) else lower)
                    
                if upper is not None:
                    condition &= self.phot[i][j][key] < (upper[i] if isinstance(upper, list) else upper)
                
                self.mask[i].append(condition)  # Directly store condition, keeping shape (4, 15)
        print(f"Added mask to {key}")

    @time_keeper
    def outlier_cut(self, sigma_cut=3, order=2, plot=True):
        self.index = [[] for _ in range(self.nccd)]  # Pre-allocate index storage
        fig, axes = plt.subplots(self.nccd, 2, figsize=(14, 4 * self.nccd))  # 2 columns per CCD
        print(f">> Fitting with polynomials (order = {order}) and cutting {sigma_cut} sigma outliers ...  (it may take a few minutes)")

        for i in range(self.nccd):
            print(f"Computing outliers for CCD{i}")
            n_cids = len(self.cids_list[i])
            n_ap = len(self.ap)

            ndata_diff = np.zeros((n_cids, n_ap))
            rms = np.zeros((n_cids, n_ap))
            self.index[i] = [[] for _ in range(n_cids)]

            for j in range(n_cids):
                phot_j = self.phot[i][j]
                exptime = phot_j['exptime']
                gjd_vals = phot_j['GJD-2450000']
                mask = self.mask[i][j] if (i < len(self.mask) and j < len(self.mask[i])) else np.ones_like(gjd_vals, dtype=bool)

                fcomp_keys = [f'flux_comp(r={self.ap[k]:.1f})' for k in range(n_ap)]
                fcomp_data = np.array([phot_j[fk] for fk in fcomp_keys])

                raw_norm = (fcomp_data / exptime) / np.median(fcomp_data / exptime, axis=1, keepdims=True)
                ndata_init = fcomp_data.shape[1]

                ye = np.sqrt(fcomp_data[:, mask]) / exptime[mask] / np.median(fcomp_data / exptime, axis=1, keepdims=True)

                for k in range(n_ap):
                    if len(ye[k]) > 0:
                        p, tcut, ycut, yecut = lc.outcut_polyfit(gjd_vals[mask], raw_norm[k][mask], ye[k], order, sigma_cut)
                        self.index[i][j].append(np.isin(gjd_vals, tcut))
                        ndata_final = len(tcut)
                    else:
                        self.index[i][j].append(np.zeros_like(gjd_vals, dtype=bool))
                        ndata_final = 0

                    ndata_diff[j, k] = ndata_final - ndata_init

                    if len(ycut) > 1:
                        diff = np.diff(ycut)
                        rms[j, k] = np.std(diff) if np.std(diff) > 0 else np.inf
                    else:
                        rms[j, k] = np.inf

            self.min_rms = np.argmin(rms, axis=None)
            min_rms_idx = np.unravel_index(self.min_rms, rms.shape)
            self.min_rms_idx_list.append(min_rms_idx)

            if plot:
                # Normalize color scales for consistency across CCDs
                norm_diff = mcolors.Normalize(vmin=np.min(ndata_diff), vmax=np.max(ndata_diff))
                norm_rms = mcolors.Normalize(vmin=np.min(rms[rms != np.inf]), vmax=np.max(rms[rms != np.inf]))

                # **Left Plot: Number of cut data points**
                im1 = axes[i, 0].imshow(ndata_diff, cmap="coolwarm", aspect="auto", norm=norm_diff)
                axes[i, 0].set_title(f"CCD {i} - Cut Data Points")
                axes[i, 0].set_xticks(range(n_ap))
                axes[i, 0].set_xticklabels([f"{self.ap[k]:.1f}" for k in range(n_ap)])
                axes[i, 0].set_yticks(range(n_cids))
                axes[i, 0].set_yticklabels(self.cids_list[i])
                fig.colorbar(im1, ax=axes[i, 0], label="Number of Cut Data Points")

                # **Right Plot: RMS of flux differences**
                im2 = axes[i, 1].imshow(rms, cmap="cividis", aspect="auto", norm=norm_rms)
                axes[i, 1].set_title(f"CCD {i} - RMS")
                axes[i, 1].set_xticks(range(n_ap))
                axes[i, 1].set_xticklabels([f"{self.ap[k]:.1f}" for k in range(n_ap)])
                axes[i, 1].set_yticks(range(n_cids))
                axes[i, 1].set_yticklabels(self.cids_list[i])
                fig.colorbar(im2, ax=axes[i, 1], label="RMS")

                # **Highlight the min RMS cell with a white square**
                j_min, k_min = min_rms_idx
                rect = patches.Rectangle((k_min - 0.5, j_min - 0.5), 1, 1, linewidth=3, edgecolor='white', facecolor='none')
                axes[i, 1].add_patch(rect)

        if plot:
            print("Plotting results")
            plt.tight_layout()
            plt.show()

        self.cIDs_best     = [self.cids_list[i][item[0]] for i, item in enumerate(self.min_rms_idx_list)]
        self.cIDs_best_idx = [item[0] for item in self.min_rms_idx_list]
        self.ap_best       = [self.ap[item[1]] for item in self.min_rms_idx_list]
        self.ap_best_idx   = [item[1] for item in self.min_rms_idx_list]

    def iterate_optimization(self):
        min_rms = np.inf
        min_rms_list = [np.inf,self.min_rms]
        best_optimization = None

        while min_rms_list[-1] < min_rms_list[-2]: #whle the rms keeps improving
            print(f"Returning to photometry for aperture optimization... (Iteration: {len(min_rms_list)-1}")
            drad = 1
            if any(idx in {self.ap[0]} for idx in self.ap_best): #if the lowest rms is the smallest aperture 
                rad1 = self.ap[0] - drad
                rad2 = rad1
            elif any(idx in {self.ap[-1]} for idx in self.ap_best): #if the lowest rms is the largest aperture 
                rad1 = self.ap[0] + drad
                rad2 = rad1
            else:
                if self.drad == 1:
                    print("Already optimal")
                    return
                else:
                    rad1 = min(self.ap_best) - drad#the smallest aperture
                    rad2 = max(self.ap_best) + drad #the largest aperture

            photometry = MuSCAT_PHOTOMETRY(parent=self)
            photometry.run_apphot(nstars=self.nstars, rad1=rad1, rad2=rad2, drad=drad, method="mapping")
            photometry.cids_list = self.cIDs_best
            photometry.create_photometry()

            optimization = MuSCAT_PHOTOMETRY_OPTIMIZATION(photometry)
            optimization.mask = self.mask #adds the same mask
            optimization.outlier_cut(plot=False)
            min_rms = optimization.min_rms
            min_rms_list.append(min_rms)
            print(f"Minimum rms: {min_rms_list[-2]} -> {min_rms_list[-1]}")
            if min_rms_list[-1] < min_rms_list[-2]:  
                best_optimization = optimization

        return best_optimization

    def plot_lc(self):
        binsize = 300/86400.
        tbin=[]
        ybin=[]
        yebin=[]
        band_names = ['g-band','r-band','i-band','z$_s$-band']
        colors = ["blue","green","orange","red"]

        offset=np.array((0,-0.1,-0.2,-0.3))

        plt.figure(figsize=(10,12))
        plt.rcParams['font.size']=18
        for i in range(self.nccd):
            
            f_key = 'flux(r=' + '{0:.1f})'.format(self.ap_best[i])
            e_key = 'flux(r=' + '{0:.1f})'.format(self.ap_best[i])

            phot_per_ccd = self.phot[i][self.cIDs_best_idx[i]] 
            best_idx = self.index[i][self.cIDs_best_idx[i]][self.ap_best_idx[i]]  # Store the best index mask

            lc_time = phot_per_ccd['GJD-2450000'][best_idx]  
            t0 = np.min(lc_time)

            best_flux = phot_per_ccd[f_key][best_idx]
            best_flux_err = phot_per_ccd[e_key][best_idx]

            tbin_tmp, ybin_tmp, yebin_tmp = lc.binning_equal_interval(lc_time, best_flux, best_flux_err, binsize, t0)
            
            tbin.append(tbin_tmp)
            ybin.append(ybin_tmp)
            yebin.append(yebin_tmp)
            
            plt.plot(lc_time,best_flux+offset[i],'.k',alpha=0.3)
        #    plt.ylim(0.985,1.015)
            plt.plot(tbin[i], ybin[i]+offset[i],'o',color=colors[i], markersize=8)
            tx=tbin[i][0]
            ty=1.005+offset[i]
            plt.text(tx,ty,band_names[i],color=colors[i])
        #plt.xlabel('')
        plt.title(self.target)
        plt.xlabel('JD-2450000')
        plt.ylabel('Relative flux')
        outfile = '{0}_{1}.png'.format(self.target,self.obsdate)
        plt.savefig(outfile,bbox_inches='tight',pad_inches=0.1)
        plt.show()

#cIDs_best = np.array((2,0,2,2)) 
#ap_best = np.array((0,0,0,1))

    '''
    @time_keeper
    def outlier_cut(self, sigma_cut=3, order=2):
        index = [[] for _ in range(self.nccd)]  # Pre-allocate index storage
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 subplots for 4 CCDs
        axes = axes.flatten()  # Flatten to 1D array for easy access
        print(f">> Fitting with polynomials (order = {order}) and cutting {sigma_cut} sigma outliers...")
        for i in range(self.nccd):
            print(f"Computing outliers for CCD{i} (it may take a few minutes)")
            n_cids = len(self.cids_list[i])
            n_ap = len(self.ap)
            
            ndata_diff = np.zeros((n_cids, n_ap))  # Initialize grid for (j, k)
            rms        = np.zeros((n_cids, n_ap))  # Initialize grid for (j, k)
            index[i] = [[] for _ in range(n_cids)]  # Pre-allocate storage for index[i]

            for j in range(n_cids):
                mask = self.mask[i][j]  # Extract precomputed mask
                phot_j = self.phot[i][j]  # Shortcut to reduce indexing operations
                
                # Precompute common values
                exptime = phot_j['exptime']
                gjd_vals = phot_j['GJD-2450000']

                fcomp_keys = [f'flux_comp(r={self.ap[k]:.1f})' for k in range(n_ap)]
                fcomp_data = np.array([phot_j[fk] for fk in fcomp_keys])  # Shape (n_ap, n_data)

                # Normalize raw flux
                raw_norm = (fcomp_data / exptime) / np.median(fcomp_data / exptime, axis=1, keepdims=True)
                ndata_init = fcomp_data.shape[1]  # Number of time points

                if np.any(mask):
                    ye = np.sqrt(fcomp_data[:, mask]) / exptime[mask] / np.median(fcomp_data / exptime, axis=1, keepdims=True)
                    
                    for k in range(n_ap):  # Only loop over apertures, reducing total iterations
                        if len(ye[k]) > 0:
                            p, tcut, ycut, yecut = lc.outcut_polyfit(gjd_vals[mask], raw_norm[k][mask], ye[k], order, sigma_cut)
                            index[i][j].append(np.isin(gjd_vals, tcut))
                            ndata_final = len(tcut)
                        else:
                            index[i][j].append(np.zeros_like(gjd_vals, dtype=bool))
                            ndata_final = 0
                        
                        ndata_diff[j, k] = ndata_final - ndata_init  # Store difference in grid
                        ycut_temp1 = ycut[1:]
                        ycut_temp2 = ycut[:-1]
                        diff = ycut_temp1 - ycut_temp2
                        if(np.std(diff)>0):
                            rms[j, k] = np.std(diff)  # store rms of difference in consecutive fluxes 
                        else:
                            rms[j, k] = np.inf  # np.inf for invalid results
                else:
                    index[i][j] = [np.zeros_like(gjd_vals, dtype=bool)] * n_ap  # If no mask, store empty arrays

            # **Plot heatmap for this CCD**
            ax = axes[i]  # Select subplot
            im = ax.imshow(ndata_diff, aspect='auto', cmap='coolwarm', origin='lower')
            ax.set_xlabel("Aperture Radius")
            ax.set_ylabel("cIDs")
            ax.set_title(f"CCD {i}")
            
            # Add colorbar to each subplot
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Number of cut data points")

        plt.tight_layout()
        plt.show()
    '''

    '''
    def outlier_cut(self,sigma_cut=3,order=2):
        index = []
        #print('band, cIDs, apID, ndata_init, ndata_final') #ccd, rad, cidの3次元ある
        for i in range(self.nccd):
            index.append([]) 
            for j in range(len(self.cids_list[i])):
                index[i].append([])
                for k in range(len(self.ap)):
                    
                    fcomp_key = f'flux_comp(r={self.ap[k]:.1f})'
                    f_key = f'flux(r={self.ap[k]:.1f})'
                    e_key = f'flux(r={self.ap[k]:.1f})'
                    
                    raw_norm = (self.phot[i][j][fcomp_key]/self.phot[i][j]['exptime']) / np.median(self.phot[i][j][fcomp_key]/self.phot[i][j]['exptime'])
                    
                    mask = self.mask[i][j]
                        
                    fcomp_median = np.median(self.phot[i][j][fcomp_key]/self.phot[i][j]['exptime'])
                    ye = np.sqrt(self.phot[i][j][fcomp_key][mask])/self.phot[i][j]['exptime'][mask]/fcomp_median #shot noiseを載せるためにexptimeで割っている
                    
                    ndata_final = len(ye)
                                
                    if(len(ye)>0):
                        p, tcut, ycut, yecut\
                            = lc.outcut_polyfit(self.phot[i][j]['GJD-2450000'][mask], raw_norm[mask], ye, order, sigma_cut) #chi square を最小化している
                        index[i][j].append(np.isin(self.phot[i][j]['GJD-2450000'], tcut))
                        ndata_final = len(tcut)
                    
                    else:
                        index[i][j].append(np.isin(self.phot[i][j]['GJD-2450000'], np.empty(0)))
                    

                    #print(i, j, k, len(self.phot[i][j]), ndata_final)




    def preview_photometry():

    
j=2 # cIDs
k=0 # ap_rad

peak_cut = [60000, 60000, 60000, 60000]
amass_cut = 2.0
raw_cut = np.array((0.9, 0.9, 0.9, 0.9))

jd_cut_min = np.min(phot[i][j]['GJD-2450000'])
jd_cut_max = np.max(phot[i][j]['GJD-2450000'])
#jd_cut_min = 10028.41
#jd_exclude_min = 9032.514
#jd_exclude_max = 9032.5375
#dx_max = 10
#dy_max = 10
fwhm_max = np.array((20, 20, 20, 20))

fcomp_key = 'flux_comp(r=' + '{0:.1f})'.format(ap[k])
f_key = 'flux(r=' + '{0:.1f})'.format(ap[k])
e_key = 'flux(r=' + '{0:.1f})'.format(ap[k])

fig, ax = plt.subplots(6,4, figsize=(16,20))
for i in range(nband):

    raw_norm = phot[i][j][fcomp_key]/phot[i][j]['exptime'] / np.median(phot[i][j][fcomp_key]/phot[i][j]['exptime'])
    
    mask = (raw_norm > raw_cut[i])\
            & (phot[i][j]['peak(ADU)']<peak_cut[i])\
            & (phot[i][j]['GJD-2450000'] > jd_cut_min) & (phot[i][j]['GJD-2450000'] < jd_cut_max)\
            # & (phot[i][j]['airmass'] < amass_cut)\
            # & (abs(phot[i][j]['dx(pix)']) < dx_max)\
            # & (abs(phot[i][j]['dy(pix)']) < dy_max)\
            # & (abs(phot[i][j]['fwhm(pix)']) < fwhm_max[i])\
    
    ax[0][i].plot(phot[i][j]['GJD-2450000'][mask],raw_norm[mask],'.')
    ax[1][i].plot(phot[i][j]['GJD-2450000'][mask],phot[i][j]['airmass'][mask],'.')
    ax[2][i].plot(phot[i][j]['GJD-2450000'][mask],phot[i][j]['dx(pix)'][mask],'.')
    ax[3][i].plot(phot[i][j]['GJD-2450000'][mask],phot[i][j]['dy(pix)'][mask],'.')
    ax[4][i].plot(phot[i][j]['GJD-2450000'][mask],phot[i][j]['fwhm(pix)'][mask],'.')
    ax[5][i].plot(phot[i][j]['GJD-2450000'][mask],phot[i][j]['peak(ADU)'][mask],'.')
    ax[0][0].set_ylabel('Relative flux')
    ax[1][0].set_ylabel('Airmass')
    ax[2][0].set_ylabel('dX')
    ax[3][0].set_ylabel('dY')
    ax[4][0].set_ylabel('FWHM')
    ax[5][0].set_ylabel('Peak')
    
    
plt.show()
    
'''