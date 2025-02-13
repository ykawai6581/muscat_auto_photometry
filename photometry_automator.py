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
from astropy import wcs
from astropy.io import fits
import barycorr
from joblib import Parallel, delayed
from multiprocessing import Process, Pool, Array, Manager
import math
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import re

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

def run_photometry(script, obsdate, target, ccd, nstars, rad, drad):
    """Runs the photometry script for a given CCD and radius."""
    cmd = f"perl {script} {obsdate} {target} {ccd} {nstars} {rad} {rad} {drad} > /dev/null"
    subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"Completed aperture photometry for CCD={ccd}, rad={rad}")

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
    def __init__(self,instrument=None,obsdate=None,parent=None):
        if not ((instrument is not None and obsdate is not None) or parent is not None):
            raise ValueError("Either both 'instrument' and 'obsdate' or 'parent' must be provided.")
        
        if parent:
            # Copy attributes from the parent if given
            self.__dict__.update(parent.__dict__)
        else:

            instrument_id = {"muscat":1,"muscat2":2,"muscat3":3,"muscat4":4}

            self.instrument = instrument

            if self.instrument not in list(instrument_id.keys()):
                print(f"Instrument has to be one of {list(instrument_id.keys())}")
                return
            
            self.ra, self.dec = query_radec(target_from_filename())

            self.nccd = 3 if self.instrument == "muscat" else 4
            self.obsdate = obsdate
            self.obslog = []
            self.instid = instrument_id[self.instrument]
            muscat_bands = {
                "muscat" : ["g","r","z"],
                "muscat2" :["g","r","i","z"],
                "muscat3" :["r","i","g","z"],
                "muscat4" :["g","r","i","z"],
            }
            self.bands = muscat_bands[self.instrument]
            os.chdir('/home/muscat/reduction_afphot/'+self.instrument)

            for i in range(self.nccd):
                print(f'\n=== CCD{i} ===')
                cmd = f'perl /home/muscat/obslog/show_obslog_summary.pl {self.instrument} {obsdate} {i}'
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

    def find_tid(self):
        with open(Path(f"{self.obsdate}/{self.target}/list/ref.lst"), 'r') as f:
            ref_file = f.read()

        ref_frame = ref_file.replace('\n','')
        ref_ccd = ref_frame[4]
        ref_file_dir = f"{self.obsdate}/{self.target}_{ref_ccd}"
        ref_file = f"/df/{ref_file_dir}/{ref_frame}.df.fits"
        pixscale = [0.358, 0.435, 0.27,0.27][self.instid-1] #pixelscales of muscats
        buffer = 0.02
        search_radius = 15 #in arcmin

        print("Running WCS Calculation of reference file...")
        cmd = f"/usr/local/astrometry/bin/solve-field --ra {self.ra} --dec {self.dec} --radius {search_radius/60} --scale-low {pixscale-buffer} --scale-high {pixscale+buffer} --scale-units arcsecperpix {ref_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        print("Complete")

        if os.path.exists(f"{ref_file_dir}/list/ref.lst"):
            def _parse_obj_file(input_file): #helper function to parse objectfile
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
            
            metadata, data = _parse_obj_file(f"{self.obsdate}/{self.target}/reference/ref-{ref_frame}.objects")
            wcsfits = f"{ref_file_dir}/df/{ref_frame}.df.new"
            with fits.open(wcsfits) as hdul:
                header = hdul[0].header
                w = wcs.WCS(header)
                ra_list, dec_list = w.all_pix2world(data["x"], data["y"], 0)
                cd_matrix = w.pixel_scale_matrix
                wcs_pixscales = np.sqrt(np.sum(cd_matrix**2, axis=0))  
                wcs_pixscales *= 3600 #convert to arcsec
                if wcs_pixscales[0] - pixscale > 0.01:
                    print("WCS calculation unsuccessful (Pixel scale mismatch)\nTry again or enter tID manually")
                    return

            threshold = 2
            threshold_deg = threshold*pixscale/3600

            for i, (ra, dec) in enumerate(zip(ra_list,dec_list)): 
                match = (self.ra - ra < threshold_deg) and (self.ra - ra > -threshold_deg) and (self.dec - dec < threshold_deg) and (self.dec - dec > -threshold_deg)
                if match:
                    tid = i + 1 #(index starts from 1 for starfind)
                    print(f"Target ID: {tid}")
                    self.tid = tid
                    return
        else:
            print("Target search unsuccessful (Reference file not found)")
            return

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
        print(f"Performing photometry for radius: {rads} | nstars = {nstars} | method = {method}")

        # Check for missing photometry files
        missing, missing_files_per_ccd = self._check_missing_photometry(rads)

        if not missing:
            print(f"Photometry is already available for radius: {available_rad}")
            return

        # Determine script to use
        script = f"scripts/auto_apphot_{method}.pl" #starfindは一回だけで十分なのでauto_apphot.plではなくapphot.plを使えば早い

        # Run photometry for missing files
        self._run_photometry_if_missing(script, nstars, rads, missing_files_per_ccd)

    def _check_missing_photometry(self, rads):
        """Checks for missing photometry files and returns a dictionary of missing files per CCD."""
        missing = False
        missing_files_per_ccd = {}

        for i in range(self.nccd):
            appphot_directory = f"{self.obsdate}/{self.target}_{i}/apphot_{self.method}"
            frame_range = self.obslog[i][self.obslog[i]["OBJECT"] == self.target]
            first_frame = int(frame_range["FRAME#1"].iloc[0])
            last_frame = int(frame_range["FRAME#2"].iloc[0])
            df, meta = self.read_photometry(ccd=i, rad=rads[0], frame=first_frame, add_metadata=True) #it takes too long to scan through all ccds, rad and frames

            def file_does_not_exist(rad, frame): #nested helper function to help judge if photometry exists
                file_path = f"{appphot_directory}/rad{rad}/MCT{self.instid}{i}_{self.obsdate}{frame:04d}.dat"
                if not os.path.exists(file_path):
                    return True  # Missing file
                return meta['nstars'] < self.nstars  #if previous photometry has smaller number of stars than requested -> need to redo
                
            missing_files = [
                f"{appphot_directory}/rad{rad}/MCT{self.instid}{i}_{self.obsdate}{frame:04d}.dat"
                for rad in rads
                for frame in range(first_frame, last_frame+1)
                if file_does_not_exist(rad, frame)
            ]

            if missing_files:
                missing = True
                missing_files_per_ccd[i] = missing_files
                #print(f"CCD {i}: Missing files for some radii: {missing_files[:5]}{'...' if len(missing_files) > 5 else ''}")
        #print("Checking for missing photometry")
        #print(f"Missing {missing}")
        return missing, missing_files_per_ccd

    def _run_photometry_if_missing(self, script, nstars, rads, missing_files_per_ccd):
        """Runs photometry for CCDs where files are missing in parallel."""
        processes = []  # Store running processes

        for i, missing_files in missing_files_per_ccd.items():
            for rad in rads:
                if any(f"rad{rad}" in f for f in missing_files):  # Only run if files for this radius are missing
                    cmd = f"perl {script} {self.obsdate} {self.target} {i} {nstars} {rad} {rad} {self.drad} > /dev/null"
                    process = subprocess.Popen(cmd, shell=True, text=True)
                    processes.append((process, i, rad))  # Store process info
                else:
                    print(f"Photometry already available for CCD={i}, rad={rad}")

        # Wait for all processes to finish
        for process, i, rad in processes:
            process.wait()  # Blocks until process completes
            print(f"Completed aperture photometry for CCD={i}, rad={rad}")

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

    def _process_single_ccd(self, ccd, rad):
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

        
    def _read_photometry_parallel(self, rad, num_processes=4): #underscore suggests the function should not be called outside the class
        """
        Read photometry data for multiple CCDs in parallel.
        
        Parameters:
        rad (int/float): Radius value
        num_processes (int): Number of parallel processes to use (default=None, uses CPU count)
        
        Returns:
        tuple: (combined_data_df, combined_metadata_df)
        """
        
        # Create partial function with fixed parameters
        process_func = partial(self._process_single_ccd, rad=rad)
        
        # Process CCDs in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_func, list(range(self.nccd)))
        
        # Filter out None results and separate data and metadata
        valid_results = [result for result in results if result is not None]
        
        return valid_results

    @time_keeper
    def check_saturation(self, rad):
        self.saturation_cids = []
        saturation_threshold = 60000
        fig, ax = plt.subplots(self.nccd,1, figsize=(15, 10))
        print(f'>> Checking for saturation with rad={rad} ... (it may take a few seconds)')
        df = self._read_photometry_parallel(rad=rad)
        print(f'## >> Done loading photometry data.')
        # Count the number of rows where peak > 60000 for this star ID
        for i in range(self.nccd):
            saturation_cids_per_ccd = []
            stop_processing = False
            for star_id in range(1,int(self.nstars)+1):
                if stop_processing:
                    break  # Exit the loop completely
                flux = df[i][df[i]["ID"] == star_id]["peak"].iloc[::-1]
                frames = list(range(len(df[i][df[i]["ID"] == star_id])))
                median = lc.moving_median(x=frames,y=flux,nsample=int(len(frames)/50))
                typical_scatter = np.std(flux-median)
                saturation_threshold_per_star = saturation_threshold - typical_scatter #flux + typical scatter が60000を超えていたらsaturation zone
                saturation_zone = np.where(median > saturation_threshold_per_star)[0]
                count_above_threshold = (flux > saturation_threshold_per_star).sum()
                percentage_above_threshold = (count_above_threshold / len(frames)) * 100
                #print(df[i])
                #print((df[i][df[i]["ID"] == star_id]["peak"] > 58000).sum())
                #print(len(df[i][df[i]["ID"] == star_id]))  
                # If more than 5% of the rows have a peak > 60000, add this star ID to the list
                if percentage_above_threshold > 5:
                    saturation_cids_per_ccd.append(star_id)
                else:
                    stop_processing = True  # Stop processing this CCD if a star is not saturated
                if i == 0:
                    label = f"Star {star_id}"
                else:
                    label = None
                ax[i].plot(frames,flux,label=label)
                ax[i].plot(frames,median,color="white",alpha=0.5)
                ax[i].plot(frames[saturation_zone],median[saturation_zone],color="red",alpha=0.5)
                #ax[i].hist(percentage_above_threshold,color=color)
            print(f'## >> CCD {i}: Done.')
            #ax[i].set_ylim(0,100)
            ax[i].set_title(f"CCD {i}")
            ax[i].set_ylim(0,saturation_threshold+2000)
            ax[i].set_xlabel("Frame")
            ax[i].set_ylabel("Peak")
            fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False, ncol=self.nstars)
            self.saturation_cids.append(saturation_cids_per_ccd)

        for i in range(self.nccd):
            print(f"WARNING: Over 5 percent of frames are saturated for cIDS {self.saturation_cids[i]} in CCD {i}")

        plt.show()
    '''
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
    '''
    def select_comparison(self, tid=None, nstars=5):
        if tid is None:
            self.find_tid()
        else:
            self.tid = tid
        print(f"{self.target} | TID = {self.tid}")
        self.check_saturation(self.rad2)
        self.cids_list = []
        for saturation_cid in self.saturation_cids:
            if saturation_cid:
                brightest_star = max(saturation_cid) + 1
            else:
                brightest_star = 1
            dimmest_star = brightest_star + nstars
            cids = list(range(brightest_star,dimmest_star))
            if self.tid in cids:
                cids.remove(self.tid)
                cids.append(dimmest_star+1)   
                cids = [str(cid) for cid in cids]
            self.cids_list.append(cids) #if too many stars are saturated, there is a risk of not having the photometry for the star. need to add logic for this

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
                outfile = f"lcf_{self.instrument}_{self.bands[i]}_{self.target}_{self.obsdate}_t{self.tid}_c{cid.replace(' ','')}_r{int(self.rad1)}-{int(self.rad2)}.csv" # file name radius must be int
                if not os.path.isfile(f"{obj_dir}/{outfile}"): #if the photometry file does not exist
                    cmd = f"perl {script_path} -apdir apphot_{self.method} -list list/object_ccd{i}.lst -r1 {int(self.rad1)} -r2 {int(self.rad2)} -dr {self.drad} -tid {self.tid} -cids {cid} -obj {self.target} -inst {self.instrument} -band {self.bands[i]} -date {self.obsdate}"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True) #this command requires the cids to be separated by space
                    #print(cmd)
                    #print(result.stdout)

                    #print(os.getcwd())
                    #print(f"Created {outfile}")
                    #print(outfile_path)
                    outfile_path = os.path.join(os.getcwd(),f"apphot_{self.method}", outfile)
                    if os.path.isfile(outfile_path): #if the photometry file now exists
                        #outfile2 = f"{instdir}/{date}/{obj}/lcf_{inst}_{bands[i]}_{obj}_{date}_t{tid}_c{suffix}_r{rad1}-{rad2}.csv"
                        subprocess.run(f"mv {outfile_path} {obj_dir}/{outfile}", shell=True)
                        print(f"## >> Created photometry for cIDs:{cid}")
                    else:
                        print(f"## >> Failed to create photometry for cIDs:{cid}")
                else:
                    print(f"## >> Photometry for cIDs:{cid} already exists.")

        os.chdir(Path(f"/home/muscat/reduction_afphot/{self.instrument}"))


class MuSCAT_PHOTOMETRY_OPTIMIZATION:
    def __init__(self, muscat_photometry):
        # Copy all attributes from the existing instance
        self.__dict__.update(muscat_photometry.__dict__)
        self.ap = np.arange(self.rad1, self.rad2+self.drad, self.drad)
        self.cids_list_opt = [[cid.replace(" ", "") for cid in cids] for cids in self.cids_list] #optphot takes cids with no space
        print('available aperture radii: ', self.ap)
        self.bands = ["g","r","i","z"] #muscat1に対応していない
        self.mask = [[[] for _ in range(len(self.cids_list_opt[i]))] for i in range(self.nccd)]

        self.min_rms_idx_list = []
        self.phot=[]
        phot_dir = f"/home/muscat/reduction_afphot/{self.instrument}/{self.obsdate}/{self.target}"

        for i in range(self.nccd):
            self.phot.append([])
            for j, cid in enumerate(self.cids_list_opt[i]):#self.cids_list_opt is only needed to access the files here
                infile = f'{phot_dir}/lcf_{self.instrument}_{self.bands[i]}_{self.target}_{self.obsdate}_t{self.tid}_c{cid}_r{str(int(self.rad1))}-{str(int(self.rad2))}.csv'
                self.phot[i].append(Table.read(infile))
                self.mask[i][j] = np.ones_like(self.phot[i][j]['GJD-2450000'], dtype=bool) #add mask depending on the number of ccds and their number of exposures

    def add_mask_per_ccd(self, key, ccd, lower=None, upper=None):
        """Applies a mask to all elements in the dataset for a given CCD."""
        for j in range(len(self.phot[ccd])):  # Loop over j (e.g., different sources)
            condition = self.mask[ccd][j]  # Start with the current mask
            
            if lower is not None:
                condition &= (self.phot[ccd][j][key] > lower[ccd])  # Apply lower bound
                
            if upper is not None:
                condition &= (self.phot[ccd][j][key] < upper[ccd])  # Apply upper bound
            
            self.mask[ccd][j] = condition  # Store mask for this j
        print(f"Added mask to {key} for CCD{ccd}")

    def add_mask(self, key, lower=None, upper=None):
        """Applies a mask to all elements in self.phot, handling lists of upper/lower bounds."""

        for i in range(len(self.phot)):  # Loop over CCDs            
            for j in range(len(self.phot[i])):  # Loop over sources, stars, or apertures
                condition = self.mask[i][j]  # Start with the current mask
                if key == "raw":
                    fcomp_key = f'flux_comp(r={self.ap[0]:.1f})'#this currently assumes the first aperture for raw flux cut
                    target_array = self.phot[i][j][fcomp_key]/self.phot[i][j]['exptime'] 
                    target_array /= np.median(target_array) #this is the normalized flux
                else:
                    target_array = self.phot[i][j][key]
                if lower is not None:
                        condition &= target_array > (lower[i] if isinstance(lower, list) else lower)
                    
                if upper is not None:
                    condition &= target_array < (upper[i] if isinstance(upper, list) else upper)
                
                self.mask[i][j] = condition  # Directly store condition, keeping shape (4, 15) 
        print(f"Added mask to {key}")
    #need to make sure masking is correct (in dimensions)

    def preview_photometry(self, cid=0, ap=0, order=2, sigma_cut=3):
        fcomp_key = f'flux_comp(r={self.ap[ap]:.1f})' # Use the aperture given in the argument
        fig, ax = plt.subplots(6, self.nccd, figsize=(16, 20), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1, 2, 2]})

        for i in range(self.nccd):
            phot_j = self.phot[i][cid]
            exptime = phot_j['exptime']
            gjd_vals = phot_j['GJD-2450000']
            raw_norm = phot_j[fcomp_key] / exptime
            raw_norm /= np.median(raw_norm)
            fcomp_data = phot_j[fcomp_key] #コンパリゾンのフラックス

            mask = self.mask[i][cid]

            ye = np.sqrt(fcomp_data[mask]) / exptime[mask] / np.median(fcomp_data[mask] / exptime[mask])

            if len(ye) > 0:
                print(">> Performing preliminary outlier detection ...")
                print(f"## >> Fitting with polynomials (order = {order}) and cutting {sigma_cut} sigma outliers ...")
                p, tcut, ycut, yecut, keep_index = lc.outcut_polyfit(gjd_vals[mask], raw_norm[mask], ye, order, sigma_cut)
                outlier_mask = np.zeros_like(mask, dtype=bool) #initialize index_mask all False
                outlier_mask[keep_index] = True #set the index_mask to True where they are not outlier points
                omittied_points = (~mask) & (~outlier_mask) #points that are either manually masked or are outliers
                mask &= outlier_mask #update the mask to exclude the outliers
                ax[0, i].plot(gjd_vals[omittied_points], raw_norm[omittied_points], 'x', c="gray")
                ax[1, i].plot(gjd_vals[omittied_points], phot_j['airmass'][omittied_points], 'x', c="gray", label=f"{sigma_cut}-sigma outliers")
                ax[2, i].plot(gjd_vals[omittied_points], phot_j['dx(pix)'][omittied_points], 'x', c="gray")
                ax[3, i].plot(gjd_vals[omittied_points], phot_j['dy(pix)'][omittied_points], 'x', c="gray")
                ax[4, i].plot(gjd_vals[omittied_points], phot_j['fwhm(pix)'][omittied_points], 'x', c="gray")
                ax[5, i].plot(gjd_vals[omittied_points], phot_j['peak(ADU)'][omittied_points], 'x', c="gray")
                for j in range(len(self.cids_list_opt)): #ここをjでループするとargumentのjと混同する
                    self.mask[i][j] = mask  # In-place modification of mask
                    print("## >> Complete and mask is updated.")

            print(f">> Ploting the photometry data for cID:{self.cids_list[i][cid]}, ap:{self.ap[ap]}")
            ax[0, i].plot(gjd_vals[mask], raw_norm[mask], '.', c="k")
            ax[1, i].plot(gjd_vals[mask], phot_j['airmass'][mask], '.', c="gray")
            ax[2, i].plot(gjd_vals[mask], phot_j['dx(pix)'][mask], '.', c="orange")
            ax[3, i].plot(gjd_vals[mask], phot_j['dy(pix)'][mask], '.', c="orange")
            ax[4, i].plot(gjd_vals[mask], phot_j['fwhm(pix)'][mask], '.', c="blue")
            ax[5, i].plot(gjd_vals[mask], phot_j['peak(ADU)'][mask], '.', c="red")

        # Set labels only on the first column
        ax[0, 0].set_ylabel('Relative flux')
        ax[1, 0].set_ylabel('Airmass')
        ax[2, 0].set_ylabel('dX')
        ax[3, 0].set_ylabel('dY')
        ax[4, 0].set_ylabel('FWHM')
        ax[5, 0].set_ylabel('Peak')

        # Set common x-axis label
        for i in range(self.nccd):
            ax[-1, i].set_xlabel('GJD - 2450000')


        plt.tight_layout(h_pad=0)  # Remove spacing between rows
        plt.legend()
        plt.show()

    @time_keeper
    def outlier_cut(self, sigma_cut=3, order=2, plot=True):
        """Performs outlier detection using polynomial fitting and sigma clipping."""
        
        self.index = [[] for _ in range(self.nccd)]  # Pre-allocate index storage
        self.ndata_diff = []  # Stores data difference arrays for each CCD
        self.rms = []  # Stores RMS arrays for each CCD
        self.min_rms_idx_list = []  # Stores min RMS indices per CCD

        print(f">> Fitting with polynomials (order = {order}) and cutting {sigma_cut} sigma outliers ... (it may take a few minutes)")

        for i in range(self.nccd):
            print(f"Computing outliers for CCD {i}")
            
            n_cids = len(self.cids_list[i])
            n_ap = len(self.ap)
            
            ndata_diff = np.zeros((n_cids, n_ap))
            rms = np.zeros((n_cids, n_ap))
            self.index[i] = [[] for _ in range(n_cids)]

            for j in range(n_cids):
                phot_j = self.phot[i][j]
                exptime = phot_j['exptime']
                gjd_vals = phot_j['GJD-2450000']
                mask = self.mask[i][j]

                fcomp_keys = [f'flux_comp(r={self.ap[k]:.1f})' for k in range(n_ap)]
                fcomp_data = np.array([phot_j[fk] for fk in fcomp_keys])

                raw_norm = (fcomp_data / exptime) / np.median(fcomp_data / exptime, axis=1, keepdims=True)
                ndata_init = fcomp_data.shape[1]

                ye = np.sqrt(fcomp_data[:, mask]) / exptime[mask] / np.median(fcomp_data / exptime, axis=1, keepdims=True)

                for k in range(n_ap):
                    if len(ye[k]) > 0:
                        p, tcut, ycut, yecut, index = lc.outcut_polyfit(gjd_vals[mask], raw_norm[k][mask], ye[k], order, sigma_cut)
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

            min_rms_idx = np.unravel_index(np.argmin(rms, axis=None), rms.shape)
            self.min_rms_idx_list.append(min_rms_idx)
            self.min_rms = rms[min_rms_idx]

            self.ndata_diff.append(ndata_diff)
            self.rms.append(rms)

        # Store best candidate values
        self.cIDs_best = [self.cids_list[i][item[0]] for i, item in enumerate(self.min_rms_idx_list)]
        self.cIDs_best_idx = [item[0] for item in self.min_rms_idx_list]
        self.ap_best = [self.ap[item[1]] for item in self.min_rms_idx_list]
        self.ap_best_idx = [item[1] for item in self.min_rms_idx_list]

        if plot:
            self.plot_outlier_cut_results()
    
    def plot_outlier_cut_results(self):
        """Plots the results of the outlier detection process."""

        fig, axes = plt.subplots(self.nccd, 2, figsize=(14, 4 * self.nccd))  # 2 columns per CCD

        for i in range(self.nccd):
            ndata_diff = self.ndata_diff[i]
            rms = self.rms[i]
            min_rms_idx = self.min_rms_idx_list[i]

            norm_diff = mcolors.Normalize(vmin=np.min(ndata_diff), vmax=np.max(ndata_diff))
            norm_rms = mcolors.Normalize(vmin=np.min(rms[rms != np.inf]), vmax=np.max(rms[rms != np.inf]))

            # **Left Plot: Number of cut data points**
            im1 = axes[i, 0].imshow(ndata_diff, cmap="coolwarm", aspect="auto", norm=norm_diff)
            axes[i, 0].set_title(f"CCD {i} - Cut Data Points")
            axes[i, 0].set_xticks(range(len(self.ap)))
            axes[i, 0].set_xticklabels([f"{self.ap[k]:.1f}" for k in range(len(self.ap))])
            axes[i, 0].set_yticks(range(len(self.cids_list[i])))
            axes[i, 0].set_yticklabels(self.cids_list[i])
            fig.colorbar(im1, ax=axes[i, 0], label="Number of Cut Data Points")

            # **Right Plot: RMS of flux differences**
            im2 = axes[i, 1].imshow(rms, cmap="cividis", aspect="auto", norm=norm_rms)
            axes[i, 1].set_title(f"CCD {i} - RMS")
            axes[i, 1].set_xticks(range(len(self.ap)))
            axes[i, 1].set_xticklabels([f"{int(self.ap[k])}" for k in range(len(self.ap))])
            axes[i, 1].set_yticks(range(len(self.cids_list[i])))
            axes[i, 1].set_yticklabels(self.cids_list[i])
            fig.colorbar(im2, ax=axes[i, 1], label="RMS")

            # **Highlight the min RMS cell with a white square**
            j_min, k_min = min_rms_idx
            rect = patches.Rectangle((k_min - 0.5, j_min - 0.5), 1, 1, linewidth=3, edgecolor='white', facecolor='none')
            axes[i, 1].add_patch(rect)

        print("Plotting results")
        plt.tight_layout()
        plt.show()

    def _reselect_comparison(self):
        reselected_cids_list = []
        for cid in self.cIDs_best:
            reselected_cids = []
            brightest_star = int(cid)
            nstars = 5
            dimmest_star = brightest_star + nstars
            cids = list(range(brightest_star,dimmest_star))
            if self.tid in cids:
                cids.remove(self.tid)
                cids.append(dimmest_star+1)    
            cids.remove(brightest_star)
            for r in range(0,nstars): #nstars choose r
                for combo in itertools.combinations(cids, r):
                    current_combo = ' '.join(str(x) for x in sorted([brightest_star] + list(combo)))
                    reselected_cids.append(current_combo)
            reselected_cids_list.append(reselected_cids)
        return reselected_cids_list
    
    def iterate_optimization(self):
        min_rms_list = [np.inf,self.min_rms]
        drad = 1

        if any(idx in {self.ap[0]} for idx in self.ap_best): #if the lowest rms is the smallest aperture 
            rad1 = self.ap[0] - drad
            rad2 = self.ap[-1]
            
        elif any(idx in {self.ap[-1]} for idx in self.ap_best): #if the lowest rms is the largest aperture 
            rad1 = self.ap[0]
            rad2 = self.ap[-1] + drad
        else:
            if self.drad == 1:
                print("Already optimal")
                return
            else:
                rad1 = min(self.ap_best) - drad #the smallest aperture
                rad2 = max(self.ap_best) + drad #the largest aperture

        reselected_cids = self._reselect_comparison()

        while min_rms_list[-1] < min_rms_list[-2]: #whle the rms keeps improving
            print(f"Returning to photometry for aperture optimization... (Iteration: {len(min_rms_list)-1})")

            photometry = MuSCAT_PHOTOMETRY(parent=self)
            photometry.run_apphot(nstars=self.nstars, rad1=rad1, rad2=rad2, drad=drad, method="mapping")
            photometry.cids_list = reselected_cids
            photometry.create_photometry()

            saved_mask = self.mask
            '''
            optimization = MuSCAT_PHOTOMETRY_OPTIMIZATION(photometry)
            optimization.mask = self.mask #adds the same mask
            optimization.outlier_cut(plot=False)
            min_rms = optimization.min_rms
            min_rms_list.append(min_rms)
            '''
            self.__init__(photometry)
            self.mask = saved_mask
            self.outlier_cut(plot=False)
            min_rms_list.append(self.min_rms)

            if any(idx in {self.ap[0]} for idx in self.ap_best):
                rad1 -= 1
            elif any(idx in {self.ap[-1]} for idx in self.ap_best):
                rad2 += 1
            else:
                rad1 -= 1
                rad2 += 1
            print(f"Minimum rms: {min_rms_list[-2]} -> {min_rms_list[-1]}")
        self.plot_outlier_cut_results()

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
        plt.savefig(f"ut3/muscat/reduction_afphot/notebooks/general/{self.target}/{outfile}",bbox_inches='tight',pad_inches=0.1)
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