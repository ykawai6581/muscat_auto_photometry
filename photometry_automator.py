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
    def __init__(self,instrument,obsdate):

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

    def select_comparison(self, tid):
        self.tid = tid
        print(f"{self.target} | TID = {tid}")
        self.check_saturation(self.rad1)
        self.cids_list = []
        for saturation_cid in self.saturation_cids:
            if saturation_cid:
                brightest_star = max(saturation_cid) + 1
            else:
                brightest_star = 1
            cids = get_combinations(brightest_star, brightest_star + 4, tid)
            self.cids_list.append(cids)

    @time_keeper
    def create_photometry(self):
        script_path = "/home/muscat/reduction_afphot/tools/afphot/script/auto_mklcmklc_flux_collect_csv.pl"
        print(">> Creating photometry file for")
        print(f"| Target = {self.target} | TID = {self.tid} | r1={self.rad1} r2={self.rad2} dr={self.drad} | (it may take minutes)")
        for i in range(self.nccd):
            print(f'>> CCD{i}')
            for cid in self.cids_list[i]:
                cmd = f"perl {script_path} -apdir apphot_{self.method} -list path/object_ccd{i}.lst -r1 {self.rad1} -r2 {self.rad2} -dr {self.drad} -tid {self.tid} -cids {cid} -obj {self.target} -inst {self.instrument} -band {self.bands[i]} -date {self.obsdate}"
                print(f"## >> Created photometry for cIDs:{cid}")
                subprocess.run(cmd, shell=True, capture_output=True, text=True)


class MuSCAT_PHOTOMETRY_OPTIMIZATION:
    def __init__(self, muscat_photometry):
        # Copy all attributes from the existing instance
        self.__dict__.update(muscat_photometry.__dict__)
        self.r = np.arange(self.rad1, self.rad2+self.drad, self.drad)
        self.cids_list = [[cid.replace(" ", "") for cid in cids] for cids in self.cids_list]
        print('available aperture radii: ', self.r)
        self.bands = ["g","r","i","z"]

        self.phot=[]
        phot_dir = f"/home/muscat/reduction_afphot/{self.instrument}/{self.obsdate}/{self.target}"

        for i in range(self.nccd):
            self.phot.append([])
            for cid in self.cids_list[i]:
                infile = f'{phot_dir}/lcf_{self.instrument}_{self.bands[i]}_{self.target}_{self.obsdate}_t{self.tid}_c{cid}_r{str(int(self.rad1))}-{str(int(self.rad2))}.csv'
                self.phot[i].append(Table.read(infile))

'''
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

    def run_apphot(self, nstars, rad1, rad2, drad, method="mapping"):
        self.rad1 = float(rad1)
        self.rad2 = float(rad2)
        self.drad = float(drad)
        self.method = method
        self.nstars = int(nstars)
        rads = np.arange(rad1, rad2+1, drad)

        print(f"Performing photometry for radius: {rads}")
        available_rad = [[p.name[3:] for p in Path(f"{self.obsdate}/{self.target}_{i}/apphot_{method}").glob("*/")] for i in range(self.nccd)][0]#assuming same radius for all bands

        missing = False
        for i in range(self.nccd):
            for j in range(len(rads)):
                rad = float(rads[j])
                appphot_directory = f'{self.obsdate}/{self.target}_{i}/apphot_{method}'
                first_frame = int(self.obslog[i][self.obslog[i]["OBJECT"] == self.target]["FRAME#1"])
                last_frame  = int(self.obslog[i][self.obslog[i]["OBJECT"] == self.target]["FRAME#2"])
                missing_files = [
                    f"{appphot_directory}/rad{rad}/MCT{self.instid}{i}_{self.obsdate}{frame:04d}.dat"
                    for frame in range(first_frame, last_frame)
                    if not os.path.exists(f"{appphot_directory}/rad{rad}/MCT{self.instid}{i}_{self.obsdate}{frame:04d}.dat")
                ]
                print(f'ccd:{i},rad:{rad}')
                print(missing_files)
                if missing_files:
                    missing = True
                else:
                    pass
        if missing:
            print(f"Photometry for this set of radius is incomplete")
        else:
            print(f"Photometry is already available for radius: {available_rad}")
            sys.exit()

        if method=='mapping':
            script = 'scripts/auto_apphot_mapping.pl'  
        elif method=='centroid':
            script = 'scripts/auto_apphot_centroid.pl'

        for i in range(self.nccd):
            appphot_directory = f'{self.obsdate}/{self.target}_{i}/apphot_{method}'
            first_frame = int(self.obslog[i][self.obslog[i]["OBJECT"] == self.target]["FRAME#1"])
            last_frame  = int(self.obslog[i][self.obslog[i]["OBJECT"] == self.target]["FRAME#2"])
            for j in range(len(rads)):
                rad = float(rads[j])
                missing_files = [
                    f"{appphot_directory}/rad{rad}/MCT{self.instid}{i}_{self.obsdate}{frame:04d}.dat"
                    for frame in range(first_frame, last_frame)
                    if not os.path.exists(f"{appphot_directory}/rad{rad}/MCT{self.instid}{i}_{self.obsdate}{frame:04d}.dat")
                ]
                if missing_files:
                    cmd = f"perl {script} {self.obsdate} {self.target} {i} {nstars} {rad} {rad} {drad} > /dev/null"
                    subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    print(f'Completed aperture photometry for CCD={i}, rad={rad}')
                else:
                    print(f"Photometry is already available for CCD={i}", rad={rad})

    
'''