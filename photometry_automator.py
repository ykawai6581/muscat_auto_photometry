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
        self.ra, self.dec = query_radec(target_from_filename())

        self.nccd = 3 if instrument == "muscat" else 4
        self.obslog = []
        self.obsdate = obsdate
        self.instrument = instrument
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

    @time_keeper
    def config_object(self):
        ## Setting configure files for object

        exposure = [float(ccd["EXPTIME(s)"][ccd["OBJECT"] == self.target]) for ccd in self.obslog]  # exposure times (sec) for object
        for i in range(self.nccd):
            exp=exposure[i]
            cmd = f'perl scripts/config_object.pl {self.obsdate} {self.target} {i} -auto_obj -auto_dark {exp}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(result.stdout)

    @time_keeper
    def reduce_flat(self):
        ## Reducing FLAT images 
        for i in range(self.nccd):
            print(f'>> Reducing FLAT images of CCD{i} ... (it may take tens of seconds)')
            cmd = f"perl scripts/auto_mkflat.pl {self.obsdate} {i} > /dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    @time_keeper
    ## Reducing Object images 
    def run_auto_mkdf(self):
        for i in range(self.nccd):
            cmd = f"perl scripts/auto_mkdf.pl {self.obsdate} {self.target} {i} > /dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(f'Completed auto_mkdf.pl for CCD{i}')
    
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
        cmd = f"perl scripts/make_reference.pl {self.obsdate} {self.target} --ccd={ref_ccd} --refid={refid}"
        subprocess.run(cmd, shell=True, capture_output=True, text=True)

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

        dec_str = f"+{self.dec}" if self.dec > 0 else f"-{self.dec}"

        IFrame(f"https://aladin.u-strasbg.fr/AladinLite/?target={self.ra}{dec_str}&fov=0.2", width=700, height=500)

    ## Performing aperture photometry
    @time_keeper
    def run_apphot(self, nstars, rad1, rad2, drad, method="mapping"):
        self.rad1 = float(rad1)
        self.rad2 = float(rad2)
        self.drad = int(drad)
        self.method = method
        self.nstars = int(nstars)
        if method=='mapping':
            script = 'scripts/auto_apphot_mapping.pl'  
        elif method=='centroid':
            script = 'scripts/auto_apphot_centroid.pl'
        rads = np.arange(rad1, rad2+1, drad)
        for i in range(self.nccd):
            for j in range(len(rads)):
                rad = rads[j]
                cmd = f"perl {script} {self.obsdate} {self.target} {i} {nstars} {rad} {rad} {drad} > /dev/null"
                subprocess.run(cmd, shell=True, capture_output=True, text=True)
                print('Completed aperture photometry for CCD={i}, rad={rad1}')


    def read_photometry(self, ccd, rad, add_metadata=False):
        filepath = f"/ut3/muscat/reduction_afphot/{self.instrument}/{self.obsdate}/{self.target}_{ccd}/apphot_{self.method}/rad{str(rad)}"
        try:
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
                    else:
                        table_data.append(line.strip())
            
            # Convert to DataFrame
            df = pd.DataFrame([row.split() for row in table_data], 
                            columns=['ID', 'xcen', 'ycen', 'nflux', 'flux', 'err', 
                                    'sky', 'sky_sdev', 'SNR', 'nbadpix', 'fwhm', 'peak'])
            
            # Convert numeric columns
            numeric_cols = df.columns.difference(['filename', 'ccd'])
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            
            # Add file information
            df['filename'] = Path(filepath).name
            df['ccd'] = Path(filepath).parent.name
            
            if add_metadata:
                for key, value in metadata.items():
                    df[key] = value
            
            # Return DataFrame with ID and peak value
            return df[['ID', 'peak']]

        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    def read_photometry_with_progress(self, files, ccd, rad, add_metadata=False):
        with ProcessPoolExecutor() as executor:
            total = len(files)
            ccd_peak_data = []  # List to store DataFrames for each CCD
            for i, df in enumerate(executor.map(partial(self.read_photometry, ccd=ccd, rad=rad, add_metadata=add_metadata), files)): 
                if df is not None:
                    ccd_peak_data.append(df)
                print(f"Progress: {i+1}/{total} files processed", end='\r')
        print("\nDone!")
        return ccd_peak_data

    def read_all_files_parallel(self, rad=None, add_metadata=False):
        base_dir = f"/ut3/muscat/reduction_afphot/{self.instrument}/{self.obsdate}/"
        if rad is None:
            rad = str(float(self.rad1))
        # Iterate over each CCD folder and process files
        ccd_peak_values = {}

        # Get all file paths
        start_time = time.time()
        
        for ccd_folder in Path(base_dir).glob(f"{self.target}_*"):
            files = list((ccd_folder / f"apphot_{self.method}/rad{rad}").glob("*.dat"))
            ccd = ccd_folder.name[-1]  # Extract CCD name from the folder]
            ccd_peak_values[ccd] = self.read_photometry_with_progress(files, ccd, rad, add_metadata)

        results = []
        self.saturation_cids = []
        for ccd, peak_data in ccd_peak_values.items():
            # Pivot DataFrame to have star IDs as columns and peak values as rows
            df_ccd = pd.concat(peak_data, axis=0).pivot(index=None, columns='ID', values='peak')
            saturation_cids_per_ccd = []
            # Identify IDs where peak count is over 60,000 in more than 5% of rows
            for star_id in df_ccd.columns:
                # Count the number of rows where peak > 60000 for this star ID
                count_above_threshold = (df_ccd[star_id] > 60000).sum()
                percentage_above_threshold = count_above_threshold / len(df_ccd) * 100
                
                # If more than 5% of the rows have a peak > 60000, add this star ID to the list
                if percentage_above_threshold > 5:
                    saturation_cids_per_ccd.append(star_id)
            self.saturation_cids.append(saturation_cids_per_ccd)
            results[ccd] = df_ccd

        end_time = time.time()
        for i in range(self.nccds):
            print(f"WARNING: Over 5 percent of frames are saturated for cIDS {self.saturation_cids[i]} in CCD {i}")
        print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
        
        return results

    def select_comparison(self, tid, cids = None):
        if cids == None:
            cids = get_combinations(1,5,tid)

        self.cids = cids 
        self.tid  = tid

        for i in range(len(self.cids)):
            cid = self.cids[i]
            print(f'>> Creating photometry file for cIDs=[{cid}] .. (it may take minutes)')
            cmd = f"perl scripts/auto_mklc.pl -date {self.obsdate} -obj {self.target} -ap_type {self.method} -r1 {self.rad1} -r2 {self.rad2} -dr {self.drad} -tid {self.tid} -cids {cid}"
            subprocess.run(cmd, shell=True, capture_output=True, text=True)

'''
class MuSCAT_PHOTOMETRY_OPTIMIZATION:
    def __init__(self, muscat_photometry):
        # Copy all attributes from the existing instance
        self.__dict__.update(muscat_photometry.__dict__)
        self.r = np.arange(self.rad1, self.rad2+self.drad, self.drad)
        self.cids = [item.replace(" ", "") for item in self.cids]
        print('available aperture radii: ', self.r)
        self.bands = ["g","r","i","z"]

        self.phot=[]
        phot_dir = '/home/muscat/reduction_afphot/' + {self.instrument} + '/' + {self.obsdate} + '/' + {self.target}
        for i in range(self.nccd):
            self.phot.append([])
            
            for j in range(len(self.cids)):
                infile = f'{phot_dir}/lcf_{self.instruments}_{self.bands[i]}_{self.target}_{self.obsdate}_t{self.tid}_c{self.cids[j]}_r{str(self.rad1)}-{str(self.rad2)}.csv'
                self.phot[i].append(Table.read(infile))

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