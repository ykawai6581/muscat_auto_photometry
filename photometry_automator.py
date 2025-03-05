import subprocess
import pandas as pd
from io import StringIO
from IPython.display import clear_output

import numpy as np
import os
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import sys
from tqdm.asyncio import tqdm as tqdm_async
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import shutil

#from tqdm import tqdm

from IPython.display import IFrame
import asyncio
from types import SimpleNamespace

from pathlib import Path
import multiprocessing as mp
from functools import partial
import glob
from concurrent.futures import ProcessPoolExecutor
import time

import LC_funcs as lc
from apphot_yg import ApPhotometry, PhotometryConfig
from file_utilities import parse_dat_file, parse_obj_file, load_par_file, load_geo_file
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
import warnings
from astropy.wcs import FITSFixedWarning 


warnings.simplefilter('ignore', FITSFixedWarning)


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
        #print("Done.")
        print(f"### Wall time: {minutes} minutes {seconds} seconds ###")
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

def run_all_ccds(nccd, method, ccd_specific_args=None, *shared_args, **shared_kwargs):
    """
    Wrapper function to run a method in parallel for all CCDs.
    - Collects return values if any.
    - Runs the method in parallel for all CCDs.
    ex:
    ccd_specific_args = [
        ("ccd0_arg1", "ccd0_arg2"),  # Arguments for CCD 0
        ("ccd1_arg1", "ccd1_arg2"),  # Arguments for CCD 1
        ("ccd2_arg1", "ccd2_arg2")   # Arguments for CCD 2
    ]
    """

    results = {}
    with ProcessPoolExecutor(max_workers=nccd) as executor:
        futures = {}
        for ccd in range(nccd):
            # Extract CCD-specific arguments if provided
            ccd_args = ccd_specific_args[ccd] if ccd_specific_args and ccd < len(ccd_specific_args) else {}
            # Submit the task with dynamic arguments
            futures[executor.submit(
                method,
                ccd,
                *shared_args,     # Pass shared positional arguments
                **ccd_args,          # Pass CCD-specific arguments
                **shared_kwargs   # Pass shared keyword arguments
            )] = ccd

        for future in futures:
            ccd = futures[future]
            try:
                result = future.result()  # Capture return value if any
                results[ccd] = result  # Store per-CCD results
            except Exception as e:
                print(f"Error in CCD {ccd}: {e}")
                results[ccd] = None

    return results if any(v is not None for v in results.values()) else None


from muscat_photometry import target_from_filename, obsdates_from_filename, query_radec

os.umask(0o002)
os.nice(19)

print(f"Running notebook for {target_from_filename()}")
print(f"Available obsdates {obsdates_from_filename()}")

class MuSCAT_PHOTOMETRY:
    def __init__(self,instrument=None,obsdate=None,parent=None,ra=None,dec=None):
        if not ((instrument is not None and obsdate is not None) or parent is not None):
            raise ValueError("Either both 'instrument' and 'obsdate' or 'parent' must be provided.")
        
        if parent:
            # Copy attributes from the parent if given
            self.__dict__.update(parent.__dict__)
        else:
            target = target_from_filename()

            instrument_id = {"muscat":1,"muscat2":2,"muscat3":3,"muscat4":4}

            self.instrument = instrument

            if self.instrument not in list(instrument_id.keys()):
                print(f"Instrument has to be one of {list(instrument_id.keys())}")
                return
            
            try:
                self.ra, self.dec = query_radec(target_from_filename())
            except:
                if ra and dec:
                    self.ra, self.dec = ra, dec #implement conversion from hms to deg
                else:
                    print("Failed to query RA and Dec from Simbad. Enter ra and dec manually.")
                    return

            self.nccd = 3 if self.instrument == "muscat" else 4
            self.obsdate = obsdate
            self.obslog = []
            self.instid = instrument_id[self.instrument]
            self.pixscale = [0.358, 0.435, 0.27,0.27][self.instid-1] #pixelscales of muscats

            muscat_bands = {
                "muscat" : ["g","r","z"],
                "muscat2" :["g","r","i","z"],
                "muscat3" :["r","i","g","z"],
                "muscat4" :["g","r","i","z"],
            }

            self.bands = muscat_bands[self.instrument]
            os.chdir('/home/muscat/reduction_afphot/'+self.instrument)

            self.load_obslog()

            self.obj_names = list(self.obslog[0]['OBJECT'][(self.obslog[0]['OBJECT'] != 'FLAT') & (self.obslog[0]['OBJECT'] != 'DARK')])
            if target in self.obj_names: #implement checks for variability in target name
                self.target = target
            else:
                pick_target = input(f"Available object names {[f'{i}|{item}' for i, item in enumerate(self.obj_names)]}")
                print(pick_target)
                self.target = self.obj_names[int(pick_target[0])]
            print(f"Continuing photometry for {self.target}")
            self.tid = None
            self.target_dir = f"{self.obsdate}/{self.target}"
            self.flat_dir = f"{self.obsdate}/FLAT"
            self.frames = [] #object.lstから読んできて保存されるようにする
            #self.target_dir = f"{self.obsdate}/{self.target}

    def load_obslog(self):
        for i in range(self.nccd):
            print(f'\n=== CCD{i} ===')
            cmd = f'perl /home/muscat/obslog/show_obslog_summary.pl {self.instrument} {self.obsdate} {i}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            obslog_perccd = result.stdout
            print(obslog_perccd)  # Optional: Print to verify
            obslog_perccd = obslog_perccd.lstrip("# ")
            obslog_perccd_df = pd.read_csv(StringIO(obslog_perccd), delim_whitespace=True)
            self.obslog.append(obslog_perccd_df)

    def config_flat(self, ccd):
        ## Setting configure files for flat

        ## Change the following values
        #======
        self.flat_first_frameIDs = [int(ccd[ccd["OBJECT"] == "FLAT"]["FRAME#1"]) for ccd in self.obslog]  # first frameIDs of flat data
        #flat_first_frameIDs = [1306, 1857, 2414] 
        #======

        flat_conf_path = f"{self.flat_dir}/list/flat_ccd{ccd}.conf"
        #print(flat_conf_path)
        if not os.path.exists(flat_conf_path):
            cmd = f'perl scripts/config_flat.pl {self.obsdate} {ccd} -set_dir_only'
            subprocess.run(cmd, shell=True, capture_output=True, text=True)

            flat_conf = f'{self.flat_dir}/list/flat_ccd{ccd}.conf'
            print(flat_conf)
            text = f'flat {self.flat_first_frameIDs[ccd]} {self.flat_first_frameIDs[ccd]+49}\nflat_dark {self.flat_first_frameIDs[ccd]+50} {self.flat_first_frameIDs[ccd]+54}'
            with open(flat_conf, mode='w') as f:
                f.write(text)
            result = subprocess.run(['cat', flat_conf], capture_output=True, text=True)
            print(result.stdout)
            print('\n')
        else:
            print(f"config file already exisits under {self.flat_dir}/list/flat_ccd{ccd}.conf")

    def config_object(self,ccd):
        ## Setting configure files for object
        obslog = self.obslog[ccd]
        exposure = float(obslog["EXPTIME(s)"][obslog["OBJECT"] == self.target])  # exposure times (sec) for object
        obj_conf_path = f"{self.target_dir}_{ccd}/list/object_ccd{ccd}.conf"
        #print(obj_conf_path)
        if not os.path.exists(obj_conf_path):
            cmd = f'perl scripts/config_object.pl {self.obsdate} {self.target} {ccd} -auto_obj -auto_dark {exposure}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(result.stdout)
        else:
            print(f"config file already exisits under {self.target_dir}_{ccd}/list/ as object_ccd{ccd}.conf")

    def reduce_flat(self, ccd):
        ## Reducing FLAT images 
        flat_path = f"{self.flat_dir}/flat/flat_ccd{ccd}.fits"
        #print(flat_path)
        if not os.path.exists(flat_path):
            print(f'>> Reducing FLAT images of CCD{ccd} ... (it may take tens of seconds)')
            cmd = f"perl scripts/auto_mkflat.pl {self.obsdate} {ccd} > /dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            print(f"flat file already exisits under {self.flat_dir}/flat/ as flat_ccd{ccd}.fits")
    
    ## Reducing Object images 
    def auto_mkdf(self,ccd):
        df_directory = f'{self.target_dir}_{ccd}/df'
        frame_range = self.obslog[ccd][self.obslog[ccd]["OBJECT"] == self.target]
        first_frame = int(frame_range["FRAME#1"].iloc[0])
        last_frame = int(frame_range["FRAME#2"].iloc[0])
        print(f'CCD{ccd}: Reducing frames {first_frame}~{last_frame} ...')
        missing_files = []
        
        for frame in range(first_frame, last_frame+1):
            if not os.path.exists(os.path.join(df_directory, f"MCT{self.instid}{ccd}_{self.obsdate}{frame:04d}.df.fits")):
                missing_files.append(f"MCT{self.instid}{ccd}_{self.obsdate}{frame:04d}.df.fits")

        if missing_files:
            cmd = f"perl scripts/auto_mkdf.pl {self.obsdate} {self.target} {ccd} > /dev/null"
            subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(f'Completed auto_mkdf.pl for CCD{ccd}')
        else:
            print(f"df file already exisits under /{self.target}_{ccd}/df/")

    def create_reference(self, ccd=0, refid_delta=0, threshold=10, rad=20):
        ## Creating a reference image

        ## Change the folloiwng value if necessary
        #======
        #ref_ccd=1      # CCD number for the reference image
        ref_ccd=ccd
        refid= int(self.obslog[ref_ccd][self.obslog[ref_ccd]["OBJECT"] == self.target]["FRAME#1"])#if you are okay with setting the first frame as reference
        refid+=refid_delta
        self.ref_file = f"MCT{self.instid}{ref_ccd}_{self.obsdate}{refid:04d}"
        #======
        cmd = f"perl scripts/make_reference.pl {self.obsdate} {self.target} --ccd={ref_ccd} --refid={refid} --th={threshold} --rad={rad}"
        subprocess.run(cmd, shell=True, capture_output=True, text=True)
        self.find_tid(ccd, refid_delta, threshold, rad)
    
    def show_frame(self, frame, xy= None ,rad=10, reference=False):
        """Plots a single FITS frame with reference markers."""
        if reference:
            x0, y0 = self.read_reference()
        elif xy:
            x0, y0 = xy
        else:
            dat_file = f"{frame.split('/')[-1][0:-8]}.dat"
            ccd = dat_file[4]
            x0, y0 = self.map_reference(ccd,dat_file)
        plt.figure(figsize=(10, 10))
        with fits.open(frame) as hdul:
            data = hdul[0].data

        norm = ImageNormalize(data, interval=ZScaleInterval())
        plt.imshow(data, origin='lower', norm=norm, cmap='gray')
        plt.title(f"{frame}")

        # Add reference points as circles
        for i, _ in enumerate(zip(x0, y0)):
            if i == self.tid - 1:
                color = "red"
                text_color = "yellow"
                text = f"{self.target}|{self.tid}"
            else:
                color = "chocolate"
                text_color = "chocolate"
                text = f"{i+1}"

            circ = plt.Circle((x0[i], y0[i]), rad, color=color, fill=False)
            plt.gca().add_patch(circ)
            plt.text(x0[i] + rad / 2, y0[i] + rad / 2, text, fontsize=14, color=text_color)

        plt.show()

    def show_missing_frames(self,rads=None):
        missing, missing_files, missing_rads, nframes = self._check_missing_photometry(rads=rads)
        frames = [f"{self.target_dir}_0/rawdata/{file[:-4]}.fits" 
                    for _, missing_files_per_ccd in missing_files.items()
                    for file in missing_files_per_ccd]  # rawdata is a symbolic link
        for frame in frames:
            self.show_frame(frame=frame)

    def read_reference(self,nstars=10):
        '''
        Reads the reference and returns the star positions.
        First called in self.read_wcs_calculation() using nstars=100
        In other instance, coordinates are returned for the larger of self.tid and 10
        '''
        if self.tid is not None:
            nstars = max(self.tid,nstars)
        ref_path = Path(f"{self.target_dir}/list/ref.lst")
        if os.path.exists(ref_path):            
            metadata, data = parse_obj_file(f"{self.target_dir}/reference/ref-{self.ref_file}.objects")
            x0, y0 = np.array(data["x"][:nstars]),np.array(data["y"][:nstars]) #array of pixel coordinates for stars in the reference frame
            return x0, y0
        else:
            print("No reference file found.")
            return
    
    def map_reference(self, ccd, frame):  #frameidにした方がいい  
        x0, y0 = self.read_reference()
        geoparam_file_path = f"{self.target_dir}_{ccd}/geoparam/{frame[:-4]}.geo" #extract the frame name and modify to geoparam path 
        geoparams = load_geo_file(geoparam_file_path)#毎回geoparamsを呼び出すのに時間がかかりそう
        geo = SimpleNamespace(**geoparams)
        x = geo.dx + geo.a * x0 + geo.b * y0
        y = geo.dy + geo.c * x0 + geo.d * y0
        return x, y 
    
    def map_all_frames(self, ccd, frames):
        starlist = []
        for frame in frames:
            #print(frame)
            x, y = self.map_reference(ccd, frame) 
            starlist.append([x,y])
        return starlist
    
    def wcs_calculation(self,ccd):
        buffer = 0.02
        search_radius = 15 #in arcmin
        ref_file_path = f"{self.target_dir}_{ccd}/df/{self.ref_file}.df.fits"

        print(">> Running WCS Calculation of reference file...")
        cmd = f"/usr/local/astrometry/bin/solve-field --ra {self.ra} --dec {self.dec} --radius {search_radius/60} --scale-low {self.pixscale-buffer} --scale-high {self.pixscale+buffer} --scale-units arcsecperpix {ref_file_path}"
        print(cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        print("## >> Complete.")

        return self.read_wcs_calculation(ccd)

    def read_wcs_calculation(self, ccd):
        wcsfits = f"{self.target_dir}_{ccd}/df/{self.ref_file}.df.new"
        x0, y0 = self.read_reference(nstars=100)
        with fits.open(wcsfits) as hdul:
            header = hdul[0].header

        w = wcs.WCS(header)
        ra_list, dec_list = w.all_pix2world(x0, y0, 0)
        cd_matrix = w.pixel_scale_matrix
        wcs_pixscales = np.sqrt(np.sum(cd_matrix**2, axis=0))  
        wcs_pixscales *= 3600 #convert to arcsec
        if wcs_pixscales[0] - self.pixscale > 0.01:
            print("## >> WCS calculation unsuccessful (Pixel scale mismatch)")
            return
        return ra_list, dec_list

    def find_tid(self, ccd=0, refid_delta=0, threshold=10, rad=20):
        wcsfits = f"{self.target_dir}_{ccd}/df/{self.ref_file}.df.new"
        if os.path.exists(wcsfits):
            ra_list, dec_list = self.read_wcs_calculation(ccd)
        else:
            ra_list, dec_list = self.wcs_calculation(ccd)
        threshold_pix = 2
        threshold_deg = threshold_pix*self.pixscale/3600

        for i, (ra, dec) in enumerate(zip(ra_list,dec_list)): 
            match = (self.ra - ra < threshold_deg) and (self.ra - ra > -threshold_deg) and (self.dec - dec < threshold_deg) and (self.dec - dec > -threshold_deg)
            if match:
                tid = i + 1 #(index starts from 1 for starfind)
                #print(f"Target ID: {tid}")
                self.tid = tid
                print("___Match!_______________________________________________")
                print(f"{self.target} | TID = {self.tid}")
                print("________________________________________________________")
                ref_fits =f"{self.target_dir}/reference/ref-{self.ref_file}.fits"
                self.show_frame(frame=ref_fits,rad=rad,reference=True)
                return
        if rad < 1:
            print("## >> WCS calculation unsuccessful (Star not detected in object file)\nTry again or enter tID manually")
            return 
        
        print(f"Locating target with rad={rad-1}")
        self.create_reference(ccd=ccd,refid_delta=refid_delta,rad=rad-1)

    def process_object(self, ccd):
        objlist = f"list/object_ccd{ccd}.lst"

        if self.instrument == "muscat":
            objlist = f"list/object_ccd{ccd}_corr.lst"

        ## Starfind
        print(f"starfind_centroid.pl {objlist}")
        subprocess.run(["starfind_centroid.pl", objlist], cwd=f"{self.target_dir}_{ccd}", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        ## Set reference
        reflist = f"{self.target_dir}/list/ref.lst"

        print(f"cp {reflist} {self.target_dir}_{ccd}/list/")
        shutil.copy(reflist, f"{self.target_dir}_{ccd}/list/")

        refdir = f"{self.target_dir}/reference"
        ref_symlink = f"{self.target_dir}_{ccd}/reference"
        
        print(f"ln -s {refdir} {self.target_dir}_{ccd}/list/")
        
        # Create symbolic link if it doesn't exist
        if os.path.islink(ref_symlink):
            os.unlink(ref_symlink)  # Remove the existing symlink

        os.symlink(os.path.abspath(refdir), ref_symlink)

        ## Starmatch
        print(f"starmatch.pl list/ref.lst {objlist}")
        result = subprocess.run(f"starmatch.pl list/ref.lst {objlist}", cwd=f"{self.target_dir}_{ccd}", shell=True, capture_output=True, text=True)
        
    ## Performing aperture photometry
    async def run_apphot(self, nstars=None, rad1=None, rad2=None, drad=None, method="mapping",
                         sky_calc_mode=1, const_sky_flag=0, const_sky_flux=0, const_sky_sdev=0):

        self.rad1, self.rad2, self.drad, self.method, self.nstars = float(rad1), float(rad2), float(drad), method, int(nstars)
        rads = np.arange(self.rad1, self.rad2 + 1, self.drad)

        # Check for missing photometry files
        missing, missing_files, missing_rads ,nframes = self._check_missing_photometry(rads=rads)

        self.rad_to_use = missing_rads
        available_rads = [rad for rad in rads if rad not in self.rad_to_use]

        if not missing:
            print(f">> Photometry is already available for radius: {available_rads}")
            return

        for i in range(self.nccd):
            if not self.rad_to_use:
                print(f">> CCD={i} | Photometry already available for rads = {rads}")
                continue
            elif len(self.rad_to_use) != len(rads):
                print(f">> CCD={i} | Photometry already available for rads = {available_rads}")

        config = self._config_photometry(sky_calc_mode, const_sky_flag, const_sky_flux, const_sky_sdev)
        
        ccd_specifc_arg = [{"frames": sorted(missing_files[i])} for i in range(self.nccd)]
        print(">> Mapping all frames to reference frames...")
        results = run_all_ccds(self.nccd,self.map_all_frames, ccd_specifc_arg)
        print("## >> Complete...")
        
        starlists = [result for _, result in results.items()]

        missing_images = [sorted(missing_files[i]) for i in range(self.nccd)]
        missing_images = [
            [
                f"{self.target_dir}_{ccd}/df/{file[:-4]}.df.fits" 
                for file in missing_images_per_ccd
            ] 
            for ccd, missing_images_per_ccd in enumerate(missing_images)
        ]

        header = f">> Performing photometry for radius: {self.rad_to_use} | nstars = {nstars} | method = {method}"
        print(header)

        monitor = asyncio.create_task(self.monitor_photometry_progress(header))

        await asyncio.to_thread(ApPhotometry.process_all_ccds,missing_images,starlists,config)
        await monitor

    def _check_missing_photometry_per_ccd(self, ccd, rads):
        """Checks for missing photometry files and returns a dictionary of missing files per CCD."""
        missing = False
        missing_files = set()
        missing_rads = set()

        apphot_directory = f"{self.target_dir}_{ccd}/apphot_{self.method}"
        frame_range = self.obslog[ccd][self.obslog[ccd]["OBJECT"] == self.target]
        first_frame = int(frame_range["FRAME#1"].iloc[0])
        last_frame = int(frame_range["FRAME#2"].iloc[0])
        nframes = last_frame-first_frame+1

        def file_exists(rad, frame): #nested helper function to help judge if photometry exists
            file_path = f"{apphot_directory}/rad{rad}/MCT{self.instid}{ccd}_{self.obsdate}{frame:04d}.dat"
            if os.path.exists(file_path):
                return True

        for rad in rads:
            for frame in range(first_frame, last_frame+1):
                    if not file_exists(rad, frame):
                        missing_files.add(f"MCT{self.instid}{ccd}_{self.obsdate}{frame:04d}.dat")
                        missing_rads.add(rad)

        if missing_files:
            missing = True

        return missing, list(missing_files), list(missing_rads) ,nframes

    def _check_missing_photometry(self,rads):
        results = run_all_ccds(self.nccd,self._check_missing_photometry_per_ccd, None, rads) 
        missing_frames = {}
        nframes = []
        missing_rads = set()
        for i, result in results.items():
            missing = result[0] if result[0] else False
            missing_frames[i] = result[1]
            nframes.append(result[3])
            [missing_rads.add(item) for item in result[2]]
        return missing, missing_frames, missing_rads, nframes
    
    def _config_photometry(self, sky_calc_mode, const_sky_flag, const_sky_flux, const_sky_sdev):
        telescope_param = load_par_file(f"{self.target_dir}/param/param-tel.par")
        tel = SimpleNamespace(**telescope_param)

        apphot_param = load_par_file(f"{self.target_dir}/param/param-apphot.par") #skysep,wid,hbox,dcen,sigma_0
        app = SimpleNamespace(**apphot_param)

        ccd_param = load_par_file(f"{self.target_dir}/param//param-ccd.par") ##gain,readnoise,darknoise,adulo,aduhi from param/param-ccd.par
        ccd = SimpleNamespace(**ccd_param)

        config = PhotometryConfig(tid               = self.tid,
                                    rads            = self.rad_to_use,
                                    gain            = ccd.gain,
                                    read_noise      = ccd.readnoise,
                                    dark_noise      = ccd.darknoise,
                                    sky_sep         = app.sky_sep,
                                    sky_wid         = app.sky_wid,
                                    hbox            = app.hbox, #number of pixels around a given point to search for max flux aperture
                                    dcen            = app.dcen,  #step in pixels to move within hbox
                                    sigma_cut       = app.sigma_cut, #sigma clipping used for sky calculation
                                    adu_lo          = ccd.ADUlo,
                                    adu_hi          = ccd.ADUhi,
                                    sigma_0         = app.sigma_0, #scintillation coefficient
                                    altitude        = tel.altitude, #observatory altitude in meters (used for scintillation noise calculation)
                                    diameter        = tel.diameter ,#telescope diameter in cm (also used for scintillation noise calculation)
                                    global_sky_flag = app.global_sky_flag, #Use global sky calculation meaning calculate sky dont assume as constant
                                    sky_calc_mode   = sky_calc_mode, #Sky calculation mode (0=mean, 1=median, 2=mode)
                                    const_sky_flag  = const_sky_flag, #Use constant sky value
                                    const_sky_flux  = const_sky_flux,#Constant sky flux value
                                    const_sky_sdev  = const_sky_sdev,#Constant sky standard deviation
                                    #max_concurrent  = max_concurrent
                                )
        return config

    async def monitor_photometry_progress(self, header, interval=5):
        """
        Monitor photometry progress, calculating processing rate and remaining time.
        Updates progress in place using ANSI escape codes.
        
        Args:
            interval (int): Time in seconds to wait between progress checks
        """

        while True:
            initial_time = time.time()
            _, missing_files1, _ ,nframes = self._check_missing_photometry(self.rad_to_use)
            total_frames_per_ccd = np.array(nframes)

            await asyncio.sleep(interval)
            
            _, missing_files2, _, _ = self._check_missing_photometry(self.rad_to_use)
            current_time = time.time()
            complete = [True for _ in range(self.nccd)]  # Track if all CCDs are complete

            # Print header
            clear_output(wait=True)  # Clear the output completely
            print(header)
            print("=" * 80)
            print(f"{'CCD':<4} {'Progress':<22} {'Completed':<15} {'Rate':<15} {'Remaining (min)':<15}")
            print("-" * 80)            
            
            for (ccd_id, missing_files_per_ccd1), (_, missing_files_per_ccd2) in zip(missing_files1.items(), missing_files2.items()):
                # Current progress
                remaining_files = len(missing_files_per_ccd2)
                completed_files = total_frames_per_ccd[ccd_id] - remaining_files
                percentage = (completed_files / total_frames_per_ccd[ccd_id]) * 100
                percentage = 100 if np.isnan(percentage) else percentage
                
                # Calculate processing rate
                initial_remaining = len(missing_files_per_ccd1)
                files_processed = initial_remaining - remaining_files
                time_diff = current_time - initial_time
                rate = files_processed / time_diff if time_diff > 0 else 0  # files per second
                
                # Create progress bar
                bar_length = 20
                completed_blocks = int((percentage / 100) * bar_length)
                progress_bar = "█" * completed_blocks + "░" * (bar_length - completed_blocks)
                
                # Format rate as files/minute
                rate_per_minute = rate * 60
                
                # Determine remaining time string
                if not remaining_files:
                    remaining_str = "Complete"
                else:
                    complete[ccd_id] = False
                    remaining_str = "∞" if rate <= 0 else f"{(remaining_files / rate) / 60:.1f}"
                
                print(f"{ccd_id:<4} {progress_bar:<22} {completed_files:>5}/{total_frames_per_ccd[ccd_id]:<7} "
                    f"{rate_per_minute:>6.1f} f/min  {remaining_str:>12}")
                
            print("=" * 80)
            if all(complete):
                break            

    def read_photometry(self, ccd, rad):
        """
        Process photometry data for a single CCD.
        """
        frame_range = self.obslog[ccd][self.obslog[ccd]["OBJECT"] == self.target]
        first_frame = int(frame_range["FRAME#1"].iloc[0])
        last_frame = int(frame_range["FRAME#2"].iloc[0])
        apphot_directory = f"{self.target_dir}_{ccd}/apphot_{self.method}"

        all_frames = []
        
        for frame in range(first_frame, last_frame+1):
            filepath = f"{apphot_directory}/rad{str(rad)}/MCT{self.instid}{ccd}_{self.obsdate}{frame:04d}.dat"
            df = parse_dat_file(filepath)
            #print(frame)
            #print(result)
            #print(type(result))
            if df is not None:
                df['frame'] = frame
                all_frames.append(df)

        if all_frames:
            combined_df = pd.concat(all_frames, ignore_index=True)
            
            # Add CCD identifier
            combined_df['ccd'] = ccd
            
            return combined_df  
        else:
            return None

    def check_saturation_per_ccd(self, ccd, rad):
        saturation_threshold = 60000
        df = self.read_photometry(ccd=ccd, rad=rad)
        # Count the number of rows where peak > 60000 for this star ID
        saturation_cids = []
        saturation_zones = []
        flux_list = []
        frames_list = []
        median_list = []
        stop_processing = False
        for star_id in range(1,int(self.nstars)+1):
            if stop_processing:
                break  # Exit the loop completely
            flux = df[df["ID"] == star_id]["peak"]
            frames = np.array(list(range(len(df[df["ID"] == star_id]))))
            median = np.array(lc.moving_median(x=frames,y=flux,nsample=int(len(frames)/50)))
            typical_scatter = np.std(flux-median)
            saturation_threshold_per_star = saturation_threshold - typical_scatter #flux + typical scatter が60000を超えていたらsaturation zone
            saturation_zone = np.where(median > saturation_threshold_per_star)[0]
            count_above_threshold = (flux > saturation_threshold_per_star).sum()
            percentage_above_threshold = (count_above_threshold / len(frames)) * 100

            # If more than 5% of the rows are in saturation zone , add this star ID to the list
            if percentage_above_threshold > 5:
                saturation_cids.append(star_id)
                saturation_zones.append(saturation_zone)
                flux_list.append(flux)
                frames_list.append(frames)
                median_list.append(median)
            else:
                stop_processing = True  # Stop processing this CCD if a star is not saturated
        return saturation_cids, frames_list, flux_list, median_list, saturation_zones
        
    def check_saturation(self, rad):
        """Runs check_saturation_per_ccd in parallel across CCDs."""
        fig, ax = plt.subplots(self.nccd, 1, figsize=(15, 10))
        # Run in parallel
        print(f'>> Checking for saturation with rad={rad} ... (it may take a few seconds)')
        results = run_all_ccds(self.nccd,self.check_saturation_per_ccd, None, rad)
        print(f'## >> Done loading photometry data.')

        self.saturation_cids = [result[0] for _, result in results.items()]
        most_saturated_ccd = np.argmax([len(ids) for ids in self.saturation_cids])

        # Collect results and plot
        for i in range(self.nccd):
            if results and results[i] is not None:
                _, frames, flux, median, saturation_zone = results[i]
                #self.saturation_cids.append(saturation_cids_per_ccd)

            for j, _ in enumerate(flux):
                label = f"ID = {j+1}" if i == most_saturated_ccd else None
                ax[i].plot(frames[j],flux[j], label=label,zorder=1)
                ax[i].plot(frames[j], median[j], color="white", alpha=0.5, zorder=2)
                ax[i].scatter(saturation_zone[j], 
                             median[j][saturation_zone[j]],
                                color="red", alpha=0.5, marker=".", s=10, zorder=3)
                
            ax[i].set_title(f"CCD {i}")
            ax[i].set_ylim(0, 62000)
            #ax[i].set_xlabel("Frame")
            ax[i].set_ylabel("Peak")

        fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0), frameon=False, ncol=self.nstars)

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
        if self.tid is None:
            if tid is not None:
                self.tid = tid
            else:
                print("Provide TID")
                return
    
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
                cids.append(dimmest_star)   
            cids = [str(cid) for cid in cids]
            self.cids_list.append(cids) #if too many stars are saturated, there is a risk of not having the photometry for the star. need to add logic for this

    def create_photometry_per_ccd(self,ccd,cids):
        #print(f'>> CCD{ccd}')
        '''
        !perl scripts/auto_mklc.pl -date $date -obj $target\
            -ap_type $method -r1 $rad1 -r2 $rad2 -dr $drad -tid $tID -cids $cID
        バンドごとにcidが違う場合を考慮したいからこのコードを使わなかった

        errorの症状:cidが複数あるときに、一つ目の星のfluxしかカウントされていない
        しかし、comparisonとしての割り算には合算したfluxが使われているよう
            →argumentとしてのcidの読み込みはうまくいっている
        auto_mklc.plでもmklc_flux_collect_csv.plでも同じエラーが出るため問題はおそらくmklc_flux_collect_csv.plにある
        comparisonの順番を数字が大きい方からにすると治ったので、何かしらの読み込み時の挙動だと思われる
        ↑これは僕の勘違いで、実際にrmsを計算するのに使っているのはflux ratioであり、fluxの合計値ではない
        '''
        script_path = "/home/muscat/reduction_afphot/tools/afphot/script/mklc_flux_collect_csv-test.pl"
        initial_obj_dir = f"{self.target_dir}_{ccd}" 
        apdir = f"apphot_{self.method}"
        lstfile = f"list/object_ccd{ccd}.lst"
        for cid in cids:
            outfile = f"lcf_{self.instrument}_{self.bands[ccd]}_{self.target}_{self.obsdate}_t{self.tid}_c{cid.replace(' ','')}_r{int(self.rad1)}-{int(self.rad2)}.csv" # file name radius must be int
            if not os.path.isfile(f"{self.target_dir}/{outfile}"): #if the photometry file does not exist
                cmd = f"perl {script_path} -apdir {apdir} -list {lstfile} -r1 {int(self.rad1)} -r2 {int(self.rad2)} -dr {self.drad} -tid {self.tid} -cids {cid} -obj {self.target} -inst {self.instrument} -band {self.bands[ccd]} -date {self.obsdate}"
                #this command requires the cids to be separated by space
                subprocess.run(cmd, cwd=initial_obj_dir, shell=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
                outfile_path = os.path.join(initial_obj_dir,apdir,outfile)
                if os.path.isfile(outfile_path): #if the photometry file now exists
                    subprocess.run(f"mv {outfile_path} {self.target_dir}/{outfile}", shell=True)
                    print(f">> CCD {ccd} | Created photometry for cIDs:{cid}")
                else:
                    print(f">> CCD {ccd} | Failed to create photometry for cIDs:{cid}")
            else:
                print(f">> CCD {ccd} | Photometry for cIDs:{cid} already exists.")

    @time_keeper
    def create_photometry(self, given_cids=None):
        if given_cids:
            self.cids_list = given_cids
        ccd_specific_arg = [{"cids": cid} for cid in self.cids_list] # must be a list of dictionaries
        run_all_ccds(self.nccd,self.create_photometry_per_ccd,ccd_specific_arg)

class MuSCAT_PHOTOMETRY_OPTIMIZATION:
    def __init__(self,muscat_photometry):#ここは継承していれば引数で与えなくてもいい
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

        self.mask_status = {
            "raw"      : {"lower": np.full(self.nccd,-np.inf), "upper": np.full(self.nccd,np.inf)},
            "airmass"  : {"lower": np.full(self.nccd,-np.inf), "upper": np.full(self.nccd,np.inf)},
            "dx(pix)"  : {"lower": np.full(self.nccd,-np.inf), "upper": np.full(self.nccd,np.inf)},
            "dy(pix)"  : {"lower": np.full(self.nccd,-np.inf), "upper": np.full(self.nccd,np.inf)},
            "fwhm(pix)": {"lower": np.full(self.nccd,-np.inf), "upper": np.full(self.nccd,np.inf)},
            "peak(ADU)": {"lower": np.full(self.nccd,-np.inf), "upper": np.full(self.nccd,np.inf)},
        }

    def print_mask_status(self):
        # Create separate DataFrames for each CCD
        print(">> Current mask status:")
        masks = []
        for i in range(self.nccd):
            mask_df = pd.DataFrame({key: {"lower": data["lower"][i], "upper": data["upper"][i]} 
                            for key, data in self.mask_status.items()}).T
            mask_df.index.name = f"CCD_{i}"  # Name index to indicate CCD number
            masks.append(mask_df)

        # Example: Print all DataFrames
        for i, mask_df in enumerate(masks):
            print(f"{mask_df} \n")

    def _apply_mask(self,ccd):
        for j in range(len(self.phot[ccd])):  # Loop over sources, stars, or apertures
            condition = np.ones_like(self.phot[ccd][j]['GJD-2450000'], dtype=bool) # initialize mask
            for key in self.mask_status.keys():
                if key == "raw":
                    fcomp_key = f'flux_comp(r={self.ap[0]:.1f})'#this currently assumes the first aperture for raw flux cut
                    target_array = self.phot[ccd][j][fcomp_key]/self.phot[ccd][j]['exptime'] 
                    target_array /= np.median(target_array) #this is the normalized flux
                else:
                    target_array = self.phot[ccd][j][key]
                condition &= target_array > self.mask_status[key]["lower"][ccd]
                condition &= target_array < self.mask_status[key]["upper"][ccd]
            self.mask[ccd][j] = condition  

    def add_mask_per_ccd(self, key, ccd, lower=None, upper=None):
        if key not in self.mask_status:
            raise ValueError(f"Invalid key: {key}, must be one of {list(self.mask_status.keys())}")
        """Applies a mask to all elements in the dataset for a given CCD."""
        #update mask status
        if lower is not None:
            self.mask_status[key]["lower"][ccd] = lower[ccd]
        if upper is not None:
            self.mask_status[key]["upper"][ccd] = upper[ccd]
        self._apply_mask(ccd)
        print(f">> Added mask to {key} for CCD{ccd}")
        self.print_mask_status()

    def add_mask(self, key, lower=None, upper=None):
        if key not in self.mask_status:
            raise ValueError(f"Invalid key: {key}, must be one of {list(self.mask_status.keys())}")
        """Applies a mask to all elements in self.phot, handling lists of upper/lower bounds."""
        for i in range(self.nccd):  # Loop over CCDs
            if lower is not None:
                self.mask_status[key]["lower"][i] = lower[i] if isinstance(lower, list) else lower
            if upper is not None:
                self.mask_status[key]["upper"][i] = upper[i] if isinstance(upper, list) else upper
            self._apply_mask(i)
        print(f">> Added mask to {key}")
        self.print_mask_status()

    def preview_photometry(self, cid=0, ap=0, order=2, sigma_cut=3):
        fcomp_key = f'flux_comp(r={self.ap[ap]:.1f})' # Use the aperture given in the argument
        fig, ax = plt.subplots(6, self.nccd, figsize=(16, 20), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1, 2, 2]})
        print(">> Performing preliminary outlier detection ...")
        print(f"## >> Fitting with polynomials (order = {order}) and cutting {sigma_cut} sigma outliers ...")
        for i in range(self.nccd):
            mask = self.mask[i][cid]
            phot_j = self.phot[i][cid][mask]
            exptime = phot_j['exptime']
            gjd_vals = phot_j['GJD-2450000']
            raw_norm = phot_j[fcomp_key] / exptime
            raw_norm /= np.median(raw_norm)
            fcomp_data = phot_j[fcomp_key] #コンパリゾンのフラックス

            ye = np.sqrt(fcomp_data) / exptime / np.median(fcomp_data / exptime)

            if len(ye) < 0:
                print(">> No data points to plot.")
                return

            p, tcut, ycut, yecut, keep_mask = lc.outcut_polyfit(gjd_vals, raw_norm, ye, order, sigma_cut)

            three_sigma_outliers = ~keep_mask

            ymax = np.max(raw_norm[keep_mask])*1.01
            ymin = np.min(raw_norm[keep_mask])*0.99

            outlier_for_plot = np.clip(raw_norm[three_sigma_outliers], ymin, ymax)

            ax[0, i].plot(gjd_vals[three_sigma_outliers], outlier_for_plot, 'x', c="gray")
            ax[1, i].plot(gjd_vals[three_sigma_outliers], phot_j['airmass'][three_sigma_outliers], 'x', c="gray", label=f"{sigma_cut}-sigma outliers")
            ax[2, i].plot(gjd_vals[three_sigma_outliers], phot_j['dx(pix)'][three_sigma_outliers], 'x', c="gray")
            ax[3, i].plot(gjd_vals[three_sigma_outliers], phot_j['dy(pix)'][three_sigma_outliers], 'x', c="gray")
            ax[4, i].plot(gjd_vals[three_sigma_outliers], phot_j['fwhm(pix)'][three_sigma_outliers], 'x', c="gray")
            ax[5, i].plot(gjd_vals[three_sigma_outliers], phot_j['peak(ADU)'][three_sigma_outliers], 'x', c="gray")
        
            print(f">> CCD {i} | Ploting the photometry data for cID:{self.cids_list[i][cid]}, ap:{self.ap[ap]}")
            ax[0, i].plot(gjd_vals[keep_mask], raw_norm[keep_mask], '.', c="k")
            ax[1, i].plot(gjd_vals[keep_mask], phot_j['airmass'][keep_mask], '.', c="gray")
            ax[2, i].plot(gjd_vals[keep_mask], phot_j['dx(pix)'][keep_mask], '.', c="orange")
            ax[3, i].plot(gjd_vals[keep_mask], phot_j['dy(pix)'][keep_mask], '.', c="orange")
            ax[4, i].plot(gjd_vals[keep_mask], phot_j['fwhm(pix)'][keep_mask], '.', c="blue")
            ax[5, i].plot(gjd_vals[keep_mask], phot_j['peak(ADU)'][keep_mask], '.', c="red")

            for key_index, key in enumerate(self.mask_status.keys()):  # Fixed iteration
                lower_val = self.mask_status[key]["lower"][i]
                upper_val = self.mask_status[key]["upper"][i]

                if not np.isinf(lower_val):  # Plot only if lower_val is finite
                    ax[key_index, i].plot(gjd_vals,np.full(len(gjd_vals), lower_val),c="gray")

                if not np.isinf(upper_val):  # Plot only if upper_val is finite
                    ax[key_index, i].plot(gjd_vals,np.full(len(gjd_vals), upper_val),c="gray")

        # Set labels only on the first column
        ax[0, 0].set_ylabel('Relative flux')
        ax[1, 0].set_ylabel('Airmass')
        ax[2, 0].set_ylabel('dX')
        ax[3, 0].set_ylabel('dY')
        ax[4, 0].set_ylabel('FWHM')
        ax[5, 0].set_ylabel('Peak of the brightest star (ADU)')

        # Set common x-axis label
        for i in range(self.nccd):
            ax[-1, i].set_xlabel('GJD - 2450000')

        plt.tight_layout(h_pad=0)  # Remove spacing between rows
        plt.legend()
        plt.show()

    def outlier_cut(self, sigma_cut=3, order=2, plot=True):
        """Performs outlier detection using polynomial fitting and sigma clipping in parallel."""
        
        print(f">> Fitting with polynomials (order = {order}) and cutting {sigma_cut} sigma outliers ...")
        
        # Run the processing for each CCD in parallel
        results = run_all_ccds(
            self.nccd,
            self.outlier_cut_per_ccd,
            None,  # No CCD-specific args
            sigma_cut,
            order
        )
        
        if results is None:
            print(">> No data processed successfully.")
            return
        
        # Initialize class attributes to store results
        self.index = []
        self.ndata_diff = []
        self.rms = []
        self.min_rms_idx_list = []
        
        # Process results from parallel execution
        for i in range(self.nccd):
            if results.get(i) is None:
                print(f">> CCD {i}: Processing failed or no data available.")
                # Add empty placeholders for this CCD
                self.index.append([])
                self.ndata_diff.append(np.array([]))
                self.rms.append(np.array([]))
                self.min_rms_idx_list.append((0, 0))  # Default index
                continue
            
            # Extract results for this CCD
            ccd_results = results[i]
            self.index.append(ccd_results['index'])
            self.ndata_diff.append(ccd_results['ndata_diff'])
            self.rms.append(ccd_results['rms'])
            self.min_rms_idx_list.append(ccd_results['min_rms_idx'])
        
        # Store best candidate values
        self.cIDs_best = [self.cids_list[i][item[0]] for i, item in enumerate(self.min_rms_idx_list)]
        self.cIDs_best_idx = [item[0] for item in self.min_rms_idx_list]
        self.ap_best = [self.ap[item[1]] for item in self.min_rms_idx_list]
        self.ap_best_idx = [item[1] for item in self.min_rms_idx_list]

        if plot:
            self.plot_outlier_cut_results()
    

    def outlier_cut_per_ccd(self, i, sigma_cut, order):
        """
        Process outlier detection for a single CCD
        """
        print(f"## >> CCD {i} | Computing outliers ...")
        
        n_cids = len(self.cids_list[i])
        n_ap = len(self.ap)
        
        ndata_diff = np.zeros((n_cids, n_ap))
        rms = np.zeros((n_cids, n_ap))
        index = []
        
        for j in range(n_cids):
            index.append([])
            phot_j = self.phot[i][j]
            exptime = phot_j['exptime']
            gjd_vals = phot_j['GJD-2450000']
            mask = self.mask[i][j]
            
            fcomp_keys = [f'flux_comp(r={self.ap[k]:.1f})' for k in range(n_ap)]
            fcomp_data = np.array([phot_j[fk] for fk in fcomp_keys])  # cid=jの総フラックス（in ADU?）のarrayをapごとに作成
            
            raw_norm = (fcomp_data / exptime) / np.median(fcomp_data / exptime, axis=1, keepdims=True)  # apごとのnormalized flux
            ndata_init = fcomp_data.shape[1]
            
            ye = np.sqrt(fcomp_data[:, mask]) / exptime[mask] / np.median(fcomp_data / exptime, axis=1, keepdims=True)
            # arrayにしたときに次元が合わなくなるからここでマスクをかけている
            
            for k in range(n_ap):
                if len(ye[k]) > 0:  # only perform outlier detection if there are data points
                    p, tcut, ycut, yecut, keep_mask = lc.outcut_polyfit(gjd_vals[mask], raw_norm[k][mask], ye[k], order, sigma_cut)
                    index[j].append(np.isin(gjd_vals, gjd_vals[mask][keep_mask]))  # indexというのは超最終的なmask tcutsに含まれなかったらfalse
                    ndata_final = len(tcut)
                else:
                    index[j].append(np.zeros_like(gjd_vals[mask], dtype=bool))
                    ndata_final = 0
                
                ndata_diff[j, k] = ndata_final - ndata_init
                
                fin_flux = phot_j[f"flux(r={self.ap[k]:.1f})"][index[j][k]]
                
                if len(fin_flux) > 1:
                    diff = np.diff(fin_flux)
                    rms[j, k] = np.std(diff) if np.std(diff) > 0 else np.inf
                else:
                    rms[j, k] = np.inf
        
        # ここまでで全てcids, apに対してoutlier detectionが終わった per ccd （1ccdのrmsとdatapointsの行列完成）
        min_rms_idx = np.unravel_index(np.argmin(rms, axis=None), rms.shape)  # 最小のrmsのindexを取得
        
        print(f"## >> CCD {i} | Complete")
        
        # Return all the computed data for this CCD
        return {
            'index': index,
            'ndata_diff': ndata_diff,
            'rms': rms,
            'min_rms_idx': min_rms_idx
        }

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
            axes[i, 0].set_xticklabels([f"{int(self.ap[k])}" for k in range(len(self.ap))])
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

        print(">> Plotting results")
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
    
    async def iterate_optimization(self):
        min_rms_list = [[1 for _ in range(self.nccd)]] # Initialize with RMS = 1 to enter loop
        min_rms_list.append([np.array(self.rms[ccd])[index] for ccd, index in enumerate(self.min_rms_idx_list)])
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
        threshold = 0.00001 #threshold for rms improvement (0.001%=10ppm)

        while any(previous - latest > threshold for previous, latest in zip(min_rms_list[-2], min_rms_list[-1])): #while latest rms keeps improving by more than 10ppm
            print(f">> Returning to photometry for aperture optimization... (Iteration: {len(min_rms_list)-1})")

            photometry = MuSCAT_PHOTOMETRY(parent=self)
            await photometry.run_apphot(nstars=self.nstars, rad1=rad1, rad2=rad2, drad=drad, method="mapping")
            photometry.cids_list = reselected_cids
            photometry.create_photometry()

            mask_status = self.mask_status
            self.__init__(photometry)
            self.mask_status = mask_status
            [self._apply_mask(i) for i in range(self.nccd)] # reapply mask
            self.outlier_cut(plot=False)
            min_rms_list.append([np.array(self.rms[ccd])[index] for ccd, index in enumerate(self.min_rms_idx_list)])

            if any(idx in {self.ap[0]} for idx in self.ap_best):
                rad1 -= 1
            elif any(idx in {self.ap[-1]} for idx in self.ap_best):
                rad2 += 1
            else:
                rad1 -= 1
                rad2 += 1

            previous_rms = [f"{rms:.3g}" for rms in min_rms_list[-2]]
            latest_rms = [f"{rms:.3g}" for rms in min_rms_list[-1]]

            print(f"Minimum rms: {previous_rms} \n          -> {latest_rms}")
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
        plt.savefig(f"/home/muscat/reduction_afphot/notebooks/general/{self.target}/{outfile}",bbox_inches='tight',pad_inches=0.1)
        plt.show()
    
    def barycentric_correction(self,ccd):
        jd = self.phot[ccd][self.cIDs_best_idx[ccd]]['GJD-2450000']
        mask = [self.index[ccd][self.cIDs_best_idx[ccd]][self.ap_best_idx[ccd]]]
        masked_jd = np.array(jd[mask] + 2450000)

        n=200 #number of data points to process at a time due to barrycorrpy request constraints

        bjd = np.array([])
        for j in range(int(len(masked_jd) / n)+1):
            index1 = j*n
            index2 = min((j+1)*n, len(jd))
            
            jd_tmp = jd[index1:index2]
            
            kwargs = {
                'jd_utc': jd_tmp,
                'ra': self.ra,
                'dec': self.dec,
            }

            bjd_tmp = barycorr.utc2bjd(**kwargs)
            bjd = np.append(bjd, bjd_tmp)
        return bjd
    
    def save_lc_per_ccd(self,ccd):
        f_key = f'flux(r={self.ap_best[ccd]})'
        e_key = f'err(r={self.ap_best[ccd]})'
        outfile = f"{self.target}_{self.obsdate}_{self.instrument}_{self.bands[ccd]}_c{self.cIDs_best[ccd]}_r{int(self.ap_best)}.csv"
        mask = [self.index[ccd][self.cIDs_best_idx[ccd]][self.ap_best_idx[ccd]]]
        print(outfile)
        bjd = self.barycentric_correction(ccd)
        out_array = np.array( (bjd,
                            np.array(self.phot[ccd][self.cIDs_best_idx[ccd]][f_key][mask]),
                            np.array(self.phot[ccd][self.cIDs_best_idx[ccd]][e_key][mask]),
                            np.array(self.phot[ccd][self.cIDs_best_idx[ccd]]['airmass'][mask]),
                            np.array(self.phot[ccd][self.cIDs_best_idx[ccd]]['dx(pix)'][mask]),
                            np.array(self.phot[ccd][self.cIDs_best_idx[ccd]]['dy(pix)'][mask]),
                            np.array(self.phot[ccd][self.cIDs_best_idx[ccd]]['fwhm(pix)'][mask]),
                            np.array(self.phot[ccd][self.cIDs_best_idx[ccd]]['peak(ADU)'][mask]),
                            )) 
        np.savetxt(outfile, out_array.T, delimiter=',', fmt='%.6f,%.5f,%.5f,%.4f,%.2f,%.2f,%.2f,%d',
                header='BJD_TDB,Flux,Err,Airmass,DX(pix),DY(pix),FWHM(pix),Peak(ADU)', comments='')
    
    def save_lc(self):
        run_all_ccds(self.nccd,self.save_lc_per_ccd)

