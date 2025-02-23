import numpy as np
from astropy.io import fits
import argparse
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import os
import asyncio
import aiofiles
from types import SimpleNamespace
import re
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Queue
import time


@dataclass
class PhotometryConfig:
    tid:            int #target star id
    rads:           np.ndarray                 #list of aperture radius
    gain:           float #ccd gain (e/ADU) 
    read_noise:     float 
    dark_noise:     float 
    sky_sep:        int  =50
    sky_wid:        int  =40
    hbox:           int  =0 #number of pixels around a given point to search for max flux aperture
    dcen:           float=0.1 #step in pixels to move within hbox
    sigma_cut:      int  =3#sigma clipping used for sky calculation
    adu_lo:         int  =1000
    adu_hi:         int  =65000#16bit adu limit 2^16-1=65,535,
    sigma_0:        float=0.064#scintillation coefficient
    altitude:       int  =1013#observatory altitude in meters (used for scintillation noise calculation)
    diameter:       int  =61#telescope diameter in cm (also used for scintillation noise calculation)
    sky_calc_mode:  int  =0 #Sky calculation mode (0=mean, 1=median, 2=mode)
    global_sky_flag:int  =0 #Use global sky calculation meaning calculate sky dont assume as constant
    const_sky_flag: int  =0 #Use constant sky value
    const_sky_flux: float=0.0#Constant sky flux value
    const_sky_sdev: float=0.0#Constant sky standard deviation
    method:         str="mapping"#apphot method either mapping or centroid
    #max_concurrent: int=20


'''
  // Parameters for aperture and sky radii
  // r1: minimum aperture radius
  // r2: maximum aperture radius
  // dr: step size of the aperture radius
  // hbox: half box size for searching center
  // dcen: step size for searching center
  // sky_sep: separation of sky annulus from the center
  // sky_wid: width of the sky annulus 
  // sigma_cut: outlier cut threshold for calculating sky value

    parser.add_argument('--sky_calc_mode', type=int, default=0, 
                        help='Sky calculation mode (0=mean, 1=median, 2=mode)')
    parser.add_argument('--global_sky_flag', type=int, default=0, 
                        help='Use global sky calculation')
    parser.add_argument('--const_sky_flag', type=int, default=0,
                        help='Use constant sky value')
    parser.add_argument('--const_sky_flux', type=float, default=0.0,
                        help='Constant sky flux value')
    parser.add_argument('--const_sky_sdev', type=float, default=0.0,
                        help='Constant sky standard deviation')
    parser.add_argument('--altitude', type=float, default=1013.0,
                        help='Observatory altitude [m]')
    parser.add_argument('--diameter', type=float, default=61.0,
                        help='Telescope diameter [cm]')
    parser.add_argument('--exptime', type=float, default=60.0,
                        help='Exposure time [s]')
    parser.add_argument('--sigma_0', type=float, default=0.064,
                        help='Scintillation coefficient')
    parser.add_argument('--airmass', type=float, default=1.0,
                        help='Airmass')

  this aperture photometry algorithm searches for the optimal center position of the star by looping over a 2*hbox x 2*hbox box
    and calculates the flux by summing the pixel values within the aperture and subtracting the sky background

  does that for all stars identified in the starlist file  
  i shoudl add to this class a functionality that allows a loop over given stellar radii r1 & r2
'''
class ApPhotometry:
    def __init__(self,
                 frame, 
                 starlist,
                 config=PhotometryConfig,
                 semaphore = asyncio.Semaphore(1000)
                 ):
        
        self.frame = frame
        self.x, self.y = starlist

        self.tid = config.tid
        self.rads = config.rads
        self.gain = config.gain
        self.read_noise = config.read_noise
        self.dark_noise = config.dark_noise
        self.sky_sep = config.sky_sep
        self.sky_wid = config.sky_wid
        self.hbox = config.hbox
        self.dcen = config.dcen
        self.sigma_cut = config.sigma_cut
        self.adu_lo = config.adu_lo
        self.adu_hi = config.adu_hi
        self.sigma_0 = config.sigma_0
        self.altitude = config.altitude
        self.diameter = config.diameter
        self.sky_calc_mode = config.sky_calc_mode
        self.global_sky_flag = config.global_sky_flag
        self.const_sky_flag = config.const_sky_flag
        self.const_sky_flux = config.const_sky_flux
        self.const_sky_sdev = config.const_sky_sdev
        self.method = config.method

        self.version = "3.0.0"
        self.semaphore = semaphore

    '''
    def set_frame(self, frame, starlist):
        self.frame = frame
        self.x, self.y = starlist
    '''
    def pixel_fraction(self, x: int, y: int, xcen: float, ycen: float, r: float) -> float:
        """Calculate the fraction of a pixel that falls within the aperture."""
        nsubpix = 100
        subpix = 1.0/nsubpix
        sumsubpix = 0
        
        for ix in range(nsubpix):
            for iy in range(nsubpix):
                x1 = x - 0.5 + ix*subpix
                y1 = y - 0.5 + iy*subpix
                sep = np.sqrt((x1 - xcen)**2 + (y1 - ycen)**2)
                if sep <= r:
                    sumsubpix += 1
                    
        return float(sumsubpix) / (nsubpix * nsubpix) #this function is not called

    def calc_sky(self, sky: np.ndarray, sigma_cut: float) -> Tuple[float, float]:
        """Calculate sky background using sigma clipping."""
        sky = np.array(sky)
        while True:
            mean = np.mean(sky)
            std = np.std(sky)
            mask = np.abs(sky - mean) <= sigma_cut * std
            if np.all(mask):
                break
            sky = sky[mask]
        return mean, std

    def calc_median_sky(self, sky: np.ndarray, sigma_cut: float) -> Tuple[float, float]:
        """Calculate sky background using median and sigma clipping."""
        sky = np.array(sky)
        while True:
            median = np.median(sky)
            std = np.std(sky)
            mask = np.abs(sky - median) <= sigma_cut * std
            if np.all(mask):
                break
            sky = sky[mask]
        return median, std

    def calc_mode_sky(self, sky: np.ndarray) -> Tuple[float, float]:
        """Calculate mode of sky background."""
        from scipy import stats
        mode = stats.mode(sky.astype(int))[0][0]
        return float(mode), 0.0

    def process_image(self):
        """Main processing function for aperture photometry."""
        #print(f"## apphot version {self.version} ##")
        dirs = self.frame.split("/") #-> obsdate/target_ccd/df/frame_df.fits
        outfile =f"{dirs[-1][:-8]}.dat"  #-> target_ccd/apphot_method/rad/frame.dat
        outpath=f"{dirs[0]}/{dirs[1]}/apphot_{self.method}_test"

        os.makedirs(outpath, exist_ok=True)

        with fits.open(self.frame) as hdul:
            image_data = hdul[0].data
            image_header = hdul[0].header

        #image_header, image_data = infile

        # Read FITS image
        exptime = image_header["EXPTIME"]
        airmass = image_header["AIRMASS"]
        mjd_strt = image_header["MJD-STRT"]
        jd_2450000_strt = mjd_strt - 49999.5 #jd = mjd - 2450000.5
        jd_2450000_mid = jd_2450000_strt + (exptime/2/86400)

        # Read star positions
        nstars = len(self.x)
        #print(f"# nstars = {nstars}")
        
        if nstars < 2:
            print("Exit!")
            return
            
        # Calculate scintillation noise
        sigma_scin = (self.sigma_0 * (airmass ** (7.0/4.0)) / 
                     (self.diameter ** (2.0/3.0)) / np.sqrt(exptime) * 
                     np.exp(-self.altitude / 8000.0))
        
        # Set up aperture parameters
        # this is the initial subimage to work with before aperture
        # at this point the center of the star is not known so hbox is the buffer given to search for the center
        # we eventually loop over 2 x hbox to find the center defined as the position that maximizes the flux
        half_cameo = int(self.hbox + self.sky_sep + self.sky_wid)
        
        results = [[] for _ in self.rads]
        # Process each star
        for starid in range(nstars):
            #if starid == self.tid - 1:
            #    continue
                
            x, y = self.x[starid], self.y[starid]
            #print(f"coord of star{starid+1}:{x,y}")
            
            # Extract sub-image (cameo) / square containing sky annulus
            x_min = max(int(x) - half_cameo, 0) #need to take care of edge stars constraining x and y minmax to within 0 and image_data.shape[0/1] - 1
            x_max = min(int(x) + half_cameo + 1, image_data.shape[0] - 1)
            y_min = max(int(y) - half_cameo, 0)
            y_max = min(int(y) + half_cameo + 1, image_data.shape[1] - 1)
            
            cameo = image_data[y_min:y_max, x_min:x_max].copy() 

            #"cameo" refers to a small sub-image or cutout centered around a star or object of interest, 
            # similar to how a cameo brooch features a small portrait or miniature scene.
            #カモフラージュのcamoとは別物勘違い
            
            # Handle bad pixels
            bad_mask = (cameo == 0) | (cameo < self.adu_lo) | (cameo > self.adu_hi)
            if np.any(bad_mask):
                for i, j in np.argwhere(bad_mask):
                    neighbors = cameo[max(0,i-1):i+2, max(0,j-1):j+2]
                    good_neighbors = neighbors[~((neighbors == 0) | 
                                              (neighbors < self.adu_lo) | 
                                              (neighbors > self.adu_hi))]
                    if len(good_neighbors) > 0:
                        cameo[i,j] = np.mean(good_neighbors) #サチってるピクセルを周囲の平均値で置き換える
            
            # Find optimal position and calculate flux
            max_flux = [0 for _ in self.rads]
            max_sky = [0 for _ in self.rads]
            max_sky_std = [0 for _ in self.rads]
            max_pos = [(0, 0) for _ in self.rads]
            aper_masks = [[] for _ in self.rads]

            # loop for xy center in (2*hbox)*(2*hbox) box and search the maximum flux
            for dx in np.arange(-self.hbox, self.hbox + self.dcen, self.dcen):
                for dy in np.arange(-self.hbox, self.hbox + self.dcen, self.dcen):
                    xcen = x + dx
                    ycen = y + dy
                    #print("xcoord,y")
                    #print(xcen)
                    #print(ycen)
                    # Calculate distances from center
                    yy, xx = np.indices(cameo.shape)
                    '''
                    # If cameo.shape is (3,4) - meaning 3 rows and 4 columns
                    # np.indices((3,4)) returns two arrays:

                    # First array (yy) - row indices:
                    # [[0 0 0 0]
                    #  [1 1 1 1]
                    #  [2 2 2 2]]

                    # Second array (xx) - column indices:
                    # [[0 1 2 3]
                    #  [0 1 2 3]
                    #  [0 1 2 3]]
                    '''
                    xx = xx + x_min #because this is a subimage, add the min value to count the number of pixels from the edge of the actual image
                    yy = yy + y_min
                    r = np.sqrt((xx - xcen)**2 + (yy - ycen)**2) #distance from center in pixels has the dimensions of 
                    #print(np.min(r))

                    #plt.imshow(r, cmap="coolwarm", aspect="auto")
                    #plt.show()
                    #print(r)
                    #takes the same shape as cameo, with each element representing the distance from the center of the star
                    sky_mask = (r >= self.sky_sep) & (r <= self.sky_sep + self.sky_wid) #pixels within the sky annulus
                        
                        # Calculate sky
                    if not self.global_sky_flag and not self.const_sky_flag:
                        sky_pixels = cameo[sky_mask]
                        if self.sky_calc_mode == 1:
                            sky, sky_std = self.calc_median_sky(sky_pixels, self.sigma_cut)
                        elif self.sky_calc_mode == 2:
                            sky, sky_std = self.calc_mode_sky(sky_pixels)
                        else:
                            sky, sky_std = self.calc_sky(sky_pixels, self.sigma_cut)
                    else:
                        sky = self.const_sky_flux
                        sky_std = self.const_sky_sdev
                        
                    for rad_index, rad in enumerate(self.rads):
                        # Define aperture and sky annulus
                        aper_mask = r <= rad #pixels within the aperture (it means that up until this point, the pixels are still in the aperture)
                        #print(ap_r)
                        #print(np.unique(aper_mask))
    
                        # Calculate flux
                        flux = np.sum(cameo[aper_mask]) - sky * np.sum(aper_mask)

                        if flux > max_flux[rad_index]:
                            max_flux[rad_index] = flux
                            max_sky[rad_index] = sky
                            max_sky_std[rad_index] = sky_std
                            max_pos[rad_index] = (xcen, ycen)
                            aper_masks[rad_index] = aper_mask

            # Calculate noise and SNR
            '''
            Note on the uncertainty in mean sky level
            ##the uncertainty affects all pixels in the aperture) (npix)
            ##the uncertainty in subtracted sky level estimated from pixels in sky annulus (npix * max_sky_std**2)/n_sky_pixels
            ####affects all pixels in the aperture hence npix but divided by n_sky_pixels because more sky pixels reduce uncertainty
            '''
            outputs = [] #outputs for different rads

            for rad_index, rad in enumerate(self.rads):
                flux = max_flux[rad_index]
                sky = max_sky[rad_index]
                sky_std = max_sky_std[rad_index]
                pos = max_pos[rad_index]
                aper_mask = aper_masks[rad_index]
                n_pix= np.sum(aper_mask)

                noise = np.sqrt(flux * self.gain + #flux*gain -> shot noise in electrons
                            n_pix * sky_std**2 * self.gain**2 + #uncertainty in pixel-to-pixel variation in the sky background in electrons (inherent)
                            n_pix**2 / len(sky_pixels) * sky_std**2 * self.gain**2 + #uncertainty in mean sky level in electrons 
                            n_pix * self.read_noise**2 + #read noise in electrons
                            n_pix * self.dark_noise**2 * exptime**2 + #dark current noise in electrons
                            (sigma_scin * flux * self.gain)**2) / self.gain #scintillation noise in electrons and division by gain to convert to ADU
                
                snr = flux / noise
                
                # Calculate FWHM

                peak_flux = np.max(cameo[aper_mask] - sky)
                hm = peak_flux / 2.0
                fwhm_mask = (cameo[aper_mask] - sky > hm)
                if np.any(fwhm_mask):
                    fwhm = 2 * np.mean(r[aper_mask][fwhm_mask])
                else:
                    fwhm = 0
                    
                output = (
                    f"# gjd - 2450000 = {jd_2450000_mid}\n\n"
                    f"## apphot version {self.version}##\n\n"
                    f"# nstars = {nstars}\n"
                    f"# filename = {self.frame}\n"
                    f"# gain = {self.gain}\n"
                    f"# readout_noise = {self.read_noise}\n"
                    f"# dark_noise = {self.dark_noise}\n"
                    f"# ADU_range = {self.adu_lo} {self.adu_hi}\n"
                    f"# r = {rad}\n"
                    f"# hbox = {self.hbox}\n"
                    f"# dcen = {self.dcen}\n"
                    f"# sigma_cut = {self.sigma_cut}\n"
                    f"# altitude = {self.altitude}\n"
                    f"# Diameter = {self.diameter}\n"
                    f"# exptime = {exptime}\n"
                    f"# sigma_0 = {self.sigma_0}\n"
                    f"# airmass = {airmass}\n\n"
                    f"# global_sky_flag = {self.global_sky_flag}\n"
                    f"# const_sky_flag = {self.const_sky_flag}\n"
                    f"# sky_calc_mode = {self.sky_calc_mode}\n"
                    f"# sky_sep = {self.sky_wid}\n"
                    f"# sky_wid = {self.sky_wid}\n"
                    f"# ID xcen ycen nflux flux err sky sky_sdev SNR nbadpix fwhm peak\n"
                )

                results[rad_index].append({
                    'id': starid+1,
                    'xcen': pos[0],
                    'ycen': pos[1],
                    'flux': flux,
                    'sky': sky,
                    'sky_std': sky_std,
                    'noise': noise,
                    'snr': snr,
                    'fwhm': fwhm,
                    'peak': peak_flux + sky
                })
                
                for result in results[rad_index]:
                    output += (
                        f"{result['id']:.0f} {result['xcen']:.3f} {result['ycen']:.3f} "
                        f"{result['flux']:.2f} {result['flux']:.2f} {result['noise']:.2f} "
                        f"{result['sky']:.2f} {result['sky_std']:.2f} {result['snr']:.2f} "
                        f"0 {result['fwhm']:.2f} {result['peak']:.1f}\n"
                    )

                outputs.append({"data": output, "path": f"{outpath}/rad{rad}/{outfile}"})

        return outputs

    async def write_results(self, outputs):
        """Write multiple results to files asynchronously."""
        #async with self.semaphore:  # Use semaphore for control
        tasks = [self._write_single_file(output["path"], output["data"]) for output in outputs]
        await asyncio.gather(*tasks)

    async def _write_single_file(self, path, data):
        """Regular blocking file write"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        async with self.semaphore:  # Use semaphore for control
            with open(path, "w") as f:
                f.write(data)

    async def photometry_routine(self):
        """Runs processing in a thread and writes asynchronously."""
        routine_start = time.time()
        try:
            #print("Starting photometry_routine")
            outputs = await asyncio.to_thread(self.process_image)
            write_start = time.time()
            await self.write_results(outputs)
            print(f"Full routine completed in {time.time() - routine_start:.2f}s "
                f"(writing took {time.time() - write_start:.2f}s)")
        except Exception as e:
            print(f"Error in photometry routine: {e}")
            raise
        '''
        try:
            #print("Starting photometry_routine")
            outputs = await asyncio.to_thread(self.process_image)
            #print("process_image completed, writing results")
            await self.write_results(outputs)
            #print("write_results completed")
            return outputs  # Return outputs for error checking
        except Exception as e:
            #print(f"Error in photometry routine: {e}")
            raise
        '''

    @classmethod
    async def process_multiple_images(cls, frames, starlists, config: PhotometryConfig, semaphore):
        instances = [cls(frame, starlist, config, semaphore) for frame, starlist in zip(frames, starlists)]
        tasks = [instance.photometry_routine() for instance in instances]
        await asyncio.gather(*tasks, return_exceptions=True)
        '''
        start_time = time.time()



        results = []
        try:
            #print(f"Task {i} status before gather: {task._state}")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            print(f"Gather completed after {time.time() - start_time:.2f} seconds")
            return results
        except Exception as e:
            print(f"Error during photometry processing: {e}")
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise
        finally:
            return results  # Return results even if partial
        

        tasks = [
            asyncio.create_task(
                cls(frame, starlist, config, semaphore).photometry_routine(),
                name=f"task_{i}"  # Add names for debugging
            )
            for i, (frame, starlist) in enumerate(zip(frames, starlists))
        ]
        
        print(f"Created {len(tasks)} tasks")
        
        # Monitor completion without gathering results
        remaining = set(tasks)
        while remaining:
            done, remaining = await asyncio.wait(
                remaining,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Handle completed tasks immediately to free memory
            for task in done:
                try:
                    # Check for exceptions but don't keep result
                    await task
                except Exception as e:
                    print(f"Task {task.get_name()} failed: {e}")
            
            # Print progress every 100 completions
            if len(remaining) % 100 == 0:
                print(f"Remaining tasks: {len(remaining)}")

        print("All tasks completed")
        '''


    @classmethod
    def process_ccd_wrapper(cls, frames, starlists, config):
        """Wrapper function to run async code in a separate process."""
        # Create new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create semaphore for this process
        semaphore = asyncio.Semaphore(10)

        try: # why use this and not context manager
            # Run the async processing
            return loop.run_until_complete(
                cls.process_multiple_images(frames, starlists, config, semaphore)
            )
        finally:
            loop.close()

    @classmethod
    def process_all_ccds(cls, frames_list, starlists_list, config: PhotometryConfig):
        """Main entry point for multiprocessing."""
        ncores = min(len(frames_list),os.cpu_count())
        #print(f"Starting photometry with {num_ccds} cores...")
        
        # Create partial function with class method
        process_ccd = partial(cls.process_ccd_wrapper)
        
        # Create process pool with one process per CCD
        with ProcessPoolExecutor(max_workers=ncores) as executor:
            # Submit all CCDs for processing
            
            futures = [
                executor.submit(process_ccd, frames, starlists, config)
                for frames, starlists in zip(frames_list,starlists_list)
            ]
        
            # Wait for all processes to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in process: {e}")