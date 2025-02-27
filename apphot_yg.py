import numpy as np
from astropy.io import fits
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import os
import asyncio
from types import SimpleNamespace
import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

from functools import partial
from multiprocessing import Queue
import time
import functools


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


class ApPhotometry:
    def __init__(self,
                 frame, 
                 starlist,
                 config=PhotometryConfig,
                 semaphore = asyncio.Semaphore(1000),
                 thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
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
        self.thread_pool = thread_pool

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
        """Main processing function for aperture photometry.
 
        #information that needs to be supplied externally
        fits file
        gain,readnoise,darknoise,adulo,aduhi from param/param-ccd.par
        telsecope diameter and altitude from param-tel.par

        geoparamから
        $dx,$dy,$a,$b,$c,$d,$rms
        の順で呼び出して、

        $x = $dx + $a*$x0[$i] + $b*$y0[$i];
        $y = $dy + $c*$x0[$i] + $d*$y0[$i];

        を計算する
        x0 y0はなんだ
        x0 y0 is the coordinates of the stars -in the ref frame- in pixels
        x y is the coordinates for the corresponding star in the object frame
        
        starlistは各行に各frameのxyが入っている
        starlistを作る段階でnstarsの情報を上げなければいけない

        """
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
                    'noise': noise,
                    'sky': sky,
                    'sky_std': sky_std,
                    'snr': snr,
                    'nbadpix': np.sum(bad_mask),
                    'fwhm': fwhm,
                    'peak': peak_flux + sky
                })
                
                for result in results[rad_index]:
                    output += (
                        f"{result['id']:.0f} {result['xcen']:.3f} {result['ycen']:.3f} "
                        f"{result['flux']:.2f}{result['noise']:.2f} "
                        f"{result['sky']:.2f} {result['sky_std']:.2f} {result['snr']:.2f} "
                        f"{result['nbadpix']}" f"0 {result['fwhm']:.2f} {result['peak']:.1f}\n"
                    )

                outputs.append({"data": output, "path": f"{outpath}/rad{rad}/{outfile}"})

        return outputs

    async def write_results(self, outputs):
        """Write multiple results to files asynchronously."""
        tasks = [self._write_single_file(output["path"], output["data"]) for output in outputs]
        await asyncio.gather(*tasks)

    async def _write_single_file(self, path, data):
        """Regular blocking file write"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(data)

    async def _run_in_limited_thread(self, func, *args, **kwargs):
        """Run a function in the controlled thread pool/to_thread does not have the feature to control threadpool"""
        loop = asyncio.get_running_loop()
        func_with_args = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(self.thread_pool, func_with_args)

    async def photometry_routine(self):
        """Runs processing in a thread and writes asynchronously."""
        try:
            async with self.semaphore:  # Use semaphore for control
                outputs = await self._run_in_limited_thread(self.process_image)
                #outputs = await asyncio.to_thread(self.process_image)
                await self.write_results(outputs)
        except Exception as e:
            print(f"Error in photometry routine: {e}")
            raise
    
    @classmethod
    async def process_multiple_images(cls, frames, starlists, config: PhotometryConfig):
        ncores = min(10,os.cpu_count())

        semaphore = asyncio.Semaphore(1000)
        threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=ncores)

        instances = [cls(frame, starlist, config, semaphore, threadpool) for frame, starlist in zip(frames, starlists)]
        tasks = [instance.photometry_routine() for instance in instances]
        await asyncio.gather(*tasks, return_exceptions=True)

    @classmethod
    def process_ccd_wrapper(cls, frames, starlists, config):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                cls.process_multiple_images(frames, starlists, config)
            )
            return result
        finally:
            loop.close()
    '''
    @classmethod
    def process_all_ccds(cls, frames_list, starlists_list, config: PhotometryConfig):
        """Main entry point for multiprocessing."""
        ncores = min(len(frames_list),os.cpu_count())
        #print(f"Starting photometry with {num_ccds} cores...")
        
        #print(f"Starting processing with {len(frames_list)} CCDs")
        #print(f"Each CCD has: {[len(frames) for frames in frames_list]} frames")
        
        process_ccd = partial(cls.process_ccd_wrapper)
        
        with ProcessPoolExecutor(max_workers=ncores) as executor:
            futures = []
            # Add index for tracking
            for i, (frames, starlists) in enumerate(zip(frames_list, starlists_list)):
                #print(f"Submitting CCD {i} with {len(frames)} frames")
                futures.append(
                    executor.submit(process_ccd, frames, starlists, config)
                )
        
            for i, future in enumerate(futures):
                try:
                    future.result()
                    #print(f"CCD {i} completed")
                except Exception as e:
                    print(f"Error in CCD {i}: {e}")
    '''
    @classmethod
    def process_all_ccds(cls, frames_list, starlists_list, config: PhotometryConfig):
        """Main entry point for multiprocessing."""
        nccds = len(frames_list)
        ncores = min(nccds,os.cpu_count())
        
        # Flatten frames_list while tracking original CCD index
        flat_frames_list = []
        flat_starlists_list = []

        for ccd_frames, ccd_starlists in zip(frames_list, starlists_list):
            flat_frames_list.extend(ccd_frames)
            flat_starlists_list.extend(ccd_starlists)  # Preserves frame-starlist pairs

        assert len(flat_frames_list) == len(flat_starlists_list), "Frames and starlists must match."

        frames_per_ccd = [len(frames) for frames in frames_list]

        new_frames_list = [[] for _ in range(nccds)]
        new_starlists_list = [[] for _ in range(nccds)]

        # Create a more balanced distribution plan
        assignment_order = []
        for frame_idx in range(max(frames_per_ccd)): #loop over max number of ccds
            for ccd in range(nccds):
                if frame_idx < frames_per_ccd[ccd]:
                    assignment_order.append((ccd, frame_idx)) #reorders the frames from ccd1,ccd1, .... ccd4,ccd4s -> ccd1,ccd2,.....ccd3,ccd4

        # Distribute according to the plan
        for i, (ccd_idx, frame_idx) in enumerate(assignment_order):
            flat_idx = sum(frames_per_ccd[:ccd_idx]) + frame_idx # the frame index counting from the very first frame (of all ccds)
            target_ccd = i % nccds
            new_frames_list[target_ccd].append(flat_frames_list[flat_idx])
            new_starlists_list[target_ccd].append(flat_starlists_list[flat_idx])

        
        #for i, (frame, starlist) in enumerate(zip(flat_frames_list, flat_starlists_list)):
        #    new_frames_list[i % nccds].append(frame) #round robin redistribition (take the modular and that becomes a circular index with max(index) = nccds-1)
        #    new_starlists_list[i % nccds].append(starlist)
        
        process_ccd = partial(cls.process_ccd_wrapper)
        
        with ProcessPoolExecutor(max_workers=ncores) as executor:
            futures = []
            # Add index for tracking
            for  frames, starlists in zip(new_frames_list, new_starlists_list):
                futures.append(
                    executor.submit(process_ccd, frames, starlists, config)
                )
        
            for i, future in enumerate(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in CCD {i}: {e}")
        