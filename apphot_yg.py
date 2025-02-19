import numpy as np
from astropy.io import fits
import argparse
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import os
import asyncio
from types import SimpleNamespace
import re
import pandas as pd
from pathlib import Path

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

def load_par_file(filename):
    """Load a .par file into a dictionary."""
    params = {}
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):  # Ignore empty lines and comments
                key, value = line.split(None, 1)  # Split on first whitespace
                try:
                    params[key] = float(value)  # Convert numerical values to float
                except ValueError:
                    params[key] = value  # Keep as string if conversion fails
    return params

def load_geo_file(filename):
    """Returns the relevant coefficients needed for transformation of pixels in ref file to object file"""
    with open(filename, "r") as file:
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

class PhotometryProcessor:
    def __init__(self, target_dir, tid, rad_to_use, nstars):
        self.target_dir = target_dir
        self.tid = tid
        self.rad_to_use = rad_to_use
        self.nstars = nstars
        # Load parameters once during initialization
        self.tel = self._load_telescope_params()
        self.app = self._load_apphot_params()
        self.ccd = self._load_ccd_params()
        self.x0, self.y0 = self._load_reference_coordinates()
        
    def _load_telescope_params(self):
        telescope_param = load_par_file(f"{self.target_dir}/param/param-tel.par")
        return SimpleNamespace(**telescope_param)
        
    def _load_apphot_params(self):
        apphot_param = load_par_file(f"{self.target_dir}/param/param-apphot.par")
        return SimpleNamespace(**apphot_param)
        
    def _load_ccd_params(self):
        ccd_param = load_par_file(f"{self.target_dir}/param/param-ccd.par")
        return SimpleNamespace(**ccd_param)
        
    def _load_reference_coordinates(self):
        with open(Path(f"{self.target_dir}/list/ref.lst"), 'r') as f:
            ref_file = f.read()
        ref_frame = ref_file.replace('\n','')
        metadata, data = parse_obj_file(f"{self.target_dir}/reference/ref-{ref_frame}.objects")
        return (np.array(data["x"][:self.nstars]), 
                np.array(data["y"][:self.nstars]))

    def _create_apphot_instance(self, sky_calc_mode, const_sky_flag, const_sky_flux, const_sky_sdev):
        return ApPhotometry(
            tid=self.tid,
            rads=self.rad_to_use,
            gain=self.ccd.gain,
            read_noise=self.ccd.readnoise,
            dark_noise=self.ccd.darknoise,
            sky_sep=self.app.sky_sep,
            sky_wid=self.app.sky_wid,
            hbox=self.app.hbox,
            dcen=self.app.dcen,
            sigma_cut=self.app.sigma_cut,
            adu_lo=self.ccd.ADUlo,
            adu_hi=self.ccd.ADUhi,
            sigma_0=self.app.sigma_0,
            altitude=self.tel.altitude,
            diameter=self.tel.diameter,
            global_sky_flag=self.app.global_sky_flag,
            sky_calc_mode=sky_calc_mode,
            const_sky_flag=const_sky_flag,
            const_sky_flux=const_sky_flux,
            const_sky_sdev=const_sky_sdev,
        )

    async def _process_single_file(self, ccd_id, file, apphot):
        """Process a single file's photometry."""
        geoparam_file_path = f"{self.target_dir}_{ccd_id}/geoparam/{file[:-4].split('/')[-1]}.geo"
        geoparams = await asyncio.to_thread(load_geo_file, geoparam_file_path)
        geo = SimpleNamespace(**geoparams)

        x = geo.dx + geo.a * self.x0 + geo.b * self.y0
        y = geo.dy + geo.c * self.x0 + geo.d * self.y0
        starlist = [x, y]
        
        dffits_file_path = f"{self.target_dir}_{ccd_id}/df/{file[:-4].split('/')[-1]}.df.fits"
        await asyncio.to_thread(apphot.add_frame, dffits_file_path, starlist)
        await asyncio.to_thread(apphot.process_image_over_rads)

    async def process_ccd(self, ccd_id, missing_files, sky_params):
        """Process all files for a single CCD."""
        print(f"## >> CCD={ccd_id} | Begin aperture photometry")
        apphot = self._create_apphot_instance(**sky_params)
        
        tasks = [self._process_single_file(ccd_id, file, apphot) 
                for file in missing_files]
        await asyncio.gather(*tasks)
        print(f"## >> CCD={ccd_id} | Completed aperture photometry")

    async def run_photometry(self, missing_files_per_ccd, sky_calc_mode, 
                           const_sky_flag, const_sky_flux, const_sky_sdev):
        """Main entry point for running photometry on all CCDs."""
        sky_params = {
            'sky_calc_mode': sky_calc_mode,
            'const_sky_flag': const_sky_flag,
            'const_sky_flux': const_sky_flux,
            'const_sky_sdev': const_sky_sdev
        }
        
        tasks = [self.process_ccd(ccd_id, files, sky_params) 
                for ccd_id, files in missing_files_per_ccd.items()]
        await asyncio.gather(*tasks)

class ApPhotometry:
    def __init__(self, 
                 tid:            int, #target star id
                 rads,                 #list of aperture radius
                 gain:           float, #ccd gain (e/ADU) 
                 read_noise:     float, 
                 dark_noise:     float, 
                 sky_sep:        int  =50,
                 sky_wid:        int  =40,
                 hbox:           int  =0, #number of pixels around a given point to search for max flux aperture
                 dcen:           float=0.1, #step in pixels to move within hbox
                 sigma_cut:      int  =3,#sigma clipping used for sky calculation
                 adu_lo:         int  =1000,
                 adu_hi:         int  =65000,#16bit adu limit 2^16-1=65,535,
                 sigma_0:        float=0.064,#scintillation coefficient
                 altitude:       int  =1013,#observatory altitude in meters (used for scintillation noise calculation)
                 diameter:       int  =61,#telescope diameter in cm (also used for scintillation noise calculation)
                 sky_calc_mode:  int  =0, #Sky calculation mode (0=mean, 1=median, 2=mode)
                 global_sky_flag:int  =0, #Use global sky calculation meaning calculate sky dont assume as constant
                 const_sky_flag: int  =0, #Use constant sky value
                 const_sky_flux: float=0.0,#Constant sky flux value
                 const_sky_sdev: float=0.0,#Constant sky standard deviation
                 method:         str="mapping"#apphot method either mapping or centroid

                 ):
        
        self.tid = tid
        self.rads = rads
        self.gain = gain
        self.read_noise = read_noise
        self.dark_noise = dark_noise
        self.sky_sep = sky_sep
        self.sky_wid = sky_wid
        self.hbox = hbox
        self.dcen = dcen
        self.sigma_cut = sigma_cut
        self.adu_lo = adu_lo
        self.adu_hi = adu_hi
        self.sigma_0 = sigma_0
        self.altitude = altitude
        self.diameter = diameter
        self.sky_calc_mode = sky_calc_mode
        self.global_sky_flag = global_sky_flag
        self.const_sky_flag = const_sky_flag
        self.const_sky_flux = const_sky_flux
        self.const_sky_sdev = const_sky_sdev
        self.method = method

        self.version = "3.0.0"

    def add_frame(self, frame, starlist):
        self.frame = frame
        self.x, self.y = starlist
        
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

    async def process_image(self, ap_r, infile, outfile, outpath) -> None:
        """Main processing function for aperture photometry."""
        #print(f"## apphot version {self.version} ##")
        os.makedirs(outpath, exist_ok=True)
        image_header, image_data = infile

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
        
        results = []
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
            max_flux = 0
            max_sky = 0
            max_sky_std = 0
            max_pos = (0, 0)

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
                    
                    # Define aperture and sky annulus
                    aper_mask = r <= ap_r #pixels within the aperture (it means that up until this point, the pixels are still in the aperture)
                    #print(ap_r)
                    #print(np.unique(aper_mask))
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
                    
                    # Calculate flux
                    flux = np.sum(cameo[aper_mask]) - sky * np.sum(aper_mask)

                    if flux > max_flux:
                        max_flux = flux
                        max_sky = sky
                        max_sky_std = sky_std
                        max_pos = (xcen, ycen)

            # Calculate noise and SNR
            '''
            Note on the uncertainty in mean sky level
            ##the uncertainty affects all pixels in the aperture) (npix)
            ##the uncertainty in subtracted sky level estimated from pixels in sky annulus (npix * max_sky_std**2)/n_sky_pixels
            ####affects all pixels in the aperture hence npix but divided by n_sky_pixels because more sky pixels reduce uncertainty
            '''
            n_pix = np.sum(aper_mask)
            noise = np.sqrt(max_flux * self.gain + #flux*gain -> shot noise in electrons
                          n_pix * max_sky_std**2 * self.gain**2 + #uncertainty in pixel-to-pixel variation in the sky background in electrons (inherent)
                          n_pix**2 / len(sky_pixels) * max_sky_std**2 * self.gain**2 + #uncertainty in mean sky level in electrons 
                          n_pix * self.read_noise**2 + #read noise in electrons
                          n_pix * self.dark_noise**2 * exptime**2 + #dark current noise in electrons
                          (sigma_scin * max_flux * self.gain)**2) / self.gain #scintillation noise in electrons and division by gain to convert to ADU
            
            snr = max_flux / noise
            
            # Calculate FWHM

            peak_flux = np.max(cameo[aper_mask] - max_sky)
            hm = peak_flux / 2.0
            fwhm_mask = (cameo[aper_mask] - max_sky > hm)
            if np.any(fwhm_mask):
                fwhm = 2 * np.mean(r[aper_mask][fwhm_mask])
            else:
                fwhm = 0
                
            results.append({
                'id': starid+1,
                'xcen': max_pos[0],
                'ycen': max_pos[1],
                'flux': max_flux,
                'sky': max_sky,
                'sky_std': max_sky_std,
                'noise': noise,
                'snr': snr,
                'fwhm': fwhm,
                'peak': peak_flux + max_sky
            })
            
        # save results
        
        with open(f"{outpath}/{outfile}", "w") as f:
            f.write(f"# gjd - 2450000 = {jd_2450000_mid}\n\n")
            f.write(f"## apphot version {self.version}##\n\n")
            f.write(f"# nstars = {nstars}\n")
            f.write(f"# filename = {outfile[:4].split('/')[-1]}.df.fits\n")
            f.write(f"# gain = {self.gain}\n")
            f.write(f"# readout_noise = {self.read_noise}\n")
            f.write(f"# dark_noise = {self.dark_noise}\n")
            f.write(f"# ADU_range = {self.adu_lo} {self.adu_hi}\n")
            f.write(f"# r = {ap_r}\n")
            f.write(f"# hbox = {self.hbox}\n")
            f.write(f"# dcen = {self.dcen}\n")
            f.write(f"# sigma_cut = {self.sigma_cut}\n")
            f.write(f"# altitude = {self.altitude}\n")
            f.write(f"# Diameter = {self.diameter}\n")
            f.write(f"# exptime = {exptime}\n")
            f.write(f"# sigma_0 = {self.sigma_0}\n")
            f.write(f"# airmass = {airmass}\n\n")

            f.write(f"# global_sky_flag = {self.global_sky_flag}\n")
            f.write(f"# const_sky_flag = {self.const_sky_flag}\n")
            f.write(f"# sky_calc_mode = {self.sky_calc_mode}\n")
            f.write(f"# sky_sep = {self.sky_wid}\n")
            f.write(f"# sky_wid = {self.sky_wid}\n")

            f.write("# ID xcen ycen nflux flux err sky sky_sdev SNR nbadpix fwhm peak\n")
            for result in results:
                f.write(f"{result['id']:.0f} {result['xcen']:.3f} {result['ycen']:.3f} "
                        f"{result['flux']:.2f} {result['flux']:.2f} {result['noise']:.2f} "
                        f"{result['sky']:.2f} {result['sky_std']:.2f} {result['snr']:.2f} "
                        f"0 {result['fwhm']:.2f} {result['peak']:.1f}\n")

        '''
        output = []
        
        output.append(f"# gjd - 2450000 = {jd_2450000_mid}\n")
        output.append(f"## apphot version {self.version}##\n")
        output.append(f"# nstars = {nstars}\n")
        output.append(f"# filename = {outfile[:4].split('/')[-1]}.df.fits\n")
        output.append(f"# gain = {self.gain}\n")
        output.append(f"# readout_noise = {self.read_noise}\n")
        output.append(f"# dark_noise = {self.dark_noise}\n")
        output.append(f"# ADU_range = {self.adu_lo} - {self.adu_hi}\n")
        output.append(f"# r = {ap_r}\n")
        output.append(f"# hbox = {self.hbox}\n")
        output.append(f"# dcen = {self.dcen}\n")
        output.append(f"# sigma_cut = {self.sigma_cut}\n")
        output.append(f"# altitude = {self.altitude}\n")
        output.append(f"# Diameter = {self.diameter}\n")
        output.append(f"# exptime = {exptime}\n")
        output.append(f"# sigma_0 = {self.sigma_0}\n")
        output.append(f"# airmass = {airmass}\n")

        output.append(f"# global_sky_flag = {self.global_sky_flag}\n")
        output.append(f"# const_sky_flag = {self.const_sky_flag}\n")
        output.append(f"# sky_calc_mode = {self.sky_calc_mode}\n")
        output.append(f"# sky_sep = {self.sky_wid}\n")
        output.append(f"# sky_wid = {self.sky_wid}\n")

        output.append("# ID xcen ycen nflux flux err sky sky_sdev SNR nbadpix fwhm peak\n")
        
        for result in results:
            output.append(
                f"{result['id']:.0f} {result['xcen']:.3f} {result['ycen']:.3f} "
                f"{result['flux']:.2f} {result['flux']:.2f} {result['noise']:.2f} "
                f"{result['sky']:.2f} {result['sky_std']:.2f} {result['snr']:.2f} "
                f"0 {result['fwhm']:.2f} {result['peak']:.1f}"
            )

        return "\n".join(output)
    
    def process_image_over_rads(self):
        dirs = self.frame.split("/") #-> obsdate/target_ccd/df/frame_df.fits

        with fits.open(self.frame) as hdul:
            image_data = hdul[0].data
            image_header = hdul[0].header
        
        for rad in self.rads:
            outpath = f"{dirs[0]}/{dirs[1]}/apphot_{self.method}_test/rad{rad}/"
            os.makedirs(outpath, exist_ok=True)
            filename =f"{dirs[-1][:-8]}.dat"  #-> target_ccd/apphot_method/rad/frame.dat
            data = self.process_image(ap_r=rad, infile=[image_header,image_data], outfile=f"{outpath}/{filename}")
            #consider asynchrounously writing
    '''
    async def process_image_over_rads(self):
        dirs = self.frame.split("/") #-> obsdate/target_ccd/df/frame_df.fits
        filename =f"{dirs[-1][:-8]}.dat"  #-> target_ccd/apphot_method/rad/frame.dat
        outpath=f"{dirs[0]}/{dirs[1]}/apphot_{self.method}_test"

        with fits.open(self.frame) as hdul:
            image_data = hdul[0].data
            image_header = hdul[0].header

        tasks = [self.process_image(ap_r=rad, infile=[image_header, image_data], outfile=filename, outpath=outpath) for rad in self.rads]
        await asyncio.gather(*tasks)