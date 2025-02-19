import numpy as np
from astropy.io import fits
import argparse
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import os

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

    def process_image(self, ap_r, outfile) -> None:
        """Main processing function for aperture photometry."""
        #print(f"## apphot version {self.version} ##")
        
        # Read FITS image
        with fits.open(self.frame) as hdul:
            image_data = hdul[0].data
            image_header = hdul[0].header

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
            if starid == self.tid - 1:
                continue
                
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

        with open(outfile, "w") as f:
            f.write(f"# gjd - 2450000 = {jd_2450000_mid}\n\n")
            f.write(f"## apphot version {self.version}##\n\n")
            f.write(f"# nstars = {nstars}\n")
            f.write(f"# filename = {outfile[:4].split('/')[-1]}.df.fits\n")
            f.write(f"# gain = {self.gain}\n")
            f.write(f"# readout_noise = {self.read_noise}\n")
            f.write(f"# dark_noise = {self.dark_noise}\n")
            f.write(f"# ADU_range = {self.adu_lo} - {self.adu_hi}\n")
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
        



        print("# ID xcen ycen nflux flux err sky sky_sdev SNR nbadpix fwhm peak")
        for result in results:
            print(f"{result['id']:.0f} {result['xcen']:.3f} {result['ycen']:.3f} "
                  f"{result['flux']:.2f} {result['flux']:.2f} {result['noise']:.2f} "
                  f"{result['sky']:.2f} {result['sky_std']:.2f} {result['snr']:.2f} "
                  f"0 {result['fwhm']:.2f} {result['peak']:.1f}")
        #this prints the parameters for all stars
        '''

    def process_image_over_rads(self):
        for rad in self.rads:
            dirs = self.frame.split("/") #-> obsdate/target_ccd/df/frame_df.fits
            
            outpath = f"{dirs[0]}/{dirs[1]}/apphot_{self.method}_test/rad{rad}/"
            os.makedirs(outpath, exist_ok=True)
            filename =f"{dirs[-1][:-8]}.dat"  #-> target_ccd/apphot_method/rad/frame.dat
            self.process_image(ap_r=rad,outfile=f"{outpath}/{filename}")
        #print(f"Done with {self.rads}")
