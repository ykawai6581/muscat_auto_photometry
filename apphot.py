import sys
import os
import numpy as np
from pathlib import Path

# Importing necessary configurations and functions
import conf
import func

def apphot_mapping(reflst, dflst, nstars, rad):
    PI = np.pi
    bin_path = conf.bin()
    rad = round(float(rad), 1)

    # Set directories
    outdir = Path(conf.dir_path("apphot_mapping"))
    outdir.mkdir(parents=True, exist_ok=True)
    outdir = outdir / f"rad{rad}"
    outdir.mkdir(exist_ok=True)

    workdir = Path(conf.dir_path("work"))
    workdir.mkdir(exist_ok=True)
    dfdir = Path(conf.dir_path("df"))

    # Set parameter file
    apparams = conf.file_path("param", "param-apphot.par")
    func.file_check(apparams)

    # Load CCD parameters
    ccdparam = conf.file_path("param", "param-ccd.par")
    ccdparams = func.load_params(ccdparam)
    gain = float(ccdparams["gain"])
    readnoise = float(ccdparams["readnoise"])
    darknoise = float(ccdparams["darknoise"])
    ADUlo = float(ccdparams["ADUlo"])
    ADUhi = float(ccdparams["ADUhi"])

    # Load telescope parameters
    telparam = conf.file_path("param", "param-tel.par")
    telparams = func.load_params(telparam)
    LAT = float(telparams["latitude"])
    LON = float(telparams["longitude"])
    ALT = float(telparams["altitude"])
    diam = float(telparams["diameter"])

    # Load fitsheader parameters
    headparam = conf.file_path("param", "param-fitsheader.par")
    func.file_check(headparam)
    
    fits_keys = {}
    with open(headparam, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) >= 3:
                fits_keys[parts[0]] = (parts[1], parts[2])

    # Load reference list
    refobjs = conf.file_path("reference", "ref-" + func.gethead(reflst) + ".objects")
    func.file_check(refobjs)
    
    x0, y0 = [], []
    with open(refobjs, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            _, x, y, _ = line.split()
            x0.append(float(x))
            y0.append(float(y))
    
    # Process df list
    with open(dflst, "r") as f:
        for dfname in f:
            dfname = dfname.strip()
            if dfname.startswith("#") or not dfname: continue

            headname = func.gethead(dfname)
            df = dfdir / f"{headname}.fits"
            
            # Check file existence
            for ext in [".df.fits", ".fits", ".dfl.fits.gz", ".df.fits.gz", ".fits.gz"]:
                if not df.exists():
                    df = dfdir / f"{headname}{ext}"

            # Exposure time
            exptime = func.get_fits_value(bin_path, df, fits_keys.get("EXPkey"))
            
            # Compute mean Julian Date
            jd1 = func.get_fits_value(bin_path, df, fits_keys.get("JD1key"))
            jd = func.compute_jd(jd1, exptime, fits_keys.get("MJDflag"))

            # Compute airmass
            airmass = func.compute_airmass(bin_path, df, LAT, LON, jd, fits_keys)

            # Load geometric parameters
            geoparam = conf.file_path("geoparam", f"{headname}.geo")
            func.file_check(geoparam)
            dx, dy, a, b, c, d, rms = func.load_geoparams(geoparam)

            # Prepare star list
            tmpstarlst = workdir / f"tmp-{headname}-{rad}.stars"
            with open(tmpstarlst, "w") as f:
                for i in range(nstars):
                    x = dx + a * x0[i] + b * y0[i]
                    y = dy + c * x0[i] + d * y0[i]
                    f.write(f"{x} {y}\n")

            # Set output file
            outfile = outdir / f"{headname}.dat"
            with open(outfile, "w") as f:
                f.write(f"# gjd - 2450000 = {jd:.6f}\n")

            # Run aperture photometry
            apphot_cmd = (
                f"apphot -frame {df} -starlist {tmpstarlst} -file {apparams} "
                f"-gain {gain} -read_out_noise {readnoise} -dark_noise {darknoise} "
                f"-r1 {rad} -r2 {rad} -dr 2 -altitude {ALT} -Diameter {diam} "
                f"-exptime {exptime} -airmass {airmass} -ADU_range {ADUlo} {ADUhi}"
            )
            os.system(f"{apphot_cmd} | tee -a {outfile}")
            os.remove(tmpstarlst)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: apphot_mapping.py [reflist] [dflist] [nstars] [radius[pix]]")
        sys.exit(1)
    
    apphot_mapping(*sys.argv[1:])
