import os
import sys


def check_input(value):
    if not value.isdigit():
        print("Invalid value!")
        sys.exit(1)
    return int(value)


def check_input2(value1, value2):
    if value1 >= value2:
        print("The last frame number must be larger than the first one!")
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print("\nUsage: python config_flat.py [date(yymmdd)] [ccd#(0/1/2/3/all)] ([options])\n")
        print("  ## Options ##")
        print("    -dark [full path to dark for flat] # if you want to use an existing dark\n")
        sys.exit(1)

    date = sys.argv[1]
    ccd = sys.argv[2]
    dark = None
    set_dir_only = False

    for i in range(len(sys.argv)):
        if sys.argv[i] == "-dark" and i + 1 < len(sys.argv):
            dark = sys.argv[i + 1]
        if sys.argv[i] == "-set_dir_only":
            set_dir_only = True

    pwd = os.getcwd()
    if "reduction_afphot/muscat2" in pwd:
        inst = "muscat2"
        datadir = f"/data/MuSCAT2/{date}"
    elif "reduction_afphot/muscat" in pwd:
        inst = "muscat"
        datadir = f"/data/MuSCAT/{date}"
    else:
        print("Please execute in ~/reduction_afphot/muscat or ~/reduction_afphot/muscat2")
        sys.exit(1)

    if ccd == "all" and inst == "muscat":
        ccds = [0, 1, 2]
    elif ccd == "all" and inst == "muscat2":
        ccds = [0, 1, 2, 3]
    else:
        ccds = [int(ccd)]

    instdir = f"/home/muscat/reduction_afphot/{inst}"
    datedir = f"{instdir}/{date}"
    flatdir = f"{datedir}/FLAT"
    listdir = f"{flatdir}/list"

    os.makedirs(datedir, exist_ok=True)
    os.makedirs(flatdir, exist_ok=True)
    os.makedirs(listdir, exist_ok=True)

    rawdata_link = f"{flatdir}/rawdata"
    if not os.path.islink(rawdata_link):
        print(f"ln -s {datadir} {rawdata_link}")
        os.symlink(datadir, rawdata_link)

    param_dir = f"{flatdir}/param"
    if not os.path.exists(param_dir):
        print(f"cp -r {instdir}/params/param_{inst} {param_dir}")
        os.system(f"cp -r {instdir}/params/param_{inst} {param_dir}")

    if set_dir_only:
        sys.exit(0)

    for ccd in ccds:
        flat1 = check_input(input(f"FIRST frame number of FLAT (CCD{ccd})? "))
        flat2 = check_input(input(f"LAST frame number of FLAT (CCD{ccd})? "))
        check_input2(flat1, flat2)

        conf_file = f"{listdir}/flat_ccd{ccd}.conf"
        with open(conf_file, "w") as conf:
            conf.write(f"flat {flat1} {flat2}\n")

            if dark:
                conf.write(f"flat_dark {dark}\n")
            else:
                fdark1 = check_input(input(f"FIRST frame number of DARK for FLAT (CCD{ccd})? "))
                fdark2 = check_input(input(f"LAST frame number of DARK for FLAT (CCD{ccd})? "))
                check_input2(fdark1, fdark2)
                conf.write(f"flat_dark {fdark1} {fdark2}\n")

        print()
        os.system(f"cat {conf_file}")

    print()


if __name__ == "__main__":
    main()
