import numpy as np
import DynspecGroup as dg
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import dynspec_tools.dedisperse as dd
import numpy as np
import os
from glob import glob
from astropy.visualization import PercentileInterval
from RMtools_1D.do_RMsynth_1D import run_rmsynth
import bc_corr
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
import pickle as pkl
from astropy.time import Time
# from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib.colors import Normalize
from astropy.constants import c
import astroplan as ap
import warnings
import gc

def Process_GPMJ1839_10(fname, do_stokes):
    t0_pulse_list = [275091226.17263716, 275091441.2599692, 275091526.40357095]
    p_pulse_list = [1318.1957, 1265.2196, 1216.3372]
    t0_orbit = 275095359
    p_orbit = 31482.3
    t0_pulse = t0_pulse_list[1]
    p_pulse = p_pulse_list[1]

    data = dg.DynspecGroup(fname, dm=273.5, mask_time_bins=False, mask_freq_bins=False, transpose=True)
    # if data.time_bc[0] > 1406000000 and data.telescope == 'VLA':
    # if data.time_bc[0] < 1376000000:
    if True:
        data.SplitOnPeriod(t0_pulse, p_pulse)
        data.FitSpec((1.0, -3.17, -0.56))
        print('-------------------- a:', data.vals['a'], data.vals_unc['a'])
        print('-------------------- q:', data.vals['q'], data.vals_unc['q'])
        data.NormaliseTo1GHz(a=-3.17, q=-0.56)
        if do_stokes and 'Q' in data.dynspecs and 'U' in data.dynspecs:
            data.CalcParallacticAngle()
            data.CalcAltAz()
            data.CalcLinear()
            # data.CalcRM(np.nan)
            # print('-------------------- RM:', data.vals['RM'])
            # data.CalcDM(250, 300, 0.1)
            # print('-------------------- DM:', data.best_dm)
            # data.dynspecs['U'] *= -1
            data.Derotate(-532)
            data.CalcLightcurves()
            data.CalcPA()
        else:
            data.CalcLightcurves()
        print('-------------------- max:', np.nanmax(data.curves['I']))
    return data

def Process_J1912_44(fname, do_stokes):
    t0_pulse = 0
    p_pulse = 319.34903
    data = dg.DynspecGroup(fname, dm=0.0, mask_time_bins=False, mask_freq_bins=False, transpose=True, mask_noise=False)
    data.SplitOnPeriod(t0_pulse, p_pulse)
    data.CalcLightcurves()
    return data

def ProcessDynspec(results_fname, process_func, *directories, plot=None, do_stokes=True):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.environ["PLANETARY_EPHEMERIS"] = "/home/septagonic/Documents/round-the-world-exploring/de430.bsp"
    results = []

    count = 0

    for directory in directories:
        os.chdir(directory)
        yaml_fnames = glob('*.yaml')

        fnames = {}
        for fname in yaml_fnames:
            stokes = fname[-6]
            if stokes not in ('I', 'Q', 'U', 'V'):
                stokes = 'I'
                # continue
            if not do_stokes and stokes in ('Q', 'U', 'V'):
                continue
            obsname = os.path.basename(fname)[:-7]
            if obsname not in fnames:
                fnames[obsname] = {}
            fnames[obsname][stokes] = fname

        for obsname in fnames:
                data = process_func(fnames[obsname], do_stokes)
                print(fnames[obsname])

                if type(plot) is str:
                    data.Plot('/home/septagonic/Documents/round-the-world-exploring/' + plot + '/' + obsname)
                elif plot:# and count == 6:
                    data.Plot(perc_int=90)

                count += 1

                data.DropDynspecs()
                results.append(data)

    os.chdir(dir_path)

    with open(results_fname, 'wb')as f:
        pkl.dump(results, f)

if __name__ == '__main__':
    # ProcessDynspec('curves_round.pkl', 'spectra_round.pkl', "/home/septagonic/Documents/round-the-world-exploring/dynspec_iquv")
    folders = ['1376592360', '1378737883', '1378744678', '1694700995', '1376594939', '1378739240', '1378746043', '1706417493']
    fnames = []
    for folder in folders:
        fnames.append("/home/septagonic/Documents/GPMJ1839-10/dynspec/" + folder)

    # ProcessDynspec('pol_results_hist.pkl',
    #     "/home/septagonic/Documents/round-the-world-exploring/dynspec_iquv",
    #     "/home/septagonic/Documents/round-the-world-exploring/ASKAP_EMIL",
    #     # *fnames,
    #     plot=False,
    #     do_stokes=False)
    #     # plot='pol_plots_meerkat_2')
    # ProcessDynspec('pol_results_vla.pkl', "/home/septagonic/Documents/round-the-world-exploring/dynspec_vla_l")

    # ProcessDynspec('pol_results_j1912-44.pkl',
    #     Process_J1912_44,
    #     '/home/septagonic/Documents/j1912-44/yaml_csv',
    #     plot='j1912-44_pol_plots',
    #     do_stokes=True)

    ProcessDynspec('pol_results_vla_l.pkl',
        Process_GPMJ1839_10,
        '/home/septagonic/Documents/VLA_L/csv_yaml',
        plot='pol_plots_vla',
        do_stokes=True)