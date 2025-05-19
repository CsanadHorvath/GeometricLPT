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

def Fold(t, t0, period):
    p_num = np.round((t - t0) / period) # Period number of each time
    residual = t - t0 - p_num * period
    return p_num, residual

def CurveLaw(v, S1GHz, a, q): # frequency must be in GHz
    # S_v = S_1GHz * (v / 1GHz)^a * exp(q * (log(v / 1GHz))^2)
    return S1GHz * (v**a) * np.exp(q * ((np.log(v))**2))

def SigmaClipRMS(x, sigma, axis=None, keepdims=False):
    rms = nanstd(x, axis=axis, keepdims=True)
    x_copy = x.copy()
    x_copy[x > sigma * rms] = np.nan
    return nanstd(x_copy, axis=axis, keepdims=keepdims)

def SubtractMeanSpec(dynspec):
    return dynspec - nanmean(dynspec, axis=1, keepdims=True)

def nanmean(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        return np.nanmean(*args, **kwargs)
    
def nansum(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        return np.nansum(*args, **kwargs)
    
def nanstd(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        return np.nanstd(*args, **kwargs)

class DynspecGroup():
    # Read the dynamic spectrum and dedisperse it
    def __init__(self, fnames, ra=None, dec=None, dm=0.0, mask_time_bins=True, mask_freq_bins=True, transpose=None, mask_noise=True):
        self.fnames = fnames
        self.dynspecs = {}
        self.dynspec_objs = {}
        self.stokes_params = []
        self.curves = {}
        self.spectra = {}
        self.unc_f = {}
        self.unc_t = {}
        self.unc = {}
        self.args = {}
        self.vals = {}
        self.vals_unc = {}
        self.ra = ra
        self.dec = dec

        if 'I' not in fnames:
            raise Exception('Stokes I dynamic spectrum not found!')

        for stokes_param in fnames:
            if stokes_param not in ['I', 'Q', 'U', 'V']:
                raise Exception(f'Unknown stokes parameter "{stokes_param}"!')


            with open(fnames[stokes_param]) as f:
                args = dd.parse_yaml(f)

            if not mask_time_bins and 'mask_time_bins' in args:
                del args['mask_time_bins']
            if not mask_freq_bins and 'mask_freq_bins' in args:
                del args['mask_freq_bins']

            if transpose is not None:
                args['transpose'] = transpose

            dynspec = dd.Dynspec(**{k:args[k] for k in args})
            dynspec.set_freq_ref('centre')
            dynspec.dynspec[np.isnan(dynspec.dynspec)] = 0
            dynspec.dedisperse(dm)
            dynspec.dynspec[dynspec.dynspec == 0.0] = np.nan

            if np.all((dynspec.dynspec == 0.0) | (~np.isfinite(dynspec.dynspec))):
                continue

            if stokes_param == 'U' and args['telescope'] == 'ASKAP':
                dynspec.dynspec *= -1

            if stokes_param == 'I':
                self.time = dynspec.get_time_at_infinite_frequency().copy()

            if ra is not None:
                dynspec.RA = ra

            if dec is not None:
                dynspec.Dec = dec

            # dynspec.bc_correct()
            # if not dynspec.bc_corrected:
            #     raise Exception("Barycentric correction failed!")
            self.dynspecs[stokes_param] = dynspec.dynspec
            self.dynspec_objs[stokes_param] = dynspec
            self.args[stokes_param] = args
            self.stokes_params.append(stokes_param)

            if stokes_param == 'I':
                self.telescope = args['telescope']
                self.freq = dynspec.f.copy()
                # self.time_bc = dynspec.get_time_at_infinite_frequency().copy()

        self.subsets = [np.ones(self.time.size, dtype=bool)]
        self.is_pulse = np.ones(self.time.size, dtype=bool)
        self.dm_curve = None
        self.best_dm = None

        if self.telescope == 'VLBA':
            location = EarthLocation(-107.61831, 34.07883)
        elif self.telescope == 'VLITE':
            location = EarthLocation.of_site('VLA')
        else:
            location = EarthLocation.of_site(self.telescope)
        coords_gpm = SkyCoord("18:39:02", "-10:31:49.5", frame="fk5", unit=(u.hour, u.deg))
        self.target = ap.FixedTarget(coord=coords_gpm)
        self.observer = ap.Observer(location=location)

        gps_times = Time(self.time, format='gps', scale='utc')
        coords_gpm = SkyCoord("18:39:02", "-10:31:49.5", frame="fk5", unit=(u.hour, u.deg))
        correction = np.array([bc_corr.bc_corr(coords_gpm, t, ephemeris_file='/home/septagonic/Documents/round-the-world-exploring/de430.bsp') for t in gps_times])
        self.time_bc = self.time + correction

        self.DetectPulses()
        self.CalcRMS()
        if mask_noise:
            self.MaskNoisyChannels()
        self.DetectPulses()
        self.CalcRMS()

    # Calculate the RMS through frequency and time, ignoring real signals
    def CalcRMS(self, *stokes_params):
        if not stokes_params:
            stokes_params = self.stokes_params

        for stokes_param in stokes_params:
            noise = self.dynspecs[stokes_param].copy()
            noise[:,self.is_pulse] = np.nan
            self.unc[stokes_param] = nanstd(noise)
            self.unc_f[stokes_param] = nanstd(noise, axis=1)
            # plt.plot(self.freq, self.unc_f[stokes_param]*1e3)
            # plt.xlabel('Frequency (MHz)')
            # plt.ylabel('Range (mJy)')
            # plt.show()
            
#    Calculate the lightcurves, which are the means through frequency
    def CalcLightcurves(self):
        for stokes_param in self.dynspecs:
            self.curves[stokes_param] = nanmean(self.dynspecs[stokes_param], axis=0)
            self.unc_t[stokes_param] = self.unc[stokes_param] / np.sqrt(np.count_nonzero(np.isfinite(self.dynspecs[stokes_param]), axis=0))# / 6

        if 'U' in self.curves and 'Q' in self.curves:
            L = np.sqrt(self.curves['Q']**2 + self.curves['U']**2)
            # sigmaI = self.unc_t['I']
            # not_zero = (L / sigmaI) >= -1.57
            # self.curves['L'][not_zero] = sigmaI[not_zero] * np.sqrt((L / sigmaI)[not_zero]**2 - 1)
            self.curves['L'] = L
            # DLdQ = Q / L
            # DL = Q / L * dQ
            # DLdU = U / L
            # DL = U / L * dU
            # dL = sqrt((Q * dQ)^2 + (U * dU)^2) / L
            self.unc_t['L'] = np.sqrt((self.curves['Q'] * self.unc_t['Q'])**2 + (self.curves['U'] * self.unc_t['U'])**2) / self.curves['L']


    # Detect the time bins where real signals are
    def DetectPulses(self):
        f = nanmean(self.dynspecs['I'], axis=0)
        t = self.time_bc
        sel = f < 0.01
        sel_mean = nanmean(f[sel])
        sel_std = nanstd(f[sel])
        # self.is_pulse = (f > 0.01) & ((f > nanstd(f[f < 0.01])*6) | (t[-1] - t[0] < 120))
        self.is_pulse = ((f - sel_mean) > 0.001) & (((f - sel_mean) > (sel_std * 5)) | (t[-1] - t[0] < 120))

    # Normalise the dynspec to 1 GHz
    def NormaliseTo1GHz(self, a=-3.17, q=-0.56):
        norm = 1.0 / CurveLaw(self.freq * 1e-3, 1.0, a, q).reshape(-1, 1)
        for stokes_param in self.stokes_params:
            self.dynspecs[stokes_param] *= norm
        self.CalcRMS()

    # Weight down high RMS frequency chanels, or mask them by setting to NaN
    def MaskNoisyChannels(self):
        rms = self.unc_f['I']
        weight = np.ones_like(rms)
        if self.telescope == 'MeerKAT':
            invalid = ~(rms < nanmean(rms))
            invalid |= self.freq > 1030
            invalid |= self.freq < 580
            weight[invalid] = np.nan
        elif self.telescope == 'VLA':
            invalid = ~(rms < 0.8*nanmean(rms))
            weight[invalid] = np.nan
        if not np.all(np.isnan(weight)):
            for stokes_param in self.stokes_params:
                self.dynspecs[stokes_param] *= weight.reshape(-1, 1)
        self.CalcRMS()

    # Make each pulse a different subset
    def SplitOnPeriod(self, t0, period):
        num, _ = Fold(self.time_bc, t0, period)
        num_unq = np.unique(num)
        self.subsets = []
        self.vals['t'] = []
        self.vals_unc['t'] = []
        for n in num_unq:
            self.subsets.append(num == n)
            self.vals['t'].append(t0 + n * period)
            self.vals_unc['t'].append(period / 2)

    # Calculate RM for each subset. RMs are stored in self.vals['RM'] which is a list of length len(self.subsets)
    def CalcRM(self, default_rm=np.nan):
        if 'U' not in self.stokes_params or 'Q' not in self.stokes_params:
            raise Exception('U and Q are needed to calculate RM!')
        
        self.vals['RM'] = []
        self.vals_unc['RM'] = []
        self.curves['RM'] = np.full(self.time_bc.size, np.nan)
        self.unc_t['RM'] = np.full(self.time_bc.size, np.nan)
        
        for subset in self.subsets:
            pulse = subset & self.is_pulse
            if np.any(pulse):
                # Mean dynspecs through time
                meanI = nanmean(self.dynspecs['I'][:,pulse], axis=1)
                meanQ = nanmean(self.dynspecs['Q'][:,pulse], axis=1)
                meanU = nanmean(self.dynspecs['U'][:,pulse], axis=1)
                # Uncertainties of means through time
                n = np.count_nonzero(pulse)
                sigmaI = np.sqrt(self.unc_f['I']**2 / n)
                sigmaQ = np.sqrt(self.unc_f['Q']**2 / n)
                sigmaU = np.sqrt(self.unc_f['U']**2 / n)
                # Calculating the RM
                try:
                    rmdict, _ = run_rmsynth(data=[self.freq*1e6, meanI, meanQ, meanU, sigmaI, sigmaQ, sigmaU])
                    RM = rmdict['phiPeakPIfit_rm2']
                    uRM = rmdict['dPhiPeakPIfit_rm2']
                except:
                    RM = default_rm
                    uRM = 0.0
            else:
                RM = default_rm
                uRM = 0.0

            self.vals['RM'].append(RM)
            self.vals_unc['RM'].append(uRM)
            
            self.curves['RM'][subset] = self.vals['RM'][-1]
            self.unc_t['RM'][subset] = self.vals_unc['RM'][-1]

    def CalcDM(self, start, stop, step):
        self.dm_curve = []
        self.best_dm = []
        for subset in self.subsets:
            args = self.args['I']
            dynspec = dd.Dynspec(**{k:args[k] for k in args})
            dynspec.set_freq_ref('centre')
            dynspec.bc_correct()
            dynspec.dynspec[np.isnan(dynspec.dynspec)] = 0.0
            # mask0 = np.all(np.isnan(self.dynspecs['I']),axis=1)
            # mask1 = np.all(np.isnan(self.dynspecs['I']),axis=0)
            # dynspec.dynspec[mask0, :] = 0.0
            # dynspec.dynspec[:, mask1] = 0.0
            dynspec.dynspec[~np.isfinite(self.dynspecs['I'])] = 0.0
            dynspec.dynspec[:,~subset] = 0.0
            dm_curve = dd.DMCurve(dynspec)
            dm_curve.run_dmtrials((start, stop, step))
            dm_curve.calc_best_dm()
            self.dm_curve.append(dm_curve)
            self.best_dm.append(dm_curve.best_dm)

    # Apply RM
    def Derotate(self, rm_in=None):
        for i in range(len(self.subsets)):
            if rm_in is None:
                rm = self.vals['RM'][i]
            else:
                rm = rm_in
            if type(rm) is not u.Quantity:
                rm = rm * u.m**2
            if not np.isnan(rm):
                s = self.subsets[i]
                L = self.dynspecs['Q'][:,s] + self.dynspecs['U'][:,s]*1j
                f = self.freq.reshape(-1,1) * u.MHz
                exp = np.exp(-2j*(rm*(c/f)**2).decompose().value)
                Lfr = L * exp
                self.dynspecs['Q'][:,s] = np.real(Lfr)
                self.dynspecs['U'][:,s] = np.imag(Lfr)

    # Calculate the linear polarisation component
    def CalcLinear(self):
        L = np.sqrt(self.dynspecs['Q']**2 + self.dynspecs['U']**2)
        # Getting rid of L offset - Lorimer handbook 7.4.3.1
        # sigmaI = self.unc_f['I'].reshape(-1, 1) * np.ones_like(self.dynspecs['I'])
        # self.dynspecs['L'] = np.zeros_like(self.dynspecs['I'])
        # not_zero = (L / sigmaI) >= 1.57
        # self.dynspecs['L'][not_zero] = sigmaI[not_zero] * np.sqrt((L / sigmaI)[not_zero]**2 - 1)
        # self.dynspecs['L'][np.isnan(self.dynspecs['Q']) | np.isnan(self.dynspecs['U'])] = np.nan
        self.dynspecs['L'] = L
        self.CalcRMS('L')
    
    # Calculate the position angle
    def CalcPA(self):
        self.curves['PA'] = np.angle(self.curves['Q'] + self.curves['U']*1j) / 2
        self.unc_t['PA'] = 28.65 * np.pi/180 * self.unc_t['I'] / self.curves['L'] # From Lorimer handbook 7.4.3.1
        # invalid = self.curves['L'] / self.unc_t['I'] < 3
        # self.curves['PA'][invalid] = np.nan
        # self.unc_t['PA'][invalid] = np.nan

    def FitSpec(self, p0):
        param_names = ['S1GHz', 'a', 'q']
        for name in param_names:
            self.vals[name] = []
            self.vals_unc[name] = []
        valid = np.all(np.isfinite(self.dynspecs['I']), axis=1)
        for subset in self.subsets:
            dynspec = self.dynspecs['I'][:,subset]
            dynspec = dynspec[valid,:]
            weight = nanmean(dynspec, axis=0, keepdims=True) ** 2
            weight /= nansum(weight)
            spec = nansum(dynspec * weight, axis=1)
            weight_masked = np.ones_like(dynspec)
            weight_masked[np.isnan(dynspec)] = np.nan
            weight_masked *= weight
            weight_masked *= np.count_nonzero(np.isfinite(weight_masked), axis=1, keepdims=True)
            sigma = self.unc_f['I'][valid] / np.sqrt(np.sum(weight_masked, axis=1))
            freqs = self.freq[valid] / 1e3
            try:
                popt, pcov = curve_fit(CurveLaw, freqs, spec, p0, sigma)
                perr = np.sqrt(np.diag(pcov))
                # spec_mod = CurveLaw(freqs, *popt)
                # plt.errorbar(freqs, spec, sigma, fmt='.', label='Data')
                # plt.plot(freqs, spec_mod, '.', label='Fit')
                # plt.legend()
                # plt.title(f'a = {popt[1]}, q = {popt[2]}')
                # plt.show()
            except:
                popt = (np.nan, np.nan, np.nan)
                perr = (np.nan, np.nan, np.nan)
            
            self.vals['S1GHz'].append(popt[0])
            self.vals['a'].append(popt[1])
            self.vals['q'].append(popt[2])
            self.vals_unc['S1GHz'].append(perr[0])
            self.vals_unc['a'].append(perr[1])
            self.vals_unc['q'].append(perr[2])
            
    def CalcParallacticAngle(self):
        arr_times = Time(self.time, format='gps')
        self.curves['q'] = self.observer.parallactic_angle(arr_times, self.target).radian

    def CalcAltAz(self):
        arr_times = Time(self.time, format='gps')
        toa_altaz = self.target.coord.transform_to(AltAz(obstime=arr_times, location=self.observer.location))
        self.altitude = toa_altaz.alt.radian
        self.azimuth = toa_altaz.az.radian

    def DropDynspecs(self):
        del self.dynspecs
        del self.dynspec_objs
        gc.collect()

    # Plot all of the information in this object
    def Plot(self, fname=None, perc_int=100):
        fig1 = plt.figure(figsize=(18, 8))
        ds_names = ['I', 'Q', 'U', 'V', 'L']

        interval = PercentileInterval(perc_int)
        vmin, vmax = interval.get_limits(self.dynspecs['I'])

        gs = grid.GridSpec(2, 6, width_ratios=[1,1,1,1,1,0.075])
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.05, hspace=0.025)
        ax_ds = {'I':fig1.add_subplot(gs[1,0])}
        fig1.text(0.5, 0.04, 'Time (s)', ha='center')
        ax_ds['I'].set_ylabel('Frequency (MHz)')
        pcm_ds = {}

        for i in range(1, len(ds_names)):
            name = ds_names[i]
            if name in self.dynspecs:
                ax_ds[name] = fig1.add_subplot(gs[1,i], sharex=ax_ds['I'], sharey=ax_ds['I'])
                ax_ds[name].get_yaxis().set_visible(False)
        for name in ds_names:
            if name in self.dynspecs:
                pcm_ds[name] = ax_ds[name].pcolormesh(self.time_bc, self.freq, self.dynspecs[name][:-1,:-1]*1e3, vmin=vmin*1e3, vmax=vmax*1e3)

        ax_col = fig1.add_subplot(gs[1,5])
        fig1.colorbar(pcm_ds['I'], cax=ax_col, label='Flux (mJy)')

        ax_lc = {'I':fig1.add_subplot(gs[0,0], sharex=ax_ds['I'])}
        for i in range(1, len(ds_names)):
            name = ds_names[i]
            if name in self.dynspecs:
                ax_lc[name] = fig1.add_subplot(gs[0,i], sharex=ax_ds['I'], sharey=ax_lc['I'])
                ax_lc[name].get_yaxis().set_visible(False)
        for name in ds_names:
            if name in self.dynspecs:
                ax_lc[name].get_xaxis().set_visible(False)
                ax_lc[name].set_title(name)
                ax_lc[name].errorbar(self.time_bc, self.curves[name]*1e3, self.unc_t[name]*1e3, fmt='-')
                # ax_lc[name].errorbar(self.time[self.is_pulse], self.curves[name][self.is_pulse]*1e3, self.unc_t[name][self.is_pulse]*1e3, fmt='r.')

        ax_lc['I'].set_ylabel('Flux (mJy)')

        fig2 = plt.figure(figsize=(10, 8))
        ds_names = ['I', 'Q', 'U', 'V', 'L']
        gs = grid.GridSpec(2, 1, height_ratios=[2,1])
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.05, hspace=0.025)

        if 'V' in self.curves and 'L' in self.curves:
            ax = fig2.add_subplot(gs[0,0], sharex=ax_ds['I'])
            ax.set_ylabel('Flux (mJy)')
            ax.get_xaxis().set_visible(False)
            ax.errorbar(self.time_bc, self.curves['I']*1e3, self.unc_t['I']*1e3, fmt='-', label='I')
            ax.errorbar(self.time_bc, self.curves['V']*1e3, self.unc_t['V']*1e3, fmt='-', label='V')
            ax.errorbar(self.time_bc, self.curves['L']*1e3, self.unc_t['L']*1e3, fmt='-', label='L')
            ax.legend()

        if 'PA' in self.curves and np.any(np.isfinite(self.curves['PA'])):
            ax_pa = fig2.add_subplot(gs[1,0], sharex=ax_ds['I'])
            ax_pa.set_ylabel('PA (deg)')
            ax_pa.errorbar(self.time_bc, self.curves['PA']*180/np.pi, self.unc_t['PA']*180/np.pi, fmt='.')
            ax_pa.set_xlabel('Time (s)')
            ax_pa.set_ylim([-180, 180])

        if 'a' in self.curves:
            fig3 = plt.figure(figsize=(10, 8))
            gs = grid.GridSpec(1, 3)
            plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.05, hspace=0.025)
            param_names = ['S1GHz', 'a', 'q']
            ax = {param_names[0]: fig3.add_subplot(gs[0,0])}
            for i in range(1, len(param_names)):
                ax[param_names[i]] = fig3.add_subplot(gs[0,i], sharex=ax[param_names[0]])
            for name in param_names:
                ax[name].errorbar(self.time_bc, self.curves[name], self.unc_t[name], fmt='.')
                ax[name].set_title(name)

        if self.dm_curve is not None:
            for i in range(len(self.subsets)):
                fig = plt.figure(f'Subset {i}')
                ax = fig.add_subplot()
                ax.plot(self.dm_curve[i].dms, self.dm_curve[i].peak_snrs)
                ax.set_xlabel('DM')
                ax.set_ylabel('Peak SNR')

        if fname is None:
            plt.show()
        else:
            fig1.savefig(fname + '_dynspec.png')
            fig2.savefig(fname + '_lightcurve.png')


