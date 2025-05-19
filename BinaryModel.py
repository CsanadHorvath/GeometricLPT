import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

def any_none(*args):
    return any([x is None for x in args])

def all_none(*args):
    return all([x is None for x in args])

def WrapValues(value, min_range, max_range):
    range_size = max_range - min_range
    wrapped_value = (value - min_range) % range_size + min_range
    return wrapped_value

def Fold(t, t0, period):
    '''Returns: period number, period residual'''
    p_num = np.round((t - t0) / period)
    residual = t - t0 - p_num * period
    return p_num, residual

def Dot(a, b):
    return np.sum(a*b, axis=-1, keepdims=True)

def Mag(vec):
    return np.sqrt(Dot(vec, vec))

def Argh(alpha, beta, theta, phi, sigma, psi):
    shape = alpha.shape
    alpha = alpha.reshape(-1)
    beta = beta.reshape(-1)
    mu = np.array([np.cos(beta)*np.sin(phi), np.sin(beta)*np.sin(phi), np.cos(phi)*np.ones_like(alpha)]).T
    r_MD = np.array([np.cos(theta)*np.cos(alpha), np.sin(alpha), -np.sin(theta)*np.cos(alpha)]).T
    r_LOS = np.array([np.sin(psi)*np.cos(sigma), np.sin(psi)*np.sin(sigma), np.cos(psi)])
    omega = 25
    omega_WD = np.array([0, 0, omega])
    omega_MD = np.array([np.sin(theta), np.zeros_like(theta), np.cos(theta)])
    B = 3*r_MD*Dot(mu, r_MD) - mu
    v = np.cross(omega_WD - omega_MD, r_MD)
    E = np.cross(v, B)
    E_mag = np.sqrt(Dot(E, E))
    print(np.max(Dot(E/E_mag, r_LOS)))
    return np.arccos(Dot(E/E_mag, r_LOS)).reshape(shape), 1#E_mag.reshape(shape)


def rotation_matrix(axis, angle):
    """Return the rotation matrix for rotating about the given axis by the given angle."""
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.sum(axis**2))
    x, y, z = axis
    c = np.cos(angle)
    a = 1 - c
    s = np.sin(angle)
    return np.array([
        [x*x*a +   c, x*y*a - z*s, x*z*a + y*s],
        [x*y*a + x*s, y*y*a +   c, y*z*a - x*s],
        [x*z*a - y*s, y*z*a + x*s, z*z*a +   c]])

def rotate_vector(vector, axis, angle):
    """Rotate vector by the given angle about the given axis."""
    rot_matrix = rotation_matrix(axis, angle)
    return np.dot(rot_matrix, vector)


def GaussianBeam(angle, width, symetric=None):
    width = width / 5
    angle = np.abs(WrapValues(angle, -np.pi, np.pi))
    amp = np.exp(-angle**2 / (2*width**2))
    if symetric is not None:
        amp2 = np.exp(-(angle-np.pi)**2 / (2*width**2))
        if symetric:
            amp += amp2
        else:
            amp -= amp2
    return amp

def CosineBeam(angle, width, symetric=None):
    angle = np.abs(WrapValues(angle, -np.pi, np.pi))
    amp = np.cos(np.pi * angle / width / 2)**2
    amp[angle > width] = 0
    if symetric is not None:
        amp2 = np.cos(np.pi * (np.pi-angle) / width / 2)**2
        amp2[(np.pi-angle) > width] = 0
        if symetric:
            amp += amp2
        else:
            amp -= amp2
    return amp

def WavyBeam(angle, width, symetric=None):
    wave = np.cos(20 * angle)**2
    return GaussianBeam(angle, width, symetric) * wave

def StepBeam(angle, width, symetric=None):
    angle = np.abs(WrapValues(angle, -np.pi, np.pi))
    amp = np.ones_like(angle)
    amp[angle > width] = 0
    if symetric is not None:
        amp2 = np.ones_like(angle)
        amp2[(np.pi-angle) > width] = 0
        if symetric:
            amp += amp2
        else:
            amp -= amp2
    return amp

class BinaryModel():
    def __init__(self, fit_coeff=True, fit_wbeam=True, fit_wmod=True, mode=0):
        self.mode          = mode
        self.fit_coeff     = fit_coeff
        self.fit_wbeam     = fit_wbeam
        self.fit_wmod      = fit_wmod
        self.has_data      = False
        self.has_timing    = False
        self.has_geometry  = False
        self.has_alphabeta = False
        self.use_priors    = True
        self.pnum          = 10
        self.pmask         = np.ones(self.pnum, dtype=bool)
        self.pmask[6]      = not self.fit_coeff
        self.pmask[7]      = not self.fit_wbeam
        self.pmask[8]      = not self.fit_wmod
        self.pmask[9]      = False
        self.ptext = np.array([   'alpha_0' ,    'beta_0' ,    'theta' ,    'phi' ,    'sigma' ,    'psi' , 'coeff', 'wbeam',  'wmod', 'r_em'])
        self.pmath = np.array(['$\\phi_\\text{orb}^{(0)}$', '$\\phi^{(0)}$', '$i$', '$\\alpha$', '$\\phi_0$', '$\\zeta$', '$A$'  , '$W_\\text{spin}$',  '$W_\\text{orb}$', '$r_{{em}}$'])
        self.pmin  = np.array([       -np.pi,       -np.pi,      -np.pi,       0.0,      -np.pi,       0.0,    0.01,    0.01,    0.01, 0])
        self.pmax  = np.array([        np.pi,        np.pi,       np.pi,     np.pi,       np.pi,     np.pi,  np.inf, 4*np.pi, 4*np.pi, 1])
        self.pidx  = {self.ptext[idx]:idx for idx in range(len(self.ptext))}

        self.alpha_min = None
        self.alpha_max = None
        self.alpha_num = None
        self.beta_min  = None
        self.beta_max  = None
        self.beta_num  = None

        self.alpha_0   = None
        self.beta_0    = None
        self.theta     = None
        self.phi       = None
        self.sigma     = None
        self.psi       = None
        self.coeff     = 1.0
        self.wbeam     = None
        self.wmod      = None
        self.r_em      = 0.1

        self.lighthouse_mode = 'mono'
        self.beam_dir = 'pole'
        self.south = 1

        self.SetBeamFunc(GaussianBeam, symetric=True)
        self.SetModFunc(GaussianBeam, symetric=None)

    def SetPriors(self, **kwargs):
        for key in kwargs:
            idx = self.pidx[key]
            self.pmin[idx] = kwargs[key][0]
            self.pmax[idx] = kwargs[key][1]

    def SetWhatToFit(self, **kwargs):
        for key in kwargs:
            idx = self.pidx[key]
            self.pmask[idx] = kwargs[key]

    def FlipX(self):
        self.alpha_0 = WrapValues(self.alpha_0 - np.pi, -np.pi, np.pi)
        self.beta_0  = WrapValues(self.beta_0  - np.pi, -np.pi, np.pi)
        self.sigma   = WrapValues(self.sigma   - np.pi, -np.pi, np.pi)
        self.theta = -self.theta

    def FlipZ(self):
        self.alpha_0 = WrapValues(self.alpha_0 - np.pi, -np.pi, np.pi)
        self.beta_0  = WrapValues(self.beta_0  - np.pi, -np.pi, np.pi)
        self.psi = -self.psi
        self.theta = -self.theta

    def SetBeamFunc(self, beam_func, *args, **kwargs):
        if type(beam_func) is str:
            beam_func = {'step':StepBeam, 'gaussian':GaussianBeam, 'step':StepBeam, 'wavy':WavyBeam}[beam_func]
        self.beam_func = beam_func
        self.beam_args = args
        self.beam_kwargs = kwargs

    def SetModFunc(self, mod_func, *args, **kwargs):
        if type(mod_func) is str:
            mod_func = {'step':StepBeam, 'gaussian':GaussianBeam, 'step':StepBeam, 'wavy':WavyBeam}[mod_func]
        self.mod_func = mod_func
        self.mod_args = args
        self.mod_kwargs = kwargs

    def SetLighthouseMode(self, lighthouse_mode):
        self.lighthouse_mode = lighthouse_mode

    def SetSouthPole(self, south):
        if type(south) is str:
            self.south = {'yes':1, 'no':0, 'inv':-1}[south]
        else:
            self.south = south

    def SetBeamDir(self, beam_dir):
        self.beam_dir = beam_dir

    def SetMode(self, beam_shape=None, mod_shape=None, south_pole=None, lighthouse=None, beam_dir=None):
        if beam_shape is not None:
            self.SetBeamFunc(beam_shape)
        if mod_shape is not None:
            self.SetModFunc(mod_shape)
        if south_pole is not None:
            self.SetSouthPole(south_pole)
        if lighthouse is not None:
            self.SetLighthouseMode(lighthouse)
        if beam_dir is not None:
            self.SetBeamDir(beam_dir)

    def Nu(self, alpha=None, beta=None):
        '''Angle between beam and M-dwarf'''
        return np.arccos(
            np.cos(beta) * np.sin(self.phi) * np.cos(self.theta) * np.cos(alpha) +
            np.sin(beta) * np.sin(self.phi) * np.sin(alpha) -
            np.cos(self.phi) * np.sin(self.theta) * np.cos(alpha))

    def Mu(self, alpha=None, beta=None):
        '''Angle between Earth and beam'''
        match self.beam_dir:
            case 'pole': return np.arccos(
                np.sin(self.psi) * np.sin(self.phi) * np.cos(self.sigma - beta) +
                np.cos(self.psi) * np.cos(self.phi))
            case 'md': return np.arccos(
                np.cos(self.theta) * np.cos(alpha) * np.sin(self.psi) * np.cos(self.sigma) +
                np.sin(alpha) * np.sin(self.psi) * np.sin(self.sigma) -
                np.sin(self.theta) * np.cos(alpha) * np.cos(self.psi))
            case 'ecme':
                shape = alpha.shape
                alpha = alpha.reshape(-1)
                beta = beta.reshape(-1)
                mu = np.array([np.cos(beta)*np.sin(self.phi), np.sin(beta)*np.sin(self.phi), np.cos(self.phi)*np.ones_like(alpha)]).T
                r_MD = np.array([np.cos(self.theta)*np.cos(alpha), np.sin(alpha), -np.sin(self.theta)*np.cos(alpha)]).T
                r_LOS = np.array([np.sin(self.psi)*np.cos(self.sigma), np.sin(self.psi)*np.sin(self.sigma), np.cos(self.psi)])
                omega = 25
                omega_WD = np.array([0, 0, omega])
                omega_MD = np.array([np.sin(self.theta), np.zeros_like(self.theta), np.cos(self.theta)])
                B = 3*r_MD*Dot(mu, r_MD) - mu
                v = np.cross(omega_WD - omega_MD, r_MD)
                E = np.cross(v, B)
                E_mag = np.sqrt(Dot(E, E))
                return np.arccos(Dot(E/E_mag, r_LOS)).reshape(shape)

    def Beam(self, mu=None, width=None, alpha=None, beta=None):
        if mu is None:
            if alpha is None:
                alpha = self.alpha
            if beta is None:
                beta = self.beta
            self.mu = self.Mu(alpha, beta)
            mu = self.mu
        if width is None:
            # Determining beam and modulation widths if needed
            if self.fit_wbeam:
                self.mu = self.Mu(self.alpha, self.beta)
                mu_pulse = self.mu[self.is_pulse]
                self.wbeam = np.max(mu_pulse)
                if self.wbeam < self.pmin[self.pidx['wbeam']]:
                    self.wbeam = self.pmin[self.pidx['wbeam']]
            width = self.wbeam
        return self.beam_func(mu, width, *self.beam_args, **self.beam_kwargs)
        # angle, mag = AngEBMD(alpha, beta, self.theta, self.phi, self.sigma,self.psi)
        # angle, mag = Argh(alpha, beta, self.theta, self.phi, self.sigma,self.psi)
        # angle, mag, _ = BRUH(alpha, beta, self.theta, self.phi, self.sigma,self.psi, self.r_em)
        return self.beam_func(angle, width, *self.beam_args, **self.beam_kwargs) * mag
    
    def Mod(self, nu=None, width=None):
        if nu is None:
            self.nu = self.Nu(self.alpha, self.beta)
            nu = self.nu
        if width is None:
            # Determining beam and modulation widths if needed
            if self.fit_wmod:
                self.nu = self.Nu(self.alpha, self.beta)
                nu_pulse = self.nu[self.is_pulse]
                if np.any(nu_pulse <= np.pi/2):
                    wmod_1 = np.max(nu_pulse[nu_pulse <= np.pi/2])
                else:
                    wmod_1 = 0.0
                if np.any(nu_pulse > np.pi/2):
                    wmod_2 = np.pi - np.min(nu_pulse[nu_pulse > np.pi/2])
                else:
                    wmod_2 = 0.0
                self.wmod = max(wmod_1, wmod_2)
                if self.wmod < self.pmin[self.pidx['wmod']]:
                    self.wmod = self.pmin[self.pidx['wmod']]
            width = self.wmod
        return self.mod_func(nu, width, *self.mod_args, **self.mod_kwargs)
    
    def SetData(self, time, flux, is_pulse=None, unc=None):
        self.time = time
        self.flux = flux
        if is_pulse is None:
            self.is_pulse = np.ones(self.time.shape, dtype=bool)
        else:
            self.is_pulse = is_pulse
        if unc is None:
            self.unc = 1.0
        else:
            self.unc = unc
        self.has_data = True

    def GetData(self):
        return self.time, self.flux, self.is_pulse

    def SetTiming(self, t0_spin, p_spin, t0_orbit, p_orbit):
        self.t0_spin    = t0_spin
        self.p_spin     = p_spin
        self.t0_orbit   = t0_orbit
        self.p_orbit    = p_orbit
        self.has_timing = True

    def GetTiming(self):
        return self.t0_spin, self.p_spin, self.t0_orbit, self.p_orbit
        
    def SetParameters(self, alpha_0=None, beta_0=None, theta=None, phi=None, sigma=None, psi=None, coeff=None, wbeam=None, wmod=None, r_em=None):
        if alpha_0 is not None: self.alpha_0      = alpha_0
        if beta_0  is not None: self.beta_0       = beta_0
        if theta   is not None: self.theta        = theta
        if phi     is not None: self.phi          = phi
        if sigma   is not None: self.sigma        = sigma
        if psi     is not None: self.psi          = psi
        if coeff   is not None: self.coeff        = coeff
        if wbeam   is not None: self.wbeam        = wbeam
        if wmod    is not None: self.wmod         = wmod
        if r_em    is not None: self.r_em         = r_em
        self.has_geometry = True

    def GetParameters(self):
        return self.alpha_0, self.beta_0, self.theta, self.phi, self.sigma, self.psi, self.coeff, self.wbeam, self.wmod, self.r_em
    
    def SetFitParameters(self, arr):
        params = np.full(self.pnum, None, dtype=object)
        params[self.pmask] = arr
        self.SetParameters(*params)

    def GetFitParameters(self):
        return np.array(self.GetParameters(), dtype=float)[self.pmask]
    
    def SetAlphaBeta(self, alpha, beta, is_pulse=None):
        self.alpha = alpha
        self.beta = beta
        if is_pulse is None:
            self.is_pulse = np.ones(self.alpha.shape, dtype=bool)
        else:
            self.is_pulse = is_pulse
        self.has_alphabeta = True

    def GetAlphaBeta(self):
        self.FoldData()
        if self.has_geometry:
            self.alpha = 2 * np.pi * self.orbit_phase + self.alpha_0
            self.beta = 2 * np.pi * self.spin_phase + self.beta_0
        else:
            self.alpha = 2 * np.pi * self.orbit_phase
            self.beta = 2 * np.pi * self.spin_phase
        self.has_alphabeta = True
        if self.has_alphabeta:
            return self.alpha, self.beta
        else:
            return None, None
    
    def FoldData(self):
        if self.has_data and self.has_timing:
            self.orbit_num, self.orbit_res = Fold(self.time, self.t0_orbit, self.p_orbit)
            self.spin_num , self.spin_res  = Fold(self.time, self.t0_spin , self.p_spin )
            self.orbit_phase = self.orbit_res / self.p_orbit
            self.spin_phase  = self.spin_res  / self.p_spin

    def Func(self, alpha, beta):
        mu = self.Mu(alpha, beta)
        nu = self.Nu(alpha, beta)
        match self.lighthouse_mode:
            case 'binary'   : pred = (self.Beam(mu) + self.south * self.Beam(mu-np.pi)) * (self.Mod(nu) + self.south * self.Mod(nu-np.pi))
            case 'mono'     : pred = self.Beam(mu) * self.Mod(nu) + self.south * self.Beam(mu-np.pi) * self.Mod(nu-np.pi)
            case 'isotropic': pred = self.Mod(nu)
            case _: raise Exception(f"Invalid lighthouse_mode '{self.lighthouse_mode}'")
        return pred

    def Predict(self, alpha=None, beta=None):
        if self.has_alphabeta:
            self.GetAlphaBeta()
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta

        self.pred = self.Func(alpha, beta)
        # Calculating  and applying the flux coefficient if needed
        if self.fit_coeff and self.has_data:
            pred = self.Func(self.alpha, self.beta)
            self.coeff = np.sum(pred * self.flux / self.unc**2) / np.sum(pred**2 / self.unc**2)
            if not np.isfinite(self.coeff) or self.coeff < 0:
                self.coeff = 1.0
        self.pred *= self.coeff

        return self.pred

    def LogPosterior(self, params=None, use_priors=None):
        if params is not None:
            self.SetFitParameters(params)
        if use_priors is None:
            use_priors = self.use_priors

        self.Predict()
        
        # Calculating the log likelyhood that the model fits the data
        if self.has_data and self.has_geometry and self.has_alphabeta:
            # Likelyhood without assumptions
            # sigma2 = self.unc ** 2
            n = 1
            sigma2 = (n*self.flux**(n-1)*self.unc) ** 2
            # self.log_likelyhood = -0.5 * np.sum(((self.flux - np.abs(self.pred))**2 / sigma2 + np.log(2*np.pi*sigma2)))
            self.log_likelyhood = -0.5 * np.sum(((self.flux**n - np.abs(self.pred)**n)**2 / sigma2 + np.log(2*np.pi*sigma2)))
            # Prior with parameter ranges
            if use_priors:
                params = self.GetFitParameters()
                if np.all((params >= self.pmin[self.pmask]) & (params < self.pmax[self.pmask])):
                    self.log_prior = 0.0
                else:
                    self.log_prior = -np.inf
            else:
                self.log_prior = 0.0
            # Posterior log probability
            if not np.isfinite(self.log_prior):
                self.log_posterior = -np.inf
            else:
                self.log_posterior = self.log_prior + self.log_likelyhood
        else:
            self.log_prior = None
            self.lof_posterior = None
            self.log_likelyhood = None
        return self.log_posterior
    
    def ChiSquared(self, mask=None):
        '''Calculate the chi^2 and reduced chi^2 statistics of the model.
        Returns: chi^2, reduced chi^2'''
        if mask is None:
            mask = np.ones(self.time.size, dtype=bool)
        self.Predict()
        self.chi2 = np.nansum((self.flux[mask] - np.abs(self.pred[mask]))**2 / self.unc[mask]**2)
        self.chi2red = self.chi2 / (self.time[mask].size - self.pnum)
        return self.chi2, self.chi2red
    
    def SetSampleGrid(self, alpha_min=None, alpha_max=None, alpha_num=None, beta_min=None, beta_max=None, beta_num=None):
        if alpha_min is not None: self.alpha_min = alpha_min
        if alpha_max is not None: self.alpha_max = alpha_max
        if alpha_num is not None: self.alpha_num = alpha_num
        if beta_min  is not None: self.beta_min  = beta_min
        if beta_max  is not None: self.beta_max  = beta_max
        if beta_num  is not None: self.beta_num  = beta_num

        if self.alpha_num is None: self.alpha_num =  200
        if self.beta_num  is None: self.beta_num  =  200
        if self.has_data:
            self.GetAlphaBeta()
            self.alpha_min = np.min(self.alpha)
            self.alpha_max = np.max(self.alpha)
            self.beta_min  = np.min(self.beta)
            self.beta_max  = np.max(self.beta)
        else:
            self.alpha_min = -np.pi + self.alpha_0
            self.alpha_max =  np.pi + self.alpha_0
            self.beta_min  = -np.pi + self.beta_0
            self.beta_max  =  np.pi + self.beta_0

        self.alpha_sample = np.linspace(self.alpha_min, self.alpha_max, self.alpha_num)
        self.beta_sample = np.linspace(self.beta_min , self.beta_max , self.beta_num )
        self.beta_grid, self.alpha_grid = np.meshgrid(self.beta_sample, self.alpha_sample)
    
    def PlotAngles(self, ax:Axes, xscale=1, yscale=1, mumin=None, mumax=None):
        self.SetSampleGrid()
        mu = self.Mu(self.alpha_grid, self.beta_grid) * 180 / np.pi
        nu = self.Nu(self.alpha_grid, self.beta_grid) * 180 / np.pi
        pcol = ax.pcolor(self.beta_sample*xscale, self.alpha_sample*yscale, mu, zorder=-1000000, cmap='Blues', vmin=mumin, vmax=mumax)
        if np.max(nu) - np.min(nu) < 45:
            levels = np.arange(0, 181, 5)
        else:
            levels = np.arange(0, 181, 15)
        levels = levels[(levels >= np.min(nu)) & (levels <= np.max(nu))]
        levels = None
        cont = ax.contour(self.beta_sample*xscale, self.alpha_sample*yscale, nu, cmap='Reds', levels=levels, zorder=1000000000)
        return pcol, cont

    def PlotPrediction(self, ax:Axes, xscale=1, yscale=1, vmin=None, vmax=None):
        self.SetSampleGrid()
        pred = self.Predict(alpha=self.alpha_grid, beta=self.beta_grid) * 1e3

        if np.min(pred) < 0:
            cmap = 'bwr'
            if vmax is None:
                vmax = np.max(np.abs(pred))
            if vmin is None:
                vmin = -vmax
        else:
            cmap = 'Reds'
            if vmax is None:
                vmax = np.max(pred)
            if vmin is None:
                vmin = 0
        pcol = ax.pcolor(self.beta_sample*xscale, self.alpha_sample*yscale, pred, zorder=-1000000, cmap=cmap, vmin=vmin, vmax=vmax)
        return pcol
            
    def PlotData(self, ax:Axes, xscale=1, yscale=1, fscale=1, flatten=False):
        self.GetAlphaBeta()
        x = self.beta * xscale
        y = self.alpha * yscale
        f = self.flux * fscale

        if flatten:
            y = y - x

        for pulse_idx in np.unique(self.spin_num):
            this_pulse_no_orbit = (self.spin_num == pulse_idx)
            for orbit_idx in np.unique(self.orbit_num[this_pulse_no_orbit]):
                this_pulse = this_pulse_no_orbit & (self.orbit_num == orbit_idx)
                if np.count_nonzero(this_pulse) > 0:
                    z = y[this_pulse][0]
                    ax.plot(x[this_pulse], f[this_pulse] + y[this_pulse], c='k', zorder=z+0.0001, lw=0.5)
                    ax.fill_between(x[this_pulse], y[this_pulse], f[this_pulse] + y[this_pulse], color='k', alpha=0.3, zorder=z)
                    # sel = this_pulse & (self.flux > 0.035)
                    # ax.plot(x[sel], f[sel] + y[sel], 'r.', zorder=z+0.0001, lw=0.5)

    def Plot(self, fname=None, fig=None, axs=None, fscale=1, scale = 180 / np.pi):
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(13, 5), width_ratios=[8,10], sharex=True, sharey=True)
        pcol0 = self.PlotPrediction(axs[0], xscale=scale, yscale=scale)
        if self.has_data:
            self.PlotData(axs[0], xscale=scale, yscale=scale, fscale=fscale)
        pcol1, cont1 = self.PlotAngles(axs[1], xscale=scale, yscale=scale)
        fig.colorbar(pcol0, ax=axs[0], label='Predicted flux desnity (mJy)')
        fig.colorbar(pcol1, ax=axs[1], label='$\\beta$ (deg)')
        fig.colorbar(cont1, ax=axs[1], label='$\\beta_\\text{MD}$ (deg)')
        # axs[0].set_xlabel('Spin angle $\\alpha$ (deg)')
        # axs[1].set_xlabel('Spin angle $\\alpha$ (deg)')
        # axs[0].set_ylabel('Orbit angle $\\beta$ (deg)')
        axs[0].set_xlabel('Spin phase')
        axs[1].set_xlabel('Spin phase')
        axs[0].set_ylabel('Orbit phase')
        # if self.has_timing:
        #     secax = axs[0].secondary_xaxis('top', functions=(lambda x: x/360, lambda x: x*360))
        #     secax.set_xlabel('Spin phase')
        # axs[1].set_yticks([])
        plt.tight_layout()
        if fname is None and axs is None:
            plt.show()
        elif fname is not None:
            plt.savefig(fname)
        return fig, axs

if __name__=='__main__':
    from matplotlib.widgets import Slider, Button
    import pickle as pkl
    from DynspecGroup import DynspecGroup

    show_data = False
    lighthouse = 0
    beam_dir = 0
    beam_shape = 0
    mod_shape = 0
    which_data = 'GPM J1839-10'

    shapes = [GaussianBeam, CosineBeam, StepBeam, WavyBeam]

    if which_data == 'GPM J1839-10':
        beat_idx = 1
        t0_pulse_list = [275091226.17263716, 275091441.2599692, 275091526.40357095]
        p_pulse_list = [1318.1957, 1265.2196, 1216.3372]
        t0_pulse = t0_pulse_list[beat_idx]
        p_pulse = p_pulse_list[beat_idx]
        t0_orbit = 275095359
        p_orbit = 31482.3 * 1
        fscale=0.13
    elif which_data == 'J1912-44':
        p_orbit = 0.1681198936*24*60*60     # Orbit period in seconds
        p_pulse = 319.34903   # Pulse period in seconds
        t0_orbit = 0.45*p_orbit  # GPS time of reference orbit 
        t0_pulse = 0.45*p_pulse  # GPS time of reference pulse
        fscale=1.3
    elif which_data == 'ArSco':
        t0_pulse = 0
        p_pulse = 1.95*60
        t0_orbit = 0
        p_orbit = 3.56*60*60
        fscale = 0.005


    fig = plt.figure(figsize=(13, 5))
    model = BinaryModel(fit_wbeam=False, fit_wmod=False, fit_coeff=False)

    if show_data:
        if which_data == 'GPM J1839-10':
            with open('pol_results_meerkat.pkl', 'rb')as f:
                data = pkl.load(f)
            with open('pol_results_askap.pkl', 'rb')as f:
                data += pkl.load(f)
            with open('pol_results_meerkat_bonus.pkl', 'rb')as f:
                data += pkl.load(f)
            time = np.concatenate([x.time_bc for x in data])
            flux = np.concatenate([x.curves['I'] for x in data])
            sigma = np.concatenate([x.unc_t['I'] for x in data])
        elif which_data == 'J1912-44':
            with open('pol_results_j1912-44.pkl', 'rb')as f:
                data = pkl.load(f)
            time = np.concatenate([x.time_bc for x in data])
            flux = np.concatenate([x.curves['I'] for x in data])
            sigma = np.concatenate([x.unc_t['I'] for x in data])
        elif which_data == 'ArSco':
            data = np.loadtxt('/home/septagonic/Documents/round-the-world-exploring/ArSco/J_A+A_611_A66/table1.dat')
            time = data[:,0]*24*60*60
            time -= np.min(time)
            flux = data[:,1]
            flux -= np.min(flux)*2
            sigma = data[:,2]
        
        valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(sigma)
        _, orbit_res = Fold(time, t0_orbit, p_orbit)
        _, pulse_res = Fold(time, t0_pulse, p_pulse)
        if which_data == 'J1912-44':
            valid &= np.abs(orbit_res/p_orbit) < 0.2
            valid &= np.abs(pulse_res/p_pulse) < 0.2
        time = time[valid]
        flux = flux[valid]
        sigma = sigma[valid] * 10
        is_pulse = flux > 0.005
        model.SetData(time, flux, is_pulse, sigma)
        model.SetTiming(t0_pulse, p_pulse, t0_orbit, p_orbit)
        model.SetData(time, flux, is_pulse, sigma)

    def Update(*_):
        params = np.array([s.val for s in sliders]) * np.pi / 180
        print(params)
        model.SetBeamFunc(shapes[beam_shape])
        model.SetModFunc(shapes[mod_shape])
        model.SetParameters(*params[:6], 1, *params[6:])
        model.SetLighthouseMode(['mono', 'binary', 'isotropic'][lighthouse])
        model.beam_dir = ['pole', 'md', 'ecme'][beam_dir]

        fig.clear()
        axs = fig.subplots(1, 2, width_ratios=[8,10])
        model.Plot(fig=fig, axs=axs, fscale=fscale, scale=1/(2*np.pi))
        fig.canvas.draw()

    def LighthouseText():
        return 'Lighthouse: ' + ['mono', 'binary', 'isotropic'][lighthouse]
    
    def BeamDirText():
        return 'beam direction: ' + ['pole', 'md', 'ecme'][beam_dir]
    
    def BeamShapeText():
        return 'Beam shape: ' + ['Gaussian', 'Cosine', 'Step', 'Wavy'][beam_shape]
    
    def ModShapeText():
        return 'Modulation shape: ' + ['Gaussian', 'Cosine', 'Step', 'Wavy'][mod_shape]

    def ToggleLighthouse(*_):
        global lighthouse
        lighthouse = (lighthouse + 1) % 3
        buttons['lighthouse'].label.set_text(LighthouseText())
        Update()

    def ToggleBeamDir(*_):
        global beam_dir
        beam_dir = (beam_dir + 1) % 3
        buttons['beam_dir'].label.set_text(BeamDirText())
        Update()

    def ToggleBeamShape(*_):
        global beam_shape
        beam_shape = (beam_shape + 1) % len(shapes)
        buttons['beam_shape'].label.set_text(BeamShapeText())
        Update()

    def ToggleModShape(*_):
        global mod_shape
        mod_shape = (mod_shape + 1) % len(shapes)
        buttons['mod_shape'].label.set_text(ModShapeText())
        Update()

    ui_fig, ui_axs = plt.subplots(13, 1, figsize=(5, 5))

    sliders = [
        Slider(ax=ui_axs[0], label="$\\alpha 0$", valmin=-180, valmax=180, valinit= 0),
        Slider(ax=ui_axs[1], label="$\\beta 0$" , valmin=-180, valmax=180, valinit= 0),
        Slider(ax=ui_axs[2], label="$\\theta$"  , valmin=-180, valmax=180, valinit=-100.1),
        Slider(ax=ui_axs[3], label="$\\phi$"    , valmin=   0, valmax=180, valinit=52.1),
        Slider(ax=ui_axs[4], label="$\\sigma$"  , valmin=-180, valmax=180, valinit= 0.71),
        Slider(ax=ui_axs[5], label="$\\psi$"    , valmin=   0, valmax=180, valinit=61),
        Slider(ax=ui_axs[6], label="$w_b$"      , valmin=0   , valmax=360, valinit=65),
        Slider(ax=ui_axs[7], label="$w_m$"      , valmin=0   , valmax=360, valinit=70),
        Slider(ax=ui_axs[8], label="$r_{{em}}$" , valmin=   0, valmax=  1, valinit=0.1)]
    
    buttons = {
        'beam_dir'  : Button(ax=ui_axs[-4], label=BeamDirText()),
        'lighthouse': Button(ax=ui_axs[-3], label=LighthouseText()),
        'beam_shape': Button(ax=ui_axs[-2], label=BeamShapeText()),
        'mod_shape' : Button(ax=ui_axs[-1], label=ModShapeText())}
    
    for slider in sliders:
        slider.on_changed(Update)

    buttons['beam_dir'].on_clicked(ToggleBeamDir)
    buttons['lighthouse'].on_clicked(ToggleLighthouse)
    buttons['beam_shape'].on_clicked(ToggleBeamShape)
    buttons['mod_shape'].on_clicked(ToggleModShape)

    Update()
    plt.show()