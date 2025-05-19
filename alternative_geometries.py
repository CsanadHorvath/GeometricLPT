from BinaryModel import BinaryModel
from BinaryModel import GaussianBeam, StepBeam
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from DynspecGroup import DynspecGroup

t0_pulse = 275091441.2599692
p_pulse = 1265.2196
t0_orbit = 275095359
p_orbit = 31482.3

model = BinaryModel(fit_wbeam=True, fit_wmod=True, fit_coeff=False)

with open('pol_results_meerkat.pkl', 'rb')as f:
        data = pkl.load(f)
with open('pol_results_askap.pkl', 'rb')as f:
    data += pkl.load(f)
with open('pol_results_meerkat_bonus.pkl', 'rb')as f:
    data += pkl.load(f)
time = np.concatenate([x.time_bc for x in data])
flux = np.concatenate([x.curves['I'] for x in data])
sigma = np.concatenate([x.unc_t['I'] for x in data])
valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(sigma)
time = time[valid]
flux = flux[valid]
sigma = sigma[valid] * 10
is_pulse = flux > 0.005

model.SetData(time, flux, is_pulse, sigma)
model.SetTiming(t0_pulse, p_pulse, t0_orbit, p_orbit)
model.SetData(time, flux, is_pulse, sigma)

titles = ['a) Isotropic emission', 'b) Spin interpulse', 'c) Orbital interpulse']
params = [np.array([0, 0,  60, 127,   0, 45, 1, 90, 60]) * np.pi / 180, # Isotropic
          np.array([0, 0,  53,  90,  85, 60, 1, 78, 77]) * np.pi / 180, # Interpulse
          np.array([0, 0, -80,  33,   0, 69, 1, 89, 38]) * np.pi / 180] # Symmetric orbit
p_orbits = [p_orbit, p_orbit, p_orbit * 2]
beam_funcs = [StepBeam    , GaussianBeam, GaussianBeam]
mod_funcs  = [GaussianBeam, GaussianBeam, GaussianBeam]
beam_sym   = [True, False, None ]
mod_sym    = [None, None , False]


fig, axs = plt.subplots(1, 3)
scale =  1/(2*np.pi)
fscale = 0.13

for i in range(3):
    model.SetBeamFunc(beam_funcs[i], symetric=beam_sym[i])
    model.SetModFunc (mod_funcs [i], symetric=mod_sym [i])
    model.SetParameters(*params[i])
    model.SetTiming(t0_pulse, p_pulse, t0_orbit, p_orbits[i])
    model.PlotData(axs[i], scale, scale, fscale)
    model.PlotPrediction(axs[i], scale, scale)
    axs[i].set_title(titles[i])
    axs[i].set_xlim([-0.49, 0.45])
    axs[i].set_ylim([-0.49, 0.45])

for i in range(1,3):
    axs[i].set_yticks([])

axs[0].set_ylabel('Orbital phase')
axs[1].set_xlabel('Spin phase')

plt.tight_layout()
plt.subplots_adjust(wspace=0.02, hspace=0.03)
plt.show()
