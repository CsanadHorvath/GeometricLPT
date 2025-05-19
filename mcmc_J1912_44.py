import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from BinaryModel import BinaryModel, GaussianBeam, StepBeam, CosineBeam
from BinaryModel import Fold
from BinaryModelMCMC import BinaryModelMCMC, MinMaxWidths

p_orbit = 0.1681198936*24*60*60     # Orbit period in seconds
p_pulse = 319.34903   # Pulse period in seconds
t0_orbit = 0.45*p_orbit  # GPS time of reference orbit 
t0_pulse = 0.45*p_pulse  # GPS time of reference pulse

# p_pulse = p_orbit / (p_orbit/p_pulse + 1)
# t0_pulse -= p_pulse * 0.56

# x0 = np.array([0.0, 0.0, 0.0,  0.8, 0.0, 1.0])
# x0 = np.array([0.0, 0.0, 0.0,  0.8, 0.0, 1.5])
# x0 = np.array([3.18654064, 2.60711926, 0.2544517,  0.04899234, 2.59694535, 0.05114357]) # poop

# x0 = np.array([ 4.60023315,  2.67716465,  1.46764385, -1.37676256,  2.84421573, -1.37264856])
# x0 = np.array([ 4.55119246,  2.60711926,  0.2544517, -1.25064937,  2.60140656, -1.25312265])
# x0 = np.array([1.91354805, 1.83820053, 0.30394509, 1.32158127, 0.6000568,  0.24610691])

# x0 = [-2.34912292, -1.49327929,  1.03660333,  0.56160501, -1.40884418, 0.62574176]
x0 = [-1.76044263, -0.67962464,  1.04219588,  0.99072053, -0.54927727,  0.98048662]
# x0 = [-2.34912292, -1.49327929,  0.56160501, -1.40884418, 0.62574176]

with open('pol_results_j1912-44.pkl', 'rb')as f:
    data = pkl.load(f)

time = np.concatenate([x.time_bc for x in data])
flux = np.concatenate([x.curves['I'] for x in data])
sigma = np.concatenate([x.unc_t['I'] for x in data])
_, orbit_res = Fold(time, t0_orbit, p_orbit)
_, pulse_res = Fold(time, t0_pulse, p_pulse)
valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(sigma)
valid &= np.abs(orbit_res/p_orbit) < 0.2
valid &= np.abs(pulse_res/p_pulse) < 0.2
flux[~valid] = 0
sigma *= 5
is_pulse = flux > 0.0028
# is_pulse = flux > 0.001

print([x.telescope for x in data])
time = np.sort(time)
unique_times, unique_counts = np.unique(np.diff(time), return_counts=True)
total_time = np.sum(unique_counts[unique_times <= 2] * unique_times[unique_times <= 2])
print(total_time / 60 / 60)
exit()

model = BinaryModel(fit_coeff=True, fit_wbeam=True, fit_wmod=True)
model.SetTiming(t0_pulse, p_pulse, t0_orbit, p_orbit)
model.SetData(time, flux, is_pulse, sigma)
model.SetBeamFunc(GaussianBeam, symetric=True)
model.SetModFunc (GaussianBeam, symetric=None)
model.SetPriors(theta=((59-6)*np.pi/180, (59+6)*np.pi/180))
# model.SetWhatToFit(theta=False)
# model.SetParameters(theta=59*np.pi/180)
model.SetFitParameters(x0)

# model.FlipX()
# model.FlipZ()
model.Plot(fscale=400)
print('theta', 180/np.pi * model.theta)
print('phi', 180/np.pi * model.phi)
print('sigma', 180/np.pi * model.sigma)
print('psi', 180/np.pi * model.psi)
print('alpha_0', 180/np.pi * model.alpha_0)
print('beta_0', 180/np.pi * model.beta_0)
print('wbeam', 180/np.pi * model.wbeam)
print('wmod', 180/np.pi * model.wmod)
print('coeff', model.coeff)
plt.show()

lower = np.array([-2.16969104, -0.9720775,   0.89549305,  0.7376763,  -0.8455281,   0.6832874 ])
upper = np.array([-1.36590774, -0.44740877,  1.17792421,  1.22879775, -0.29324626,  1.2688053 ])

wbeam_range, wmod_range = MinMaxWidths(model, lower, upper)

print(lower * 180 / np.pi)
print(upper * 180 / np.pi)
print(180/np.pi * wbeam_range, 180/np.pi * wmod_range)
# exit()

# mcmc = BinaryModelMCMC(model)
mcmc = BinaryModelMCMC(model, 32, 1000)
# mcmc = BinaryModelMCMC(model, 64, 5000)
mcmc.GetSamples()

dname = 'mcmc_results_j1912-44'
code = 'o1s0_bgs_mga_limited'
mcmc.Print(f'{dname}/{code}_mcmc.txt')
mcmc.Plot(f'{dname}/{code}_mcmc.png')
model.SetFitParameters(mcmc.best_fit_parameters)
model.Plot(f'{dname}/{code}_best_fit.png', fscale=400)
model.SetFitParameters(mcmc.median_parameters)
model.Plot(f'{dname}/{code}_median.png', fscale=400)
plt.show()