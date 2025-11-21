import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from BinaryModel import BinaryModel, GaussianBeam
from BinaryModel import Fold
from BinaryModelMCMC import MinMaxWidths

# Creates the J1912 plot in an overly complicated way

p_orbit = 0.1681198936*24*60*60     # Orbit period in seconds
p_pulse = 319.34903   # Pulse period in seconds
t0_orbit = 0.45*p_orbit  # GPS time of reference orbit 
t0_pulse = 0.45*p_pulse  # GPS time of reference pulse

p_pulse_list = [p_pulse, p_pulse, p_orbit / (p_orbit/p_pulse + 1)]
t0_pulse_list = [t0_pulse, t0_pulse, t0_pulse - p_pulse_list[2] * 0.56]
params_list = np.array([
    [-1.76044263, -0.67962464,  1.04219588,  0.99072053, -0.54927727,  0.98048662],
    [3.34455173, 3.5029956 , 0.27424975,  0.09988638, 3.48755945,  0.10006158],
    [1.91354805, 1.83820053, 0.30394509,  1.32158127, 0.6000568 ,  0.24610691]])
lower_list = np.array([
    [-2.16969104, -0.9720775,   0.89549305,  0.7376763,  -0.8455281,   0.6832874],
    [3.30955428, 3.40196959, 0.25751315,  0.07478352, 3.38545448,  0.07469623],
    [1.86413182, 1.78891518, 0.29526796,  1.30344556, 0.53292912,  0.22438826]])
upper_list = np.array([
    [-1.36590774, -0.44740877,  1.17792421,  1.22879775, -0.29324626,  1.2688053],
    [3.36925148, 3.57296215, 0.2895043 ,  0.11982779, 3.55829672,  0.12013933],
    [1.96887287, 1.89299644, 0.31017248,  1.3387369 , 0.71335633,  0.26569805]])
ranges_list = np.abs(lower_list - upper_list) / 2

print(params_list)

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
sigma *= 1
is_pulse = flux > 0.0028

model = BinaryModel(fit_coeff=True, fit_wbeam=True, fit_wmod=True)
model.SetData(time, flux, is_pulse, sigma)
model.SetBeamFunc(GaussianBeam, symetric=True)
model.SetModFunc (GaussianBeam, symetric=None)

# fig, axs = plt.subplots(2, 4, width_ratios=[10,10,10,0.5], height_ratios=[10,8])
fig, axs = plt.subplots(1, 2, width_ratios=[10,0.5], squeeze=False)
origin_x = [0.5, 0.5, 0.5]
origin_y = [0, 0, 0]
letters = ['A', 'A', 'B']
small_letters = ['a', 'b', 'c']

for i in range(1):
    ax = axs[0,i]
    print(p_pulse_list[i])
    model.SetTiming(t0_pulse_list[i], p_pulse_list[i], t0_orbit, p_orbit)
    model.SetFitParameters(params_list[i])
    model.PlotData(ax, 1/2/np.pi, 1/2/np.pi, 1)
    im = model.PlotPrediction(ax, 1/2/np.pi, 1/2/np.pi)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmid = (xlim[1] + xlim[0]) / 2
    ymid = (ylim[1] + ylim[0]) / 2
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    ax.set_xlim([xmid - xrange/4, xmid + xrange/4])
    ax.set_ylim([ymid - yrange/4, ymid + yrange/4])
    xticks = np.linspace(xmid - xrange*0.2, xmid + xrange*0.2, 5)
    yticks = np.linspace(ymid - yrange*0.2, ymid + yrange*0.2, 5)
    xlabels = np.linspace(-0.2, 0.2, 5)
    ylabels = np.linspace(-0.2, 0.2, 5)
    ax.set_xticks(xticks, [f'{x:g}' for x in xlabels])
    ax.set_yticks(yticks, [f'{y:g}' for y in ylabels])
    # ax.set_xlabel(f'Period {letters[i]}$^*$ phase')
    ax.set_xlabel(f'Spin phase')

#     ax = axs[1,i]
#     ax.axis('off')
#     wbeam = model.wbeam
#     wmod = model.wmod
#     wbeam_range, wmod_range = MinMaxWidths(model, lower_list[i], upper_list[i])
#     text = \
# f'''{small_letters[i]})
#       $\\theta = {np.ceil(params_list[i,2]*180/np.pi):.0f} \pm {np.ceil(ranges_list[i,2]*180/np.pi):.0f} ^\\circ$
#       $\\phi   = {np.ceil(params_list[i,3]*180/np.pi):.0f} \pm {np.ceil(ranges_list[i,3]*180/np.pi):.0f} ^\\circ$
#       $\\sigma = {np.ceil(params_list[i,4]*180/np.pi):.0f} \pm {np.ceil(ranges_list[i,4]*180/np.pi):.0f} ^\\circ$
#       $\\psi   = {np.ceil(params_list[i,5]*180/np.pi):.0f} \pm {np.ceil(ranges_list[i,5]*180/np.pi):.0f} ^\\circ$
#       $w_B     = {np.ceil(wbeam           *180/np.pi):.0f} \pm {np.ceil(wbeam_range     *180/np.pi):.0f} ^\\circ$
#       $w_M     = {np.ceil(wmod            *180/np.pi):.0f} \pm {np.ceil(wmod_range      *180/np.pi):.0f} ^\\circ$'''
#     ax.text(0, 0, text, ma='left')

# for i in range(1, 3):
#     axs[0,i].set_yticks([], [])

axs[0,0].set_ylabel('Orbital phase')

plt.colorbar(im, axs[0,-1], label='Predicted flux')
axs[0,-1].set_yticks(axs[0,-1].get_ylim(), ['Min', 'Max'])
# axs[1,3].axis('off')

# plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.tight_layout()

plt.show()