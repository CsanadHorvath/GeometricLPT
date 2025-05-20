import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from DynspecGroup import DynspecGroup
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from astropy.time import Time

def Fold(t, t0, period):
    p_num = np.round((t - t0) / period) # Period number of each time
    residual = t - t0 - p_num * period
    return p_num, residual

def BinMean(x, y, xmin, xmax, n):
    x_bins = np.linspace(xmin, xmax, n) + ((xmax-xmin) / (n-1) / 2)       # The center value of each bin
    bin_idx = np.round((x - xmin) / (xmax - xmin) * (n - 1)).astype(int)  # The bin index of each value in x
    y_bins = np.zeros(n)                                                  # The mean y value in each bin
    count_bins = np.zeros(n)                                              # The number of values in each bin
    for j in range(len(x)):                                               # Sum y in each bin and count them
        y_bins[bin_idx[j]] += y[j]
        count_bins[bin_idx[j]] += 1
    is_nan = (count_bins == 0)                                            # Bins with no values are nan
    y_bins[~is_nan] = y_bins[~is_nan] / count_bins[~is_nan]
    y_bins[is_nan] = np.nan
    return x_bins, y_bins

def WrapValues(value, min_range, max_range):
    range_size = max_range - min_range
    wrapped_value = (value - min_range) % range_size + min_range
    return wrapped_value

def ProcessPA(pa, upa, para, telescope):
    pa = pa * 180 / np.pi
    upa = upa * 180 / np.pi

    if telescope == 'MeerKAT':
        # pa -= WrapValues(para * 180 / np.pi + 180, -180, 180)
        pa -= para * 180 / np.pi
        pa -= 62.46938660751961
    elif telescope == 'ASKAP':
        pa -= 73.35506940019839
        pa *= -1
    elif telescope == 'VLA':
        pa += 33

    pa = WrapValues(pa, -135, 45) + 90
    # pa += 90

    return pa, upa

tel_colors = {'MeerKAT':'green', 'ASKAP':'purple', 'VLA':'orange'} # Choose other colours for telescopes
stokes_colors = {'I':'black', 'L':'red', 'V':'blue'}

def DrawStack(curves, t0_pulse, p_pulse, t0_orbit, p_orbit, ax_stack, axh_mean, axh_pa=None, axv_mean=None, axv_pa=None, xscale=1, yscale=1, stokes=True, telcol=True):    
    global count_test
    
    flux_list = {'I':[], 'L':[], 'V':[], 'PA':[]}
    curve_list = {'I':[], 'L':[], 'V':[], 'PA':[]}
    unc_list = {'I':[], 'L':[], 'V':[], 'PA':[]}
    beta_list = {'I':[], 'L':[], 'V':[], 'PA':[]}
    alpha_list = {'I':[], 'L':[], 'V':[], 'PA':[]}

    total_time = 0

    for observation in curves:
        lc = observation.curves
        unc = observation.unc_t
        t = observation.time_bc
        dt = t[1] - t[0]
        total_time += t[-1] - t[0]
        stokes_params = observation.curves.keys()
        pulse_num, pulse_res = Fold(t, t0_pulse, p_pulse)
        orbit_num, orbit_res = Fold(t, t0_orbit, p_orbit)

        # Travel time through orbital radius
        # corr = -np.cos(orbit_res / p_orbit * 2 * np.pi) * 5
        # pulse_res += corr
        # orbit_res += corr

        x = pulse_res * xscale
        y = orbit_res * yscale

        if telcol and observation.telescope in tel_colors:
            color = tel_colors[observation.telescope]
        else:
            color = 'black'

        for pulse_idx in np.unique(pulse_num):
            this_pulse_no_orbit = (pulse_num == pulse_idx)
            for orbit_idx in np.unique(orbit_num[this_pulse_no_orbit]):
                this_pulse = this_pulse_no_orbit & (orbit_num == orbit_idx)
                if np.count_nonzero(this_pulse) > 0 and stokes_key in stokes_params:
                    s = x[this_pulse] # pulse_res[this_pulse]
                    o = y[this_pulse] # orbit_res[this_pulse]
                    f = lc[stokes_key][this_pulse]
                    f[np.isnan(f)] = 0.0
                    mid_idx = np.argmin(np.abs(s))
                    yy = o#[mid_idx]
                    z = -o[mid_idx]
                    ax_stack.plot(s, f * lc_scale + yy, c='black', zorder=z+0.0001, lw=0.5)
                    ax_stack.fill_between(s, yy, f * lc_scale + yy, color=color, alpha=0.3, zorder=z)
                    # sel = f > 0.005
                    # ax_stack.plot(s[sel], f[sel] * lc_scale + yy[sel], 'r.', zorder=z+0.0001, lw=0.5)
                    for name in ['I', 'L', 'V']:
                        if name in lc:
                            flux_list[name].append(lc['I'][this_pulse])
                            curve_list[name].append(lc[name][this_pulse])
                            unc_list[name].append(unc[name][this_pulse])
                            beta_list[name].append(s)
                            alpha_list[name].append(yy)

                    if 'PA' in lc:
                        valid = lc['L'][this_pulse] / unc['I'][this_pulse] > 7
                        pa, upa = ProcessPA(lc['PA'][this_pulse], unc['PA'][this_pulse], lc['q'][this_pulse], observation.telescope)
                        if axh_pa is not None:
                            axh_pa.errorbar(s[valid], pa[valid], upa[valid], fmt='.', markersize=1, color=color)
                        if axv_pa is not None:
                            axv_pa.errorbar(pa[valid], o[valid], None, upa[valid], fmt='.', markersize=1, color=color)
                        curve_list['PA'].append(pa[valid])
                        beta_list['PA'].append(s[valid])

                    if axv_mean is not None and 'L' in lc and 'V' in lc:
                        valid = (s > -0.1) & ( s < 0.1)
                        if np.any(valid):
                            i_fluence = np.nansum(lc['I'][this_pulse][valid]) * dt
                            l_fluence = np.nansum(lc['L'][this_pulse][valid]) * dt
                            v_fluence = np.nansum(np.abs(lc['V'][this_pulse][valid])) * dt
                            i_unc = np.nansum(unc['I'][this_pulse][valid]) * dt
                            l_unc = np.nansum(unc['L'][this_pulse][valid]) * dt
                            v_unc = np.nansum(unc['V'][this_pulse][valid]) * dt
                            li_frac = l_fluence / i_fluence
                            vi_frac = v_fluence / i_fluence
                            li_unc = np.abs((i_unc/i_fluence + l_unc/l_fluence) * li_frac)
                            vi_unc = np.abs((i_unc/i_fluence + v_unc/v_fluence) * vi_frac)
                            # axv_mean.errorbar(l_fluence, z, None, l_unc, fmt='.', color=color)
                            # axv_mean.errorbar(v_fluence, z, None, v_unc, fmt='.', color=color)
                            axv_mean.errorbar(i_fluence, z, None, i_unc, fmt='.', color=color)

    print(f'Total time = {total_time/60/60} hours')
    pa = np.concatenate(curve_list['PA'])
    beta = np.concatenate(beta_list['PA'])
    print('PA STD:', (np.std(pa[(beta < 0) & (pa < 45)]) + np.std(pa[(beta > 0) & (pa > 45)])) / 2)

    if axh_mean is not None:
        n_bins = 400
        if stokes:
            which_stokes = ['I', 'L', 'V']
        else:
            which_stokes = ['I']
        for name in which_stokes:
            beta_bins = np.empty((len(beta_list[name]), n_bins))
            curve_bins = np.empty((len(beta_list[name]), n_bins))

            for i in range(len(beta_list[name])):
                alpha = alpha_list[name][i]
                beta = beta_list[name][i]
                flux = flux_list[name][i]
                curve = curve_list[name][i]
                valid = (alpha < 0.2) & (alpha > -0.2)
                if p_pulse < 1300 and p_orbit < 40000:
                    valid[beta < 0] &= alpha[beta < 0] < 0
                    valid[beta > 0] &= alpha[beta > 0] > 0
                # valid &= (flux > 0.0025)
                if name == 'V':
                    curve = np.abs(curve)
                beta_bins[i], curve_bins[i] = BinMean(beta[valid], curve[valid], -0.5, 0.5, n_bins)

            curve_mean = np.nanmean(curve_bins, axis=0)
            curve_mean -= np.nanmin(curve_mean)
            curve_mean[np.isnan(curve_mean)] = 0.0
            axh_mean.plot(beta_bins[0], curve_mean*1e3, c=stokes_colors[name])
            if name == 'I':
                mean_flux = curve_mean
                mean_beta = beta_bins[0]
    return mean_flux, mean_beta

def Set_GPMJ1839_10():
    global t0_orbit, t0_pulse, p_orbit, p_pulse, p_spin, stokes_key, lc_scale, plot_angles, curves, xlims, ylim, t0_pulse_list, p_pulse_list

    t0_orbit = 275095359  # GPS time of reference orbit 
    t0_pulse = 275091232  # GPS time of reference pulse
    p_orbit = 31482.3     # Orbit period in seconds
    p_pulse = 1318.1957   # Pulse period in seconds
    orbit_dir = 1
    p_pulse = p_orbit / (p_orbit/p_pulse + 1) * 1
    p_orbit *= 1
    p_spin = p_orbit / (p_orbit/p_pulse + orbit_dir)
    t0_pulse += 0.11 * p_pulse
    stokes_key = 'I'
    lc_scale = 1.0e-1 * 0.5
    plot_angles = False

    xlims = [[-0.18, 0.18], [-0.22, 0.22], [-0.39, 0.39]]
    ylim = [-0.22, 0.22]

    t0_pulse_list = [275091226.17263716, 275091441.2599692-80, 275091526.40357095]
    p_pulse_list = [1318.1957, 1265.2197, 1216.3372]

    with open('pol_results_meerkat.pkl', 'rb')as f:
        curves = pkl.load(f)

    with open('pol_results_askap.pkl', 'rb')as f:
        curves += pkl.load(f)

    with open('pol_results_meerkat_bonus.pkl', 'rb')as f:
        curves += pkl.load(f)

    with open('pol_results_vla_l.pkl', 'rb')as f:
        curves += pkl.load(f)

if __name__=='__main__':

    # Set_J1912_44()
    Set_GPMJ1839_10()
    nperiods = 2

    fig = plt.figure()
    # gspec = GridSpec(3, 5, height_ratios=[1, 1, 3], width_ratios=[2, 2, 2, 1, 1])
    gspec = GridSpec(3, nperiods+1, height_ratios=[1, 1, 3], width_ratios=[2]*nperiods + [1])

    ax_stack  = [plt.subplot(gspec[2, 0])]
    ax_stack += [plt.subplot(gspec[2, i], sharey=ax_stack[0]) for i in range(1,nperiods)]
    axh_mean  = [plt.subplot(gspec[1, 0], sharex=ax_stack[0])]
    axh_mean += [plt.subplot(gspec[1, i], sharex=ax_stack[i], sharey=axh_mean[0]) for i in range(1,nperiods)]
    axh_pa    = [plt.subplot(gspec[0, 0], sharex=ax_stack[0])]
    axh_pa   += [plt.subplot(gspec[0, i], sharex=ax_stack[i], sharey=axh_pa[0]) for i in range(1, nperiods)]

    axv_pa    =  plt.subplot(gspec[2, -1], sharey=ax_stack[0])
    axv_pa.set_xticks([0, 90])
    axv_pa.yaxis.set_tick_params(labelbottom=False, bottom=False)

    if gspec.ncols == 5:
        axv_mean  =  plt.subplot(gspec[2, 3], sharey=ax_stack[0])
        axv_mean.yaxis.set_tick_params(labelbottom=False, bottom=False)
        axv_mean.set_xlabel('Fluence (Jy s)')
    else:
        axv_mean = None

    ax_tel = plt.subplot(gspec[0, nperiods])
    ax_tel.axis('off')
    ax_tel.legend(
        [Line2D([0], [0], color=tel_colors[x], lw=4) for x in tel_colors],
        [x for x  in tel_colors], loc='center')

    ax_stokes = plt.subplot(gspec[1, nperiods])
    ax_stokes.axis('off')
    stokes_names = {'I':'mean I', 'L':'mean L', 'V':'mean |V|'}
    ax_stokes.legend(
        [Line2D([0], [0], color=stokes_colors[x], lw=4) for x in stokes_colors],
        [stokes_names[x] for x in stokes_colors], loc='center')

    axh_mean[0].set_ylabel('Mean flux (mJy)')
    axh_pa[0].set_ylabel('PA (deg)')
    axh_pa[0].set_yticks([0, 90])
    # axh_pa[0].set_yticks([-180, 180])
    axv_pa.set_xlabel('PA (deg)')
    ax_stack[0].set_ylabel('Orbital phase')
    secax = axv_pa.secondary_yaxis('right', functions=(lambda x: x*p_orbit, lambda x: x/p_orbit))
    secax.set_ylabel('Orbital residual (s)')

    # period_names = ['Period A', 'Period B', 'Period C']
    period_names = ['Period A (beat)', 'Period B (spin)', 'Period C']

    for i in range(nperiods):
        ax_stack[i].set_xlabel(f'{period_names[i]} phase')
        ax_stack[i].set_xlim(xlims[i])
        ax_stack[i].set_ylim(ylim)
        axh_mean[i].xaxis.set_tick_params(labelbottom=False, bottom=False)
        axh_pa[i].xaxis.set_tick_params(labelbottom=False, bottom=False)
        secax = axh_pa[i].secondary_xaxis('top', functions=(lambda x: x*p_pulse, lambda x: x/p_pulse))
        secax.set_xlabel(f'{period_names[i]} residual (s)')

    for i in range(1, nperiods):
        ax_stack[i].yaxis.set_tick_params(labelbottom=False, bottom=False)
        axh_mean[i].yaxis.set_tick_params(labelbottom=False, bottom=False)
        axh_pa[i].yaxis.set_tick_params(labelbottom=False, bottom=False)

    plt.subplots_adjust(wspace=0.02, hspace=0.03)

    for i in range(nperiods):
        p_pulse = p_pulse_list[i]
        # p_pulse = p_orbit / (p_orbit/p_pulse_list[0] + i)
        print(p_pulse)
        t0_pulse = t0_pulse_list[i]
        if i == 0:
            DrawStack(curves, t0_pulse, p_pulse, t0_orbit, p_orbit, ax_stack[i], axh_mean[i], axh_pa[i], axv_mean, axv_pa, xscale=1/p_pulse, yscale=1/p_orbit)
        else:
            mean_flux, mean_res = DrawStack(curves, t0_pulse, p_pulse, t0_orbit, p_orbit, ax_stack[i], axh_mean[i], axh_pa[i], xscale=1/p_pulse, yscale=1/p_orbit)

    plt.show()
