import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import pickle as pkl
import BinaryModel as bm
from DynspecGroup import DynspecGroup

class BinaryModelMCMC():
    def __init__(self, model:bm.BinaryModel, nwalkers=32, nsteps=500):
        print(nwalkers, nsteps)
        self.model = model
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.pguess = model.GetFitParameters()
        self.ndim = self.pguess.size
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.model.LogPosterior, args={'use_priors':True}, a=10)
        self.initial_positions = [self.pguess + 1e-4 * np.random.randn(self.ndim) for _ in range(self.nwalkers)]
        self.sampler.run_mcmc(self.initial_positions, nsteps, progress=True)

    def GetSamples(self):
        self.samples = self.sampler.get_chain(discard=100, thin=15, flat=True)
        # Assuming 'sampler' is your emcee sampler and 'samples' is the array of MCMC samples
        self.log_prob_samples = self.sampler.get_log_prob(discard=100, thin=15, flat=True)
        self.max_prob_index = np.argmax(self.log_prob_samples)
        self.best_fit_parameters = self.samples[self.max_prob_index]

        self.median_parameters = np.median(self.samples, axis=0)
        self.lower_1sigma = np.percentile(self.samples, 16, axis=0)
        self.upper_1sigma = np.percentile(self.samples, 84, axis=0)

        return self.samples

    def Print(self, fname=None):
        if fname is None:
            args = {}
        else:
            args = {'file':open(fname, 'w')}

        print(f'log P = {self.model.LogPosterior(self.best_fit_parameters)}', **args)
        print("Best fit parameters (MLE):", self.best_fit_parameters, **args)
        print("Median parameters:", self.median_parameters, **args)
        print("1σ credible interval:", **args)
        print("Lower bound:", self.lower_1sigma, **args)
        print("Upper bound:", self.upper_1sigma, **args)

        if fname is not None:
            args['file'].close

    def Plot(self, fname=None):
        labels = self.model.pmath[self.model.pmask]
        fig = plt.figure(figsize=(15,15))
        corner.corner(self.samples, labels=labels, fig=fig)
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname)

def generate_combinations(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("The input lists must have the same length.")
    
    # Recursive helper function
    def helper(index, current_combination):
        if index == len(list1):
            combinations.append(current_combination)
            return
        
        # Include the value from the first list at the current index
        helper(index + 1, current_combination + [list1[index]])
        
        # Include the value from the second list at the current index
        helper(index + 1, current_combination + [list2[index]])
    
    combinations = []
    helper(0, [])
    return combinations


def MinMaxWidths(model:bm.BinaryModel, pmin, pmax):
    param_combs = generate_combinations(pmin, pmax)
    wbeam_list = []
    wmod_list = []
    coeff_list = []
    for params in param_combs:
        model.SetFitParameters(params)
        model.Predict()
        wbeam_list.append(model.wbeam)
        wmod_list.append(model.wmod)
        coeff_list.append(model.coeff)
    min_wbeam = np.min(wbeam_list)
    max_wbeam = np.max(wbeam_list)
    min_wmod = np.min(wmod_list)
    max_wmod = np.max(wmod_list)
    min_coeff = np.min(coeff_list)
    max_coeff = np.max(coeff_list)
    print(min_wbeam*180/np.pi, max_wbeam*180/np.pi, min_wmod*180/np.pi, max_wmod*180/np.pi)
    print((max_wbeam-min_wbeam)*180/np.pi, (max_wmod-min_wmod)*180/np.pi, max_coeff-min_coeff)
    return max_wbeam-min_wbeam, max_wmod-min_wmod

if __name__=='__main__':
    redo = True
    dname = 'mcmc_results_bruh'

    spin_idx = 1
    orbit_idx = 1
    beam_shape = 'gaussian'
    mod_shape = 'gaussian'
    lighthouse = 'mono'
    beam_dir = 'pole'
    south_pole = 'yes'


    code = f's{spin_idx}_o{orbit_idx}_b-{beam_shape}_m-{mod_shape}_{beam_dir}_{south_pole}'
    pkl_fname = f'{dname}/{code}_mcmc.pkl'
    txt_fname = f'{dname}/{code}_mcmc.txt'
    best_fit_fname = f'{dname}/{code}_best_fit.png'

    if redo:
        corner_fname = f'{dname}/{code}_mcmc.png'

        t0_pulse_list = [275091226.17263716, 275091441.2599692, 275091526.40357095]
        p_pulse_list = [1318.1957, 1265.2196, 1216.3372]
        t0_orbit = 275095359
        p_orbit = 31482.3 * orbit_idx

        if orbit_idx == 1:
            t0_pulse = t0_pulse_list[spin_idx]
            p_pulse = p_pulse_list[spin_idx]
        else:
            t0_pulse = t0_pulse_list[0]
            p_pulse = p_orbit / (p_orbit/p_pulse_list[0] + spin_idx)

        init_params = np.array([0.05111426, -0.01539476, -1.74713391,  0.90768997, -0.0123257,   1.06216085])

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

        model = bm.BinaryModel(fit_coeff=True, fit_wbeam=True, fit_wmod=True)
        model.SetTiming(t0_pulse, p_pulse, t0_orbit, p_orbit)
        model.SetData(time, flux, is_pulse, sigma)
        model.SetFitParameters(init_params)
        model.SetMode(beam_shape, mod_shape, south_pole, lighthouse, beam_dir)
        
        model.FlipX()
        model.Plot(fscale=0.13, scale=1/(2*np.pi))
        print('theta'  , model.theta  , 180/np.pi * model.theta  )
        print('phi'    , model.phi    , 180/np.pi * model.phi    )
        print('sigma'  , model.sigma  , 180/np.pi * model.sigma  )
        print('psi'    , model.psi    , 180/np.pi * model.psi    )
        print('alpha_0', model.alpha_0, 180/np.pi * model.alpha_0)
        print('beta_0' , model.beta_0 , 180/np.pi * model.beta_0 )
        print('wbeam'  , model.wbeam  , 180/np.pi * model.wbeam  )
        print('wmod'   , model.wmod   , 180/np.pi * model.wmod   )
        print('coeff'  , model.coeff)
        # MinMaxWidths(model, [0.04830025, -0.01744726, -1.75259628,  0.90344793, -0.01313435, 1.04386986], [0.05286246, -0.01332588, -1.73374129, 0.91464894, -0.01178076,  1.10295479])
        plt.show()
        # exit()

        if orbit_idx == 1:
            mask = np.abs(model.orbit_phase) < 0.2
        elif orbit_idx == 2:
            mask = (np.abs(model.orbit_phase) < 0.1) | (np.abs(model.orbit_phase) > 0.4)
        print(model.ChiSquared(mask))
        # exit()

        mcmc = BinaryModelMCMC(model)
        mcmc.GetSamples()
        mcmc.Print(txt_fname)
        mcmc.Plot(corner_fname)

        with open(pkl_fname, 'wb') as f:
            pkl.dump(mcmc, f)

        model.SetFitParameters(mcmc.best_fit_parameters)
        model.Plot(best_fit_fname, fscale=40)
    else:
        with open(pkl_fname, 'rb') as f:
            mcmc = pkl.load(f)

        mcmc.Plot()
