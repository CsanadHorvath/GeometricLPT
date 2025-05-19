import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pkl
from glob import glob
from astropy.time import Time
import yaml
import os
# from dstools.dynamic_spectrum import DynamicSpectrum
import h5py

def StokesCircular(RR, RL, LR, LL):
    I = np.real((RR + LL) / 2)
    Q = np.real((LR + RL) / 2)
    U = np.real(1j*(RL - LR) / 2)
    V = np.real((RR - LL) / 2)
    return I, Q, U, V

def StokesLinear(XX, XY, YX, YY):
    I = np.real((XX + YY) / 2)
    Q = np.real((XX - YY) / 2)
    U = np.real((YX + XY) / 2)
    V = np.real((YX - XY) / 2 * 1j)
    return I, Q, U, V

def StokesNothing(I, Q, U, V):
    return np.real(I), np.real(Q), np.real(U), np.real(V)

def MJD2GPS(times):
    return Time(times/(60*60*24), scale="utc", format="mjd").gps

def WriteYAMLandCSV(data, directory, obsname):
    if data['DS'].ndim == 3:
        ds = data['DS']
    else:
        ds = data['DS_MED']

    unique, counts = np.unique(np.diff(data['FREQS']), return_counts=True)
    unique_diffs = np.diff(unique)
    if len(unique) != 1 and np.max(unique_diffs) > 0.001:
        raise Exception('Non-consecutive frequency channels!!!!')
    
    feeds = {'ASKAP':StokesLinear, 'MeerKAT':StokesLinear, 'MWA':StokesLinear, 'VLA':StokesCircular}
    if data['TELESCOPE'] in feeds:
        feed = feeds[data['TELESCOPE']]
    else:
        feed = StokesNothing
    iquv = feed(*[ds[:,:,i] for i in (0,1,2,3)])
    iquv_names = ['I', 'Q', 'U', 'V']
    for i in [0,1,2,3]:
        times = MJD2GPS(data['TIMES'])
        obsid = str(int(MJD2GPS(data['TIMES'][0])))[0:10]
        yaml_fname = f'{obsname}-{obsid}-{iquv_names[i]}.yaml'
        csv_fname = f'{obsname}-{obsid}-{iquv_names[i]}_dyn_dynamic_spectrum.csv'

        with open(directory+'/'+yaml_fname, 'w')as f:

            f.write(f'Apply barycentric correction: true\n')
            f.write(f'Dynamic spectrum:\n')
            f.write(f'  Centre of lowest channel (MHz): {data["FREQS"][0]*1e-6}\n')
            f.write(f'  Channel width (MHz): {(data["FREQS"][1]-data["FREQS"][0])*1e-6}\n')
            f.write(f'  Input file: {csv_fname}\n')
            f.write(f'  Sample time (s): {times[1] - times[0]}\n')
            f.write(f'  T0 (s): {times[0]}\n')
            f.write(f'  Transpose: true\n')
            f.write(f'ObsID: {obsid}\n')
            f.write(f'Reference frequency (MHz): high\n')
            f.write(f'Telescope: {data["TELESCOPE"]}\n')
            f.write(f'RFI Mask:\n')
            f.write(f'  Value: 0.0\n')
            f.write(f'RA: 18.6505\n')
            f.write(f'Dec: -10.5304\n')
            f.write(f'Padding: DM\n')
    
        dynspec = iquv[i]
        dynspec[np.isnan(dynspec) | np.isinf(dynspec)] = 0
        np.savetxt(directory+'/'+csv_fname, dynspec, delimiter=" ", encoding='UTF-8')
        print(csv_fname)

def PKL_to_YAMLCSV(fname, outdir):
    if os.path.isdir(fname):
        fnames = glob(f'{fname}/*.pkl')
        fnames = [i for i in fnames if "demo" not in i]
    else:
        fnames = [fname]

    for fname in fnames:
        with open(fname, 'rb') as f:
            data = pkl.load(f)
        data_list = SplitEpochs(data)
        obsname = os.path.basename(fname)[:-4]
        for data in data_list:
            WriteYAMLandCSV(data, outdir, obsname)

def SplitEpochs(data, sigma=0.01):
    diffs = np.round(np.diff(data['TIMES']) / sigma).astype(np.int64) * sigma
    unique, counts = np.unique(diffs, return_counts=True)
    # print(unique)
    # print(counts)
    start_idx = 0
    tstep = np.min(unique)
    # print(tstep)
    data_list = []
    while start_idx < len(diffs):
        while start_idx < len(diffs) and diffs[start_idx] != tstep:
            start_idx += 1
        end_idx = start_idx
        while end_idx < len(diffs) and diffs[end_idx] == tstep:
            end_idx += 1
        new_data = data.copy()
        new_data['DS'] = data['DS'][start_idx:end_idx+1]
        new_data['TIMES'] = data['TIMES'][start_idx:end_idx+1]
        # print(start_idx, end_idx)
        start_idx = end_idx
        data_list.append(new_data)
    return data_list

def HDF5_to_YAMLCSV(fname, outdir):
    f = h5py.File(fname, 'r')

    data = {
        'DS': np.squeeze(np.array(f['flux'][()], dtype=np.complex64)),
        'TIMES': np.array(f['time'][()], dtype=float),
        'FREQS': np.array(f['frequency'][()], dtype=float),
        'TELESCOPE': 'VLA'}
    
    data_list = SplitEpochs(data)
    obsname = 'VLA_L'
    for data in data_list:
        WriteYAMLandCSV(data, outdir, obsname)

def HDF5_to_YAMLCSV_bruh(fname):
    # Create DS object
    ds = DynamicSpectrum(ds_path='/home/septagonic/Documents/VLA_P/VLA_P_dynspec.hdf5')

    # Plot Stokes I dynamic spectrum with real visibilities and color-scale clipped at 20 mJy
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    ds.plot_ds(stokes='I', cmax=3000, imag=False, fig=fig, ax=axs[0,0])
    ds.plot_ds(stokes='Q', cmax=1000, imag=False, fig=fig, ax=axs[0,1])
    ds.plot_ds(stokes='U', cmax=1000, imag=False, fig=fig, ax=axs[1,0])
    ds.plot_ds(stokes='V', cmax=1000, imag=False, fig=fig, ax=axs[1,1])
    print(np.min(ds.time))
    print(np.max(ds.time))
    print(ds.tmin)
    print(ds.tmax)
    print(ds._timelabel)
    plt.show()

    # print(ds)
    # print(ds.data['I'].shape)
    # print(ds.freq)
    # print(ds.time)

if __name__ == '__main__':
    # PKL_to_YAMLCSV('/home/septagonic/Documents/j1912-44/dynspec', '/home/septagonic/Documents/j1912-44/yaml_csv')
    HDF5_to_YAMLCSV('/home/septagonic/Documents/VLA_L/GPM1839-L-1S1M.ds', '/home/septagonic/Documents/VLA_L/csv_yaml')
