import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.xmargin'] = 0

from scipy.signal import butter, filtfilt, hilbert
from nengo.utils.matplotlib import rasterplot
from nengo.utils.ensemble import sorted_neurons

## bandpass filter (for alpha results:)
def bandpass_filter(data, fs, lowcut=8.0, highcut=12.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)

def plot_run(results, savepath=None):
    t = results["t"]

    plt.figure(figsize=(10, 10))

    # Decoded memory output
    plt.subplot(3, 1, 1)
    plt.plot(t, results["mem"])
    plt.title("Decoded Memory Ensemble")
    plt.ylabel("Decoded activity")

    ### stacked mem and inh spikes:
    plt.subplot(3, 1, 2)
    spikes = np.hstack([
        results["mem_sp"],
        results["inh_sp"]
    ])
    colors = (
        ['darkblue'] * results["mem_sp"].shape[1] + 
        ['darkred'] * results["inh_sp"].shape[1] ) 
    rasterplot(t, spikes, colors=colors)
    plt.title("Spikes (in Mem and Inh)")
    # plt.xlabel("Time [s]")
    plt.ylabel("Neuron index")


    ### Neural efficacy:
    plt.subplot(3, 1, 3)
    plt.plot(t, results['decoded_eff'])
    plt.title("Decoded Synaptic efficacy in memory ensemble")
    plt.xlabel("Time [s]")
    plt.ylabel("Decoded synaptic efficacy")

    if savepath is None:
        plt.show()
        return
    plt.savefig(savepath, format="svg")
    plt.close()
    return

def cat_analysis_1item(model, model_results, target=0, window=None, plot=None):
    """Analyzes a categorical 1-item memory trial.
    """

    ## get some basic parameters from the sim:
    cfg = model.config
    t = model_results["t"]
    fs = 1 / (t[1] - t[0]) # frequency (1000Hz)
    stim_times = cfg["stim_times"]
    stim_vals = cfg["stim_vals"]
    D = cfg['D']

    # weight vector: 1 for target, negative weights for nontargets
    stimvec = -np.ones(D)/(D-1); stimvec[target]=1
    
    ## compute decoded neural efficacy (synaptic WM)
    eff = model_results['resources'] * model_results['calcium']
    decoded_eff= np.dot(eff, model.decoding_weights.T)
    model_results['decoded_eff'] = decoded_eff

    # Determine analysis window
    if window is None:
        window = (stim_times[-1][1], t[-1])  # after last stimulus end
    widx = np.where((t >= window[0]) & (t <= window[1]))[0]

    ## selectivity in (persistent) activity
    memvec = np.mean(model_results["mem"][widx, :], 0) # D-dimensional activity vec
    persistent_selectivity = np.dot(memvec, stimvec)

    # Selectivity in (alpha band) activity:
    filtered = bandpass_filter(model_results["mem"], fs)
    power = np.abs(hilbert(filtered, axis=0)) ** 2
    alhpa_memvec = np.mean(power[widx, :], 0) # D-dimensional activity vec
    alpha_selectivity = np.dot(alhpa_memvec, stimvec)

    # Activity-silent selectivity (dropouts below baseline)
    # baseline = np.mean(decoded_eff[:5], axis=0)
    baseline = np.mean(decoded_eff[:5])
    # everything below baseline is (re) set to baseline
    win = decoded_eff[widx, :]
    win[win < baseline] = baseline
    eff_memvec = np.mean(win, 0) # D-dimensional activity vec
    eff_selectivity = np.dot(eff_memvec,stimvec)

    analysis_results_dict = {
        "persistent_sel": persistent_selectivity,
        "alpha_sel": alpha_selectivity,
        "eff_sel": eff_selectivity,
    }

    # print (analysis_results_dict)
    summary = "; ".join(f"{k}={v:.3f}" for k, v in analysis_results_dict.items())
    print (summary)

    # then deal with three plot cases:
    if plot == True: # show plot:
        plot_run(model_results)
    elif plot: # don't show plot but save plot:
        plot_run(model_results, savepath=plot)
    # else, no plot:
    return analysis_results_dict # the three measures, as a dict.

def cat_analysis_2items(model,resuls,target=0, window=None, plot=None):
    # No analysis thought out yet...
    # ...
    # ...but we can plot:
    if plot == True: # show plot:
        plot_run(model_results)
        return
    if plot: # save plot:
        plot_run(model_results, savepath=plot)
        return


    return