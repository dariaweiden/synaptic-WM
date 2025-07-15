import matplotlib.pyplot as plt
import numpy as np
import nengo
from nengo.utils.matplotlib import rasterplot
from nengo.utils.ensemble import sorted_neurons


from model import WMModel  # assumes model.py is in the same directory
from analysis_tools import cat_analysis_1item

nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'

def simdata2dict(sim, probes):
    return {
        "t": sim.trange(),
        "mem": sim.data[probes["mem"]],
        "inh": sim.data[probes["inh"]],
        "mem_sp": sim.data[probes["mem_sp"]],
        "inh_sp": sim.data[probes["inh_sp"]],
        "resources": sim.data[probes["resources"]],
        "calcium": sim.data[probes["calcium"]],
    }

def main():
    # Instantiate and run the model with default configs
    cfg_dict = dict(
        representation='cat',
        D = 4,
        ## with ping:
        # stim_times = [(0.05,0.255), (0.755, 0.855)  ],
        # stim_vals  = [1, 'ping'],
        ## without ping:
        stim_times = [(0.05,0.255)],
        stim_vals  = [1],
    )

    model = WMModel(**cfg_dict)     # sets parameters
    model.define_model()            # defines model architecture
    sim, probes = model.run_model(run_time=1.5) # runs simulation

    # Retrieve results
    results = simdata2dict(sim, probes)
    cat_analysis_1item(model, results, target=cfg_dict['stim_vals'][0], plot=True)

if __name__ == "__main__":
    main()
