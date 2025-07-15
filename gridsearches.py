import os, math
import numpy as np 
import joblib 
import multiprocessing
from itertools import product
from single_run import simdata2dict
import nengo
import json, csv

from model import WMModel
from analysis_tools import cat_analysis_1item

# NJOBS = 1 # use 4
NJOBS = 4 # use 4

nengo.rc['progress']['progress_bar'] = 'none'

def _setup_grid_search(base_cfg, loop_values, fldrname, add=False):
    """
    Sets up a folder to store grid search results in. Use JSON to log
    model settings
    """
    if add:
        ### explicitly adding to already existing folder: no problem.
        assert os.path.isdir(fldrname)
        return
    # else
    # make folder; error if it already exists:
    assert not os.path.isdir(fldrname) # delete fldr or use `add=True` if you want to add results to folder!
    os.makedirs(fldrname)

    # make a dummy model with base_cfg as kwargs;
    full_cfg = WMModel(**base_cfg).config
    # delete loop_values full_cfg, as well as matrix/function:
    static_cfg = {
        k: v for k, v in full_cfg.items() 
        if  k not in loop_values and 
            k not in ["encoders","input_func"]
    }

    # next, store a json file with 2 fields (with subfields)
    jsondict = {
        "static_cfg": static_cfg,
        "loop_values": loop_values,
    }
    with open(os.path.join(fldrname, "config.json"), "w") as f:
        json.dump(jsondict, f, indent=4)

    # make a csv table, with a column per loop_values key + 3 analysis outputs
    fieldnames = list(loop_values.keys()) + [
        "persistent_sel", "alpha_sel", "eff_sel"
    ]
    with open(os.path.join(fldrname, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return

def _run_batches(param_dicts, base_cfg, outfolder):
    n_total = len(param_dicts)
    n_batches = math.ceil(n_total / NJOBS)
    # write results from this batch to file:
    csv_path = os.path.join(outfolder, "results.csv")
    ## get fieldnames (in order):
    with open(csv_path, "r", newline="") as f_read:
        reader = csv.reader(f_read)
        fieldnames = next(reader)

    for batch_idx in range(n_batches):
        print(f"\n=== Batch {batch_idx+1}/{n_batches} ===")

        # which param values are in this batch?
        batch = param_dicts[batch_idx * NJOBS : (batch_idx + 1) * NJOBS]

        # spawn a batch of njobs jobs:
        batch_results = joblib.Parallel(n_jobs=NJOBS)(
            joblib.delayed(run_one)(param_dict, base_cfg, outfolder) for param_dict in batch
        )

        # Add batch results to the file
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for result in batch_results:
                row = result["params"].copy()
                row.update(result["analysis"])
                writer.writerow(row)

### define the simulation function:
def run_one(param_dict, base_cfg, outfolder):
    # name this run:
    run_name = "+".join(f"{k}_{v:.3f}" for k, v in param_dict.items())
    plotname = os.path.join(outfolder, run_name + '.svg')
    # set run vals into cfg:
    cfg = base_cfg.copy()
    cfg.update(param_dict)
    # build and run model:
    model = WMModel(**cfg)
    model.define_model()
    sim, probes = model.run_model(run_time=1.75)
    results = simdata2dict(sim, probes)

    # Analysis + plots:
    scores = cat_analysis_1item(model, results,
        target=cfg['stim_vals'][0], 
        plot=plotname )

    print(f"Done: {run_name}")
    return {
        "params": param_dict,
        "analysis": scores
    }

def test_search(outfolder='test_search'):
    base_cfg = {
        "representation": 'cat', 
        "D": 4, 
        "s_mei": 0.01,
        "seed": 2025, 
        "stim_times": [(0.1, 0.25), (0.85, 0.95)],
        "stim_vals": [1, 'ping'],
    }

    loop_values = dict(
        s_mie = [0.01, 0.03], 
        m_to_i = [0.002, 0.01],
        i_to_m = [0.002, 0.01],
    )

    # setup folder to gather results:
    _setup_grid_search(base_cfg, loop_values, outfolder)
    results_csv_path = os.path.join(outfolder, "results.csv")

    # Build list of param dicts (i.e., specify the grid)
    keys, values = zip(*loop_values.items())
    param_dicts = [dict(zip(keys, combo)) for combo in product(*values)]
    _run_batches(param_dicts, base_cfg, outfolder)
    return


def search_1item_ping(outfolder='search_1item_ping'):
    base_cfg = {
        "representation": 'cat', 
        "D": 4, 
        "s_mei": 0.01,
        "seed": 2025, 
        "stim_times": [(0.1, 0.35), (0.85, 0.95)],
        "stim_vals": [1, 'ping'],
    }

    loop_values = dict(
        s_mie = [0.01, 0.02], 
        m_to_i = np.geomspace(0.002, 0.021, num=6).tolist(),
        i_to_m = np.geomspace(0.002, 0.021, num=6).tolist(),
        noise_weight = np.arange(0.0,0.71, 0.1).tolist(),
    )
    # loop_values = dict(
    #     s_mie = [0.01, 0.02], 
    #     m_to_i = np.linspace(0.002, 0.032, num=7).tolist(),
    #     i_to_m = np.linspace(0.002, 0.032, num=7).tolist(),
    #     noise_weight = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75],
    # )

    # setup folder to gather results:
    _setup_grid_search(base_cfg, loop_values, outfolder)
    results_csv_path = os.path.join(outfolder, "results.csv")

    # Build list of param dicts (i.e., specify the grid)
    keys, values = zip(*loop_values.items())
    param_dicts = [dict(zip(keys, combo)) for combo in product(*values)]
    _run_batches(param_dicts, base_cfg, outfolder)
    return

if __name__ == '__main__':
    # test_search()
    # search_1item_ping() # geomspaced grid;
    search_1item_ping(outfolder='search_1item_ping_geomspace')
    # search_1item_ping(outfolder='search_1item_ping_linspace')
