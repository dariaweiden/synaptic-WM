import numpy as np
import nengo
from nengo.dists import Uniform

# own-written stsp module:
from stsp import stpLIF, STP
import encoders

### poisson noise function:
def _make_poisson_noise(rate, n_neurons):
    def noise_func(t):
        dt = 0.001  # default Nengo dt unless otherwise changed
        p = rate * dt
        return np.random.binomial(1, p, size=n_neurons).astype(float)
    return noise_func

class WMModel:
    def __init__(self, **kwargs):
        # Default parameters, overwritten by any passed kwargs
        self.config = cfg = {
            # architecture:
            "representation" : 'cat', # or 'circ'
            "D": 6,
            "Nm": 1000,
            "Ni": 100,

            ## connections:
            "m_to_i": 0.003,
            "i_to_m": 0.002,
            "s_mei": 0.01,
            "s_mie": 0.01,

            # noise parameters:
            "noise_rate": 50,
            "noise_weight": 0.5,
            "seed": None,

            # define stimuli:
            "stim_times": [(0.05,0.255), (1.05,1.15)],
            "stim_vals" : [0, 'ping'],
            # (default) params for cat and circ encoding:
            "p_nonsel" : 0.05,
            # (default) params for circular encoding:
            "sigma": 1.0,
            "stim_sigma": 30,
        }
        # update config with the passed on kwargs:
        cfg.update(kwargs)

        assert self.config['representation'] in ['cat','circ']

        ### representation-specific things
        if self.config['representation'] == 'cat':
            E, make_stim_func = encoders.categorical_E(
                D=cfg["D"], Nm=cfg["Nm"], p_nonsel=cfg["p_nonsel"]
            )
            cfg["encoders"] = E
            cfg["input_func"] = make_stim_func(
                cfg["stim_times"], cfg["stim_vals"] )

        if self.config['representation'] == 'circ':
            # define input represenation and function:
            E, make_stim_func = encoders.circular_E(
                D=cfg["D"], Nm=cfg["Nm"], p_nonsel=cfg['p_nonsel'],
                sigma=cfg["sigma"], stim_sigma=cfg["stim_sigma"] )
            
            cfg["encoders"] = E
            cfg["input_func"] = make_stim_func(cfg["stim_times"], cfg["stim_vals"])

        self.sim = None
        self.probes = {}
        self.model = None
        return

    def define_model(self):
        """ builds the basic model architecture """
        cfg = self.config
        model = nengo.Network(seed=cfg["seed"], label="WMModel")

        with model:

            # Inputs #

            ### Input node presenting stimuli to the network at set times:
            stim_node = nengo.Node(cfg["input_func"], label="visual input")
            ### noise_node presenting random spikes to the network.
            noise_node = nengo.Node(_make_poisson_noise(cfg["noise_rate"], cfg["Nm"]))

            # Ensembles #
            self.mem = mem = nengo.Ensemble(cfg["Nm"], cfg["D"],
                encoders=cfg["encoders"],
                neuron_type=stpLIF(),
                intercepts=Uniform(0.01, 0.1),
                eval_points=Uniform(0, 1.1),
                radius=1,
                label="memory"
            )

            self.inh = inh = nengo.Ensemble( cfg["Ni"], dimensions=1,
                encoders=Uniform(0.01, 1.0),
                intercepts=Uniform(0.01, 0.1),
                radius=1,
                label="inhibitory"
            )

            # Connections # 

            ## input connection (encoders take care of this)
            nengo.Connection(stim_node, mem, transform=1)

            ## injected spiking background noise:
            nengo.Connection(noise_node, mem.neurons, 
                synapse=None, 
                transform=cfg["noise_weight"])

            ## simple recurrent connection with stsp
            Mmm = nengo.Connection(mem, mem,
               transform=1, synapse=0.01,
               eval_points=Uniform(0, 1.1),
               learning_rule_type=STP(),
               solver=nengo.solvers.LstsqL2(weights=True))

            # excitation to inhib and v.v.
            Mei=nengo.Connection(mem.neurons, inh.neurons,
                transform=cfg["m_to_i"] * (1 / cfg["Nm"]) * np.ones((cfg["Ni"], cfg["Nm"])),
                synapse=cfg["s_mei"])

            Mie = nengo.Connection(inh.neurons, mem.neurons,
                transform=-cfg["i_to_m"] * (1 / cfg["Ni"]) * np.ones((cfg["Nm"], cfg["Ni"])),
                synapse=cfg["s_mie"])


            # Probes
            self.probes["mem"] = nengo.Probe(mem, synapse=0.01)
            self.probes["inh"] = nengo.Probe(inh, synapse=0.01)
            self.probes["mem_sp"] = nengo.Probe(mem.neurons)
            self.probes["inh_sp"] = nengo.Probe(inh.neurons)
            self.probes["resources"] = nengo.Probe(mem.neurons, "resources")
            self.probes["calcium"] = nengo.Probe(mem.neurons, "calcium")

            ### this dummy node can be used to get decoding weights for the memory population:
            dummy = nengo.Node(size_in=cfg["D"])
            # probe_dec = nengo.Probe(dummy, 'input', synapse=None)
            self.decoder_connection = nengo.Connection(self.mem, dummy, synapse=None)


        self.model = model

        return model

    def run_model(self, run_time):
        if self.model is None:
            self.define_model()

        with nengo.Simulator(self.model, seed=self.config["seed"], progress_bar=None) as sim:
            sim.run(run_time)
        ## store decoding weights:
        self.decoding_weights = sim.data[self.decoder_connection].weights
        self.sim = sim

        return self.sim, self.probes

