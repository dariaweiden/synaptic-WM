"""
adapted from Matthijs Pals' code
https://github.com/Matthijspals/STSP/blob/master/stp_ocl_implementation.py
adjusted by Wouter Kruijne to work with nengo 4.0.0
"""
from nengo.neurons import LIF
from nengo.dists import Uniform, Choice
from nengo.learning_rules import LearningRuleType
from nengo.builder.learning_rules import Builder, get_pre_ens
from nengo.builder import Signal
from nengo.builder.operator import Operator, DotInc
import numpy as np

### deleted by me :)

###############################################
#   new LIF neuron with calcium and resources #
###############################################

class stpLIF(LIF):
    """LIF neuron with Short-Term Plasticity (STP)."""
    probeable = ("spikes", "voltage", "refractory_time", "resources", "calcium")

    def __init__(self, tau_x=0.2, tau_u=1.5, U=0.2, **lif_args):
        """Initialize STP parameters and inherit LIF dynamics."""
        super().__init__(**lif_args)
        self.tau_x = tau_x  # Resource recovery time constant
        self.tau_u = tau_u  # Calcium decay time constant
        self.U = U          # Baseline release probability

    def make_state(self, n_neurons, rng, dtype):
        """Initialize states for LIF and STP."""
        # Standard LIF states
        state = super().make_state(n_neurons, rng, dtype)

        # Add custom STP states
        state["resources"] = np.ones(n_neurons, dtype=dtype)  # Resource availability
        state["calcium"] = np.full(n_neurons, self.U, dtype=dtype)  # Calcium level

        return state

    def step(self, dt, J, output, voltage, refractory_time, resources, calcium):
        """Update LIF dynamics and STP dynamics."""
        # Call the parent LIF step method to update voltage and spikes
        super().step(dt, J, output, voltage, refractory_time)

        # Update resources (x) and calcium (u) for STP dynamics
        dx = dt * ((1 - resources) / self.tau_x - calcium * resources * output)
        du = dt * ((self.U - calcium) / self.tau_u + self.U * (1 - calcium) * output)

        # Update states in place
        resources += dx
        calcium += du


### TODO: Matthijs builder.registered stpLIF... but I think I don't need to do that

#######################################################################
#  New learning rule, uses calcium and resources to change connection #
#######################################################################
#create new learning rule to model short term plasticity (only works if pre-ensemble has neuron type StpLIF)
class STP(LearningRuleType):
    """STP learning rule.
    Modifies connection weights according to the calcium and resources of the neuron presynaptic
    """
    modifies = 'weights'
    probeable = ('delta', 'calcium', 'resources')

    def __init__(self):
        super(STP, self).__init__(size_in=0)

class SimSTP(Operator):
    r"""Calculate connection weight change according to the STP rule.
    Implements the STP learning rule of the form:
    .. math:: omega_{ij} = ((u_i * x_i) / U_i) * omega_{ij-initial}
    where
    * :math:`\omega_{ij}` is the connection weight between the two neurons.
    * :math:`u_i` is the calcium level of the presynaptic neuron.
    * :math:`x_i` is the resources level of the presynaptic neuron.
    * :math:`U_i` is the baseline calcium level of the presynaptic neuron.
    * :math:`\omega_{ij-initial}` is the initial connection weight between the two neurons.
    Parameters
    ----------
    weights : Signal
        The connection weight matrix, :math:`\omega_{ij}`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta ((u_i * x_i) / U_i) * initial_omega_{ij} - omega_{ij}`.
    calcium : Signal
        The calcium level of the presynaptic neuron, :math:`u_i`.
    resources : Signal
        The resources level of the presynaptic neuron, :math:`x_i`.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.
    Attributes
    ----------
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    calcium : Signal
        The calcium level of the presynaptic neuron, :math:`u_i`.
    resources : Signal
        The resources level of the presynaptic neuron, :math:`x_i`.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    weights : Signal
        The connection weight matrix, :math:`\omega_{ij}`.
    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[weights, calcium, resources]``
    4. updates ``[delta]``
    """

    def __init__(self, calcium, resources, weights, delta,
                 tag=None):
        super(SimSTP, self).__init__(tag=tag)
        self.sets = []
        self.incs = []
        self.reads = [weights, calcium, resources]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def weights(self):
        return self.reads[0]
        
    @property
    def calcium(self):
        return self.reads[1]
    
    @property
    def resources(self):
        return self.reads[2]
     
    def _descstr(self):
        return '%s' % (self.delta)       

    def make_step(self, signals, dt, rng):
        weights = signals[self.weights]
        delta = signals[self.delta]
        init_weights = self.weights.initial_value
        calcium = signals[self.calcium]
        resources = signals[self.resources]
        U=self.calcium.initial_value
        def step_simstp():
            # perform update
                delta[...] = ((calcium * resources)/U) * init_weights - weights
            
        return step_simstp
    
@Builder.register(STP)
def build_stp(model, stp, rule):
    """Builds a `.STP` object into a model.
   
    Parameters
    ----------
    model : Model
        The model to build into.
    stp : STP
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.STP` instance.
    """

    conn = rule.connection
    calcium = model.sig[get_pre_ens(conn).neurons]['calcium']
    resources = model.sig[get_pre_ens(conn).neurons]['resources']

    model.add_op(SimSTP(calcium,
                        resources,
                        model.sig[conn]['weights'],
                        model.sig[rule]['delta'],
                        ))

    # expose these for probes
    model.sig[rule]['calcium'] = calcium
    model.sig[rule]['resources'] = resources
