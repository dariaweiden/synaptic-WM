import nengo
import numpy as np 
import matplotlib.pyplot as plt 



def _generate_stimuli():
    pass

def _generate_pals_svd():
    ## generate stimuli if they don't exist yet
    ## fit the PCA
    return

def _generate_svd_kruijne():
    ## generate stimuli if they don't exist yet
    ## fit the PCA
    return
    
def svd_E(type='pals'):
    # if type=='pals':
        # if the file exists
        # load it and return it
    return

def categorical_E(D=4, Nm=1000, p_nonsel=0.1):
    """
    Generates a categorical encoder matrix and a corresponding stimulus function.
    
    Parameters:
        D (int): Number of categories / dimensions
        Nm (int): Number of neurons
        p_nonsel (float): Proportion of nonselective neurons
    
    Returns:
        E (np.ndarray): Encoder matrix of shape (Nm, D)
        make_stim_func (callable): Function(stim_times, stim_vals) → stim_func(t)
    """
    E = np.zeros((Nm, D))
    nperstim = int((Nm - p_nonsel * Nm) // D)
    nsel = nperstim * D
    nnonsel = Nm - nsel

    # Nonselective neurons: respond to all categories
    E[:nnonsel, :] = 1

    # Selective blocks: respond only to one category
    start = nnonsel
    for i in range(D):
        E[start:start + nperstim, i] = 1
        start += nperstim

    # Build stim_func generator (Nengo-compatible)
    def make_stim_func(stim_times, stim_vals):
        def stim_func(t):
            for (start, end), val in zip(stim_times, stim_vals):
                if start < t < end:
                    if val == 'ping':
                        return [0.25] * D
                    onehot = np.zeros(D)
                    onehot[int(val)] = 1
                    return onehot
            return np.zeros(D)
        return stim_func

    return E, make_stim_func
def circular_E(D=6, Nm=1000, p_nonsel = 0.1, sigma=1.0, stim_sigma=30):
    """
    Generates a circular encoder matrix, and a corresponding stimulus function.
    
    Parameters:
        D (int): Number of representational dimensions / base angles.
        Nm (int): Number of memory neurons.
        sigma (float): Width of tuning curves in encoder space.
        stim_sigma (float): Width of tuning curve for input vector.
    
    Returns:
        E (np.ndarray): Encoder matrix of shape (Nm, D)
        stim_func (callable): Function(t) → input vector
        decoder_labels (list of str): Labels for decoded dimensions (e.g., ["0°", "30°", ...])
    """    
    # what angles are being represented:
    base_angles = np.linspace(0, 180, D, endpoint=False) 

    E = np.zeros((Nm, D))

    # define number of stimuli:
    nperstim = int( (Nm - p_nonsel * Nm) // D )
    nsel = nperstim * D 
    nnonsel = Nm - nsel
    
    ### fill the encoder matrix:
    E[:nnonsel, :] = 1  # non-selective neurons, respond to all

    ### selective stimuli, use gaussians with circular dist;
    def circular_dist(i, j, D):
        return min(abs(i - j), D - abs(i - j))

    # Build selective encoders, block-by-block:
    start = nnonsel
    for i in range(D):
        for _ in range(nperstim):
            row = np.exp(-0.5 * (np.array([circular_dist(i, k, D) for k in range(D)]) / sigma) ** 2)
            row /= np.linalg.norm(row)
            E[start, :] = row
            start += 1


    # Factory that returns a Nengo-compatible stim_func(t)
    def make_stim_func(stim_times, stim_vals):
        assert len(stim_times) == len(stim_vals)

        def angular_diff(a1, a2):
            return np.abs(((a1 - a2 + 90) % 180) - 90)

        def orientation_to_vector(angle_deg):
            diffs = np.array([angular_diff(angle_deg, b) for b in base_angles])
            stim_vector = np.exp(-0.5 * (diffs / stim_sigma) ** 2)
            return stim_vector / np.linalg.norm(stim_vector)

        # actual nengo input function:
        def stim_func(t):
            for (start, end), angle in zip(stim_times, stim_vals):
                if start < t < end:
                    if angle == 'ping':
                        return base_angles.size * [0.25]
                    return orientation_to_vector(angle)
            return np.zeros(D)
        
        return stim_func

    return E, make_stim_func



