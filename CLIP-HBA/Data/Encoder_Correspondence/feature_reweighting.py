import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def combine_activations(weights, activations):

    # combine activations using the weights
    combined_activations = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)

    # compute activation RDM
    activation_rdm = 1 - cosine_similarity(combined_activations)
    np.fill_diagonal(activation_rdm, 0)

    return activation_rdm

def objective_function(weights, activations, target_rdm):
    """Objective function to minimize (negative correlation)."""
    activation_rdm = combine_activations(weights, activations)
    # Flatten the RDMs to compute the correlation
    activation_rdm_flat = activation_rdm[np.triu_indices(activation_rdm.shape[0], k=1)]
    target_rdm_flat = target_rdm[np.triu_indices(target_rdm.shape[0], k=1)]
    # Pearson correlation loss
    return 1-pearsonr(activation_rdm_flat, target_rdm_flat)[0]

def find_optimal_weights(layer_activations, target_rdm):
    """Find the optimal weights using constrained optimization."""
    n_layers = layer_activations.shape[0]
    initial_weights = np.ones(n_layers) / n_layers  # Start with equal weights

    constraints = (
        {'type': 'eq', 'fun': lambda w: 1 - np.sum(w)},  # Sum of weights == 1
        {'type': 'ineq', 'fun': lambda w: w}  # Weights >= 0
    )

    result = minimize(
        objective_function, 
        initial_weights, 
        args=(layer_activations, target_rdm), 
        method='SLSQP', 
        constraints=constraints
    )

    return result.x


# Saving Matrix path
saving_path = "./weighting_matrix/weighting_matrix_cichy.npy"

# Initiate MEG RDMs
meg_rdm_path = "../Data/Cichy/Cichy_MEG_RDM_Rescaled.npy"
meg_rdms = np.load(meg_rdm_path)
# meg_rdms = meg_rdms[:-1, :, :, :] # use the first 3 participants, 4th data quality is not ideal
meg_rdms = np.mean(meg_rdms, axis=0)

# Initiate 24 layer activations
activation_path = "./CLIPViT_visual_encoder_24layers_activations_cichy.npy"
activations = np.load(activation_path) # shape: [24, n_stimuli, n_stimuli]

# Initiate the weighting matrix
weighting_matrix = np.zeros((meg_rdms.shape[0], activations.shape[0]))  # [281, 24]
print(f"Weighting matrix shape: {weighting_matrix.shape}")

for t in tqdm(range(weighting_matrix.shape[0]), desc="Optimizing"):  # Iterate over each timepoint
    target_rdm = meg_rdms[t]  # [1854, 1854] MEG RDM for timepoint t
    weights = find_optimal_weights(activations, target_rdm)
    weighting_matrix[t, :] = weights
    print(f"Timepoint {t}: {np.round(weights, 3)}")
    weight_sum = np.sum(weights)
    print(f"Sum of weights: {weight_sum}")
    np.save(saving_path, weighting_matrix)

# Save the weighting matrix
print(f"Weighting matrix saved to {saving_path}")
