import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

def combine_activations(weights, activations):
    combined_activations = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
    activation_rdm = 1 - cosine_similarity(combined_activations)
    np.fill_diagonal(activation_rdm, 0)
    return activation_rdm

def objective_function(weights, activations, target_rdm):
    activation_rdm = combine_activations(weights, activations)
    activation_rdm_flat = activation_rdm[np.triu_indices(activation_rdm.shape[0], k=1)]
    target_rdm_flat = target_rdm[np.triu_indices(target_rdm.shape[0], k=1)]
    return 1 - pearsonr(activation_rdm_flat, target_rdm_flat)[0]

def find_optimal_weights(layer_activations, target_rdm):
    n_layers = layer_activations.shape[0]
    initial_weights = np.ones(n_layers) / n_layers
    constraints = (
        {'type': 'eq', 'fun': lambda w: 1 - np.sum(w)},  # Sum of weights == 1
        {'type': 'ineq', 'fun': lambda w: w}
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
saving_path = "./weighting_matrix/weighting_matrix_things_cv.npy"

# Load MEG RDMs
meg_rdm_path = "../Data/ThingsMEG_RDMs/THingsMEG_RDM_4P.npy"
meg_rdms = np.load(meg_rdm_path)
meg_rdms = np.mean(meg_rdms, axis=0)

# Load layer activations
activation_path = "./CLIPViT_visual_encoder_24layers_activations.npy"
activations = np.load(activation_path)  # shape: [24, n_stimuli, n_stimuli]

# Initialize weighting matrix
try:
    weighting_matrix = np.load(saving_path)
    saving_path = saving_path.replace(".npy", "_updated.npy")
    print("saving_path:", saving_path)
except FileNotFoundError:
    print("Weighting matrix not found, initializing new matrix.")
    weighting_matrix = np.zeros((meg_rdms.shape[0], activations.shape[0]))  # [281, 24]
    saving_path = saving_path.replace(".npy", "_v0.npy")
    print("saving_path:", saving_path)

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

for t in tqdm(range(weighting_matrix.shape[0]), desc="Optimizing"):
    target_rdm = meg_rdms[t]

    if weighting_matrix[t].sum() != 0:
        print(f"Timepoint {t} already optimized, skipping...")
        continue

    outer_weights = []
    for train_index, test_index in outer_cv.split(np.arange(activations.shape[1])):
        train_activations = activations[:, train_index, :]
        test_activations = activations[:, test_index, :]

        train_rdm = target_rdm[train_index][:, train_index]
        test_rdm = target_rdm[test_index][:, test_index]

        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        best_inner_weight = None
        best_inner_corr = -np.inf

        for inner_train_index, inner_test_index in inner_cv.split(train_index):
            inner_train_activations = train_activations[:, inner_train_index, :]
            inner_train_rdm = train_rdm[inner_train_index][:, inner_train_index]

            weights = find_optimal_weights(inner_train_activations, inner_train_rdm)
            inner_test_rdm = train_rdm[inner_test_index][:, inner_test_index]
            inner_corr = 1 - objective_function(weights, train_activations[:, inner_test_index, :], inner_test_rdm)
            
            if inner_corr > best_inner_corr:
                best_inner_corr = inner_corr
                best_inner_weight = weights
        
        outer_weights.append(best_inner_weight)
    
    final_weights = np.mean(outer_weights, axis=0)
    weighting_matrix[t, :] = final_weights

    print(f"Timepoint {t}: {np.round(final_weights, 3)}")
    np.save(saving_path, weighting_matrix)

print(f"Weighting matrix saved to {saving_path}")
