import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr


def ms_to_timepoints(ms, ms_start=-100, ms_step=5):
    timepoint = (ms - ms_start) // ms_step + 1
    return timepoint

# Load the .npy files
meg_rdm = "../Data/Cichy/Cichy_MEG_RDM_Rescaled.npy"
layer_activation_rdm = "./CLIPViT_visual_encoder_24layers_RDM_cichy.npy"

start_ms, end_ms, ms_step = -100, 1000, 1
start_time = ms_to_timepoints(ms = start_ms, ms_start = start_ms, ms_step=ms_step)
end_time = ms_to_timepoints(ms = end_ms, ms_start = start_ms, ms_step=ms_step)

print("start_time:", start_time, "end_time:", end_time)

print("Loading Things MEG RDM")
all_rdms = np.load(meg_rdm, allow_pickle=True)
meg_mean_rdm = np.mean(all_rdms, axis=0)
meg_mean_rdm = np.nan_to_num(meg_mean_rdm)

print("Loading Layer RDMs")
layer_activation_rdm = np.load(layer_activation_rdm)

# Initialize variables
n_timepoints = end_time - start_time + 1
n_layers = layer_activation_rdm.shape[0]
n_images = layer_activation_rdm.shape[1]

print(f"n_timepoints: {n_timepoints}, n_layers: {n_layers}, n_images: {n_images}")
print("meg_mean_rdm shape:", meg_mean_rdm.shape)
print("layer_activation_rdm shape:", layer_activation_rdm.shape)

# Function to extract the upper triangle excluding the diagonal
def extract_upper_triangle(arr):
    if arr.ndim == 3:
        indices = np.triu_indices(arr.shape[1], k=1)  # Get the indices for the upper triangle excluding the diagonal
        upper_triangles = arr[:, indices[0], indices[1]]
    elif arr.ndim == 2:
        indices = np.triu_indices(arr.shape[0], k=1)  # Get the indices for the upper triangle excluding the diagonal
        upper_triangles = arr[indices]
    else:
        raise ValueError("Input array must be 2D or 3D")
    return upper_triangles

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def min_max_scale(weights):
    min_weight = np.min(weights)
    max_weight = np.max(weights)
    scaled_weights = (weights - min_weight) / (max_weight - min_weight)
    return scaled_weights


# Function to compute the weighted average RDM
def weighted_average(weights, layer_rdms):
    # Compute the weighted sum
    weighted_sum = np.tensordot(weights, layer_rdms, axes=(0, 0))
    # Compute the sum of the weights
    sum_of_weights = np.sum(weights)
    # Compute the weighted average
    weighted_avg = weighted_sum / sum_of_weights
    return weighted_avg

# Prepare cross-validation and ridge regression parameters
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ridge = Ridge()
param_grid = {'alpha': np.logspace(-6, 6, 13)}
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Initialize array to store best weights for each timepoint
optimal_weights = np.zeros((n_timepoints, n_layers))
print("output weights shape:", optimal_weights.shape)

# Loop through each timepoint
for t in tqdm(range(n_timepoints), desc='Timepoints'):
    best_score = -np.inf
    meg_rdm_t = meg_mean_rdm[t]
    # Outer cross-validation
    for train_index, test_index in outer_cv.split(np.arange(n_images)):
        # Split layer_activation_rdm and meg_mean_rdm_t using the generated indices
        X_train_full = layer_activation_rdm[:, train_index, :][:, :, train_index]
        X_test_full = layer_activation_rdm[:, test_index, :][:, :, test_index]
        meg_start = max(0, t-10)
        meg_end = min(end_time, t+10)
        # print("meg_start:", meg_start, "meg_end:", meg_end)
        y_train_full = meg_mean_rdm[meg_start:meg_end][:, train_index, :][:, :, train_index].mean(axis=0)
        y_test_full = meg_mean_rdm[meg_start:meg_end][:, train_index, :][:, :, train_index].mean(axis=0)
        # print("y_train_full shape:", y_train_full.shape)
        # print("y_test_full shape:", y_test_full.shape)
        
        # Extract upper triangles after splitting
        X_train = extract_upper_triangle(X_train_full)
        X_test = extract_upper_triangle(X_test_full)
        y_train = extract_upper_triangle(y_train_full)
        y_test = extract_upper_triangle(y_test_full)
        
        # Transpose X_train and X_test to match dimensions
        X_train = X_train.T  # Shape (24, num_train_samples)
        X_test = X_test.T  # Shape (24, num_test_samples)
        
        grid_search = GridSearchCV(ridge, param_grid, cv=inner_cv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        weights = best_model.coef_

        # Normalize the weights to sum to 1
        weights = softmax(weights)


        # Reshape weights for weighted RDM calculation
        weights_reshaped = weights.reshape(n_layers, 1, 1)

        # Compute the weighted RDM for the test set
        weighted_rdms_test = weighted_average(weights_reshaped, layer_activation_rdm).squeeze()

        weight_rdms_test_ut = extract_upper_triangle(weighted_rdms_test)
        meg_rdm_t_ut = extract_upper_triangle(meg_rdm_t)

        # Evaluate with Spearman correlation
        rho, _ = spearmanr(weight_rdms_test_ut, meg_rdm_t_ut)
        
        if rho > best_score:
            best_score = rho
            best_weights = weights


    optimal_weights[t] = best_weights
    print(optimal_weights[t])

print("Optimal weights computed for each timepoint.")
np.save("CLIPViT_visual_encoder_24layers_optimal_weights_cichy.npy", optimal_weights)