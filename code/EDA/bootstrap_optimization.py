# Code to replace in the bootstrap cell (cell 13)
# Add these imports:
from multiprocessing import Pool, cpu_count
from functools import partial

# Replace the bootstrap_nmf function with this parallelized version:

def _bootstrap_single_iteration(args):
    """Helper function for a single bootstrap iteration (runs in parallel)"""
    i, data, n_components, H_ref, n_timepoints, n_features_original = args
    
    # Resample with replacement
    n_samples = data.shape[0]
    bootstrap_indices = resample(np.arange(n_samples), n_samples=n_samples, random_state=i)
    data_bootstrap = data[bootstrap_indices, :]
    
    # Fit NMF on bootstrap sample
    nmf_bs = NMF(n_components=n_components, random_state=i, max_iter=1000)
    W_bs = nmf_bs.fit_transform(data_bootstrap)
    H_bs = nmf_bs.components_
    
    # Calculate correlation between reference and bootstrap components
    correlations = np.abs(np.corrcoef(H_ref.flatten(), H_bs.flatten()))[0, 1]
    
    # Calculate feature importance (variance in W matrix per feature)
    feature_importance = None
    if W_bs.shape[0] == n_timepoints * n_features_original:
        W_reshaped = W_bs.reshape(n_features_original, n_timepoints, n_components)
        feature_importance = np.var(W_reshaped, axis=1).sum(axis=1)  # Sum across components
    
    return correlations, feature_importance

def bootstrap_nmf(data, n_components, n_bootstrap=100, random_state=42, n_jobs=None):
    """
    Perform bootstrapped NMF to assess component stability (parallelized).
    
    Parameters:
    - n_jobs: Number of parallel jobs. If None, uses all available CPUs.
    
    Returns:
    - component_stabilities: correlation of components across bootstrap runs
    - feature_importances_bootstrap: feature importance across all runs
    """
    np.random.seed(random_state)
    
    n_samples, n_features = data.shape
    n_timepoints = len(time_index)
    n_features_original = len(feature_names)
    
    # Reference run with full data
    nmf_ref = NMF(n_components=n_components, random_state=42, max_iter=1000)
    W_ref = nmf_ref.fit_transform(data)
    H_ref = nmf_ref.components_
    
    # Prepare arguments for parallel processing
    if n_jobs is None:
        n_jobs = cpu_count()
    
    print(f"Running {n_bootstrap} bootstrap iterations with {n_jobs} parallel workers...")
    
    # Prepare arguments for each iteration
    args_list = [(i, data, n_components, H_ref, n_timepoints, n_features_original) 
                 for i in range(n_bootstrap)]
    
    # Run bootstrap iterations in parallel
    with Pool(n_jobs) as pool:
        results = pool.map(_bootstrap_single_iteration, args_list)
    
    # Unpack results
    component_stabilities = [r[0] for r in results]
    feature_importances_bootstrap = [r[1] for r in results if r[1] is not None]
    
    print(f"Completed {n_bootstrap} bootstrap iterations!")
    
    return {
        'component_stabilities': component_stabilities,
        'feature_importances': np.array(feature_importances_bootstrap) if feature_importances_bootstrap else np.array([]),
        'mean_stability': np.mean(component_stabilities),
        'std_stability': np.std(component_stabilities)
    }
