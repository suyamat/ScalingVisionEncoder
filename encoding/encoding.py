import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from himalaya.scoring import correlation_score
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.ridge import RidgeCV
from himalaya.backend import set_backend

backend = set_backend("torch_cuda", on_error="warn")

def encoder(
    stim: np.ndarray, 
    resp: np.ndarray, 
    return_model: bool
) -> tuple([np.ndarray, object]):
    
    ncv = 8
    alphas = np.logspace(0, 15, 30)
    n_samples, n_features = stim.shape
    
    if n_samples >= n_features:
        print("Solving ridge regression...")
        ridge = RidgeCV(
            alphas=alphas, cv=ncv, solver_params={"score_func": correlation_score}
        )
        pipeline = make_pipeline(StandardScaler(with_mean=True, with_std=False), ridge)

    else:
        print("Solving kernel ridge regression...")

        ridge = KernelRidgeCV(
            alphas=alphas, cv=ncv, solver_params={"score_func": correlation_score}
        )
        pipeline = make_pipeline(StandardScaler(with_mean=True, with_std=False), ridge)

    pipeline.fit(stim, resp)
    scores = ridge.cv_scores_
    scores = backend.to_numpy(scores)
    print(f"Mean cv scores: {np.mean(scores)}")
    
    if return_model==True:
        return scores, ridge
    
    else:
        return scores, None