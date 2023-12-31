import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from himalaya.backend import set_backend
import gc

from encoding.encoding import encoder

backend = set_backend("torch_cuda", on_error="warn")

def searcher(
    model_name: str,
    subject_name: str,
    layer_start: int,
    layer_step: int,
    layer_end: int,
    kernel_start: int,
    kernel_step: int,
    kernel_end: int,
    use_ratio: float,
) -> None:

    model_layer = {
    "eva02-clip": 64,
    "eva-g": 40,
    "clip-convnext": 40,
    "ONE-PEACE": 40,
    }

    print(f"Searching {model_name}'s hyper parameters...")
    print(f"Layer setting: Start layer={layer_start}, End layer = {layer_end}, Step layer = {layer_step}")
    print(f"Pooling setting: Start size={kernel_start}, End size = {kernel_end}, Step size = {kernel_step}")

    if subject_name == "all":
        subject_name = [f"subj{str(s).zfill(2)}" for s in range(1, 9)]
    else:
        subject_name = [subject_name]
        
    for sub in subject_name:
        best_scores = {}
        for hemisphere in ['lh', 'rh']:
            print(f"Using {sub}'s {hemisphere} response data...")
            best_scores[hemisphere] = {}
            resp = np.load(f"./data/resp/{sub}/training_split/training_fmri/{hemisphere}_training_fmri.npy")
            n_samples = resp.shape[0]
            use_samples = int(n_samples * use_ratio)
            resp = resp[:use_samples]
            resp = resp.astype("float32")

            print(f"Number of using samples: {use_samples} of {n_samples}")

            for l in range(layer_start, layer_end+1, layer_step):
                best_scores[hemisphere][f'layer{l}'] = {}
                stim = np.load(f"./data/features/{model_name}/{sub}/training/layer{l}.npy")
                stim = stim[:use_samples]
                stim = stim.astype('float32')
                stim = torch.from_numpy(stim)
                for k in range(kernel_start, kernel_end+1, kernel_step):
                    print(f"layer{l}, kernel{k}")
                    pooling = nn.AdaptiveMaxPool2d((k, k))
                    pooled_stim = pooling(stim)
                    pooled_stim = pooled_stim.reshape(pooled_stim.shape[0], -1)
                    pooled_stim = pooled_stim.to('cpu').detach().numpy()
                    res, _ = encoder(stim=pooled_stim, resp=resp, return_model=False)
                    best_scores[hemisphere][f'layer{l}'][f'kernel{k}'] = res
                    
                    del pooled_stim, res
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            
        save_dir = f"./data/results/hparams/{model_name}/{sub}"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/hparams_score.npy", best_scores)