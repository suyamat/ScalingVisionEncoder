import numpy as np
import os
import torch
import torch.nn as nn
from himalaya.backend import set_backend
from tqdm import tqdm
import gc

from encoding.encoding import encoder

backend = set_backend("torch_cuda", on_error="warn") 
test_num = [159, 159, 293, 395, 159, 293, 159, 395] #テストデータのサンプル数

def predictor(
    model_name: str,
    subject_name: str,
) -> None:
    
    print(f"Making {model_name}'s final predictions...")

    if subject_name == "all":
        subject_name = [f"subj{str(s).zfill(2)}" for s in range(1, 9)]
    else:
        subject_name = [subject_name]
        
    for sub_idx, sub in enumerate(subject_name):
        print(sub)
        score_roi = np.load(f'./data/params/{model_name}/{sub}/wb_params_score.npy',
        allow_pickle=True)
        score_roi = score_roi.item()

        for hemisphere in ['lh', 'rh']:
            print(f"Using {sub}'s {hemisphere} response data...")
            resp_train = np.load(f"./data/resp/{sub}/training_split/training_fmri/{hemisphere}_training_fmri.npy")
            whole_pred = np.zeros((test_num[sub_idx], resp_train.shape[1]))
            
            score = np.zeros(resp_train.shape[1])
            best_params = np.ones((resp_train.shape[1], 2))
            best_params[:, 0] = best_params[:, 0]*4
            for l in range(1, 64, 2):
                for k in range(1, 16, 2):
                    temp_score = score_roi[hemisphere][f'layer{l}'][f'kernel{k}']
                    best_params[temp_score > score, :] = l, k
                    score[temp_score > score] = temp_score[temp_score > score]
            best_params = best_params.astype(np.int)

            sum = 0
            for l in tqdm(range(1, 64, 2)):
                stim_train = np.load(f'./data/features/{model_name}/{sub}/train/layer{l}.npy')
                stim_test = np.load(f'./data/features/{model_name}/{sub}/test/layer{l}.npy')
                stim_train = torch.from_numpy(stim_train)
                stim_test = torch.from_numpy(stim_test)
                layer_index = np.where(best_params[:, 0]==l)[0] #最も精度の高かった層
                for k in range(1, 16, 1):
                    kernel_index = np.where(best_params[:, 1]==k)[0] #最も精度の高かったカーネル
                    lk_index = np.intersect1d(layer_index, kernel_index)
                    sum += lk_index.shape[0]
                    # print(lk_index.shape)
                    if lk_index.shape[0] == 0:
                        continue
                    best_kernel = k

                    m = nn.AdaptiveMaxPool2d((best_kernel, best_kernel))
                    pooled_stim_train = m(stim_train)
                    pooled_stim_test = m(stim_test)
                    pooled_stim_train = pooled_stim_train.to('cpu').detach().numpy().astype('float32')
                    pooled_stim_test = pooled_stim_test.to('cpu').detach().numpy().astype('float32')
                    pooled_stim_train = pooled_stim_train.reshape(pooled_stim_train.shape[0], -1)
                    pooled_stim_test = pooled_stim_test.reshape(pooled_stim_test.shape[0], -1)

                    resp_roi_train = resp_train[:, lk_index]

                    res, reg = encoder(stim=pooled_stim_train, resp=resp_roi_train, return_model=False)
                    
                    pred_test = reg.predict(pooled_stim_test) #予測 0.618,
                    pred_test = pred_test.to('cpu').detach().numpy()

                    whole_pred[:, lk_index] = pred_test

                    del pooled_stim_train, pooled_stim_test, res, reg
                    gc.collect()
                    torch.cuda.empty_cache()

            whole_pred = whole_pred.astype(np.float32)

            path = f"./data/submission/test_pred_{model_name}/{sub}"
            os.makedirs(f'{path}', exist_ok=True)
            if hemisphere == 'lh':
                np.save(f'{path}/lh_pred_test.npy', whole_pred)
            elif hemisphere == 'rh':
                np.save(f'{path}/rh_pred_test.npy', whole_pred)