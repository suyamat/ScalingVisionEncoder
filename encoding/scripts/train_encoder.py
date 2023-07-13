import numpy as np
from sklearn.preprocessing import StandardScaler
from himalaya.scoring import correlation_score
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import pickle
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.backend import set_backend
from tqdm import tqdm
import gc


#Algonaus2023_scoreLOI_submit 
backend = set_backend("torch_cuda", on_error="warn") 
sc = StandardScaler() #標準化のインスタンスを作成 
model = 'eva02_clip'
test_num = [159, 159, 293, 395, 159, 293, 159, 395] #テストデータのサンプル数

def main():
    for j in range(1, 9):
        num = str(j)
        num = num.zfill(2)
        sub = f"subj{num}"
        print(sub)

        lh_fmri = np.load(f'/mount/nfs5/matsuyama-takuya/dataset/alg2023/{sub}/training_split/training_fmri/lh_training_fmri.npy') #[9841, 19004]
        rh_fmri = np.load(f'/mount/nfs5/matsuyama-takuya/dataset/alg2023/{sub}/training_split/training_fmri/rh_training_fmri.npy')

        for hemisphere in ['lh', 'rh']:
            print(hemisphere)

            if hemisphere == 'lh': 
                whole_pred = np.zeros((test_num[j-1], lh_fmri.shape[1])) #提出用の入れ物を作成
                Y_train = lh_fmri
            if hemisphere == 'rh':
                whole_pred = np.zeros((test_num[j-1], rh_fmri.shape[1]))
                Y_train = rh_fmri
            
            score_roi = np.load(f'/mount/nfs5/matsuyama-takuya/dataset/alg2023/alg_params/{model}/{sub}/wb_params_score.npy',
                allow_pickle=True)
            score_roi = score_roi.item()
            
            score = np.zeros(Y_train.shape[1])
            best_params = np.ones((Y_train.shape[1], 2))
            best_params[:, 0] = best_params[:, 0]*4
            for l in range(1, 64, 2):
                for k in range(1, 16, 2):
                    temp_score = score_roi[hemisphere][f'layer{l}'][f'kernel{k}']
                    best_params[temp_score > score, :] = l, k
                    score[temp_score > score] = temp_score[temp_score > score]
            best_params = best_params.astype(np.int)

            sum = 0
            for l in range(1, 64, 2):
                layer_index = np.where(best_params[:, 0]==l)[0] #最も精度の高かった層
                for k in range(1, 16, 1):
                    kernel_index = np.where(best_params[:, 1]==k)[0] #最も精度の高かったカーネル
                    lk_index = np.intersect1d(layer_index, kernel_index)
                    sum += lk_index.shape[0]
                    # print(lk_index.shape)
                    if lk_index.shape[0] == 0:
                        continue

                    X_best_layer = l
                    X_best_kernel = k
                    X_train = np.load(f'/mount/nfs5/matsuyama-takuya/dataset/alg2023/features/{model}/{sub}/train/layer{X_best_layer}.npy')
                    X_test = np.load(f'/mount/nfs5/matsuyama-takuya/dataset/alg2023/features/{model}/{sub}/test/layer{X_best_layer}.npy')
                    X_train = torch.from_numpy(X_train)
                    X_test = torch.from_numpy(X_test)
                    m = nn.AdaptiveMaxPool2d((X_best_kernel, X_best_kernel))
                    X_train = m(X_train)
                    X_test = m(X_test)
                    X_train = X_train.to('cpu').detach().numpy().astype('float32')
                    X_test = X_test.to('cpu').detach().numpy().astype('float32')
                    X_train = X_train.reshape(X_train.shape[0], -1)
                    X_test = X_test.reshape(X_test.shape[0], -1)
                    X_train = sc.fit_transform(X_train)
                    X_test = sc.fit_transform(X_test)

                    X_train = backend.asarray(X_train)
                    X_test = backend.asarray(X_test)
                    Y_roi_train = backend.asarray(Y_train[:, lk_index])

                    reg = KernelRidgeCV(kernel="linear", alphas = np.logspace(0, 15, 50), cv=8, Y_in_cpu=True) #e4 ~ e12 [30, 30s, 0.573] [50, 30s, 0.573] [100, 37s, 0.574]
                    reg.fit(X_train, Y_roi_train)
                    cv_score = reg.cv_scores_.to('cpu').detach().numpy()
                    # print(-np.median(cv_score))
                    pred_test = reg.predict(X_test) #予測 0.618, 
                    pred_test = pred_test.to('cpu').detach().numpy()

                    whole_pred[:, lk_index] = pred_test

                    del reg, pred_test, X_train, X_test
                    gc.collect()
                    torch.cuda.empty_cache()

            whole_pred = whole_pred.astype(np.float32)

            path = f"/mount/nfs5/matsuyama-takuya/dataset/alg2023/test_pred_{model}_scoreroi/{sub}"
            if not os.path.exists(f'{path}'):
                    os.makedirs(f'{path}')
            if hemisphere == 'lh':
                np.save(f'{path}/lh_pred_test.npy', whole_pred)
            elif hemisphere == 'rh':
                np.save(f'{path}/rh_pred_test.npy', whole_pred)

if __name__ == "__main__":
    main()