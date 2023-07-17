import os
import numpy as np
import torch
from torchvision import transforms
import glob
from PIL import Image
import torch.nn as nn
import itertools
from tqdm import tqdm

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

from models.model import define_model
import gc

has_compile = hasattr(torch, 'compile')

torch.backends.cudnn.benchmark = True

TF = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model_layer = {
    "eva02-clip-enormous": 64,
    "eva02-clip-large": 24,
    "eva02-clip-base": 12,
    "eva-clip-giant": 40,
    "clip-convnext": 40,
    "ONE-PEACE": 40,
    "InternImage": 52
}

model_img_size = {
    "eva02-clip-enormous": 224,
    "eva02-clip-large": 224,
    "eva02-clip-base": 224,
    "eva-clip-giant": 224,
    "clip-convnext": 256,
    "ONE-PEACE": 256,
    "InternImage": 512
}

model_spatial_res = {
    "eva02-clip-enormous": 16,
    "eva02-clip-large": 16,
    "eva02-clip-base": 14,
    "eva-clip-giant": 16,
    "clip-convnext": None,
    "ONE-PEACE": 16,
    "InternImage": None
}


def inference(
    model_name: str,
    subject_name: str,
    skip: int,
    n_device: int,
    batch_size: int,
    resp_path: str,
    save_path: str
    
) -> None:
    
    device = 'cuda:0'
    device = torch.device(device)
    layer = model_layer[model_name]
    img_size = model_img_size[model_name]
    spatial_res = model_spatial_res[model_name]
    
    if subject_name == "all":
        subject_name = [f"subj{str(s).zfill(2)}" for s in range(1, 9)]
    else:
        subject_name = [subject_name]

    for sub in subject_name:
        print(f"Using {sub}'s response data...")
        
        for split in ["training", "test"]:
            print(f"Split: {split}")
            files = glob.glob(f"{resp_path}/{sub}/{split}_split/{split}_img_{img_size}px/*")
            files = sorted(files)
            all_im = []
            for i, im_path in enumerate(files):
                # if i==256:
                #     break
                im = Image.open(im_path)
                im = TF(im)
                all_im.append(im)

            all_im = torch.stack(all_im, axis=0)
            im_dataset = torch.utils.data.TensorDataset(all_im)
            im_loader = torch.utils.data.DataLoader(im_dataset, batch_size=batch_size, shuffle=False)

            for l in range(0, layer, skip):
                print(f"Extracting features from layer{l+1}")
                model = define_model(model_name=model_name, depth=l)
                model = nn.DataParallel(model, device_ids=[i for i in range(n_device)])
                model = model.to(device)
                model.eval()

                preds = []
                with torch.no_grad():
                    for i, img in enumerate(tqdm(im_loader)):
                        img = torch.tensor(img[0], device=device)
                        pred = model(img) #[:, 1:] #[B, S, C] = [64, 24x24, 1408]
                
                        if model_name == "clip-convnext":
                            m = nn.AdaptiveMaxPool2d((21, 21))
                            pred = m(pred)
                        if model_name == "InternImage":
                            m = nn.AdaptiveMaxPool2d((14, 14))
                            pred = pred.permute(0, 3, 1, 2)
                            pred = m(pred)
                        else:
                            pred = pred[:, 1:, :]
                            pred = pred.reshape(pred.shape[0], spatial_res, spatial_res, -1)
                            pred = pred.permute(0, 3, 1, 2)
                            
                        pred = pred.data.to('cpu').detach().numpy().copy()
                        preds.append(pred)

                preds = np.array(list(itertools.chain.from_iterable(preds)))
                save_dir = f"{save_path}/features/{model_name}/{sub}/{split}"
                os.makedirs(save_dir, exist_ok=True)
                np.save(f'{save_dir}/layer{l+1}.npy', preds)

                del pred, model
                gc.collect()
                torch.cuda.empty_cache()
