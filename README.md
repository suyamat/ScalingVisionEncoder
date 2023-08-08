# Applicability of scaling laws to vision encoding models
This repository provides codes for solving The Algonauts Project 2023 Challenge. For more information, see the our solution paper and The Algonauts Project 2023 page.

[[Our paper](https://arxiv.org/abs/2308.00678)]
[[Challenge website](http://algonauts.csail.mit.edu/)]


# Support model

# Installation
1. ```git clone https://github.com/suyamat/ScalingVisionEncoder```
2. ```cd ScalingVisionEncoder```
3. ```conda create -n scaling_vis_enc python==3.8```
4. ```conda activate scaling_vis_enc```
5. ```pip insall -r requirements.txt```
6. ```echo -e 'DATA_DIR=data\nPYTHONPATH=./' > .env ```

# Data preparation
Place challenge's data to data directory like data/resp/subj01/...

# Usage
## Extract vision models' features
For example, to extract EVA02-CLIP-large' features with 4 GPUs (1 nodes x 4 GPUs), you can run
```
python -m encoding.scripts.extract_features \
    --model_name "eva02-clip-large" \
    --subject_name "all" \
    --skip "1" \
    --n_device "4" \
    --batch_size "128"
```

## Search hyper-parameters
For example, to search the optimal combinations of the layer and the kernel size of maxpooling, using all layers, all kernel sizes and 100% of samples, you can run
```
python -m encoding.scripts.search_hparams \
    --model_name "eva02-clip-large" \
    --subject_name "all" \
    --layer_start "1" \
    --layer_step "1" \
    --layer_end "24" \
    --kernel_start "1" \
    --kernel_step "1" \
    --kernel_end "16" \
    --use_ratio "1.0"
```

## Final predictions
For example, to make final predictions using EVA02-CLIP-large, you can run
```
python -m encoding.scripts.predict \
    --model_name "eva02-clip-large" \
    --subject_name "all" \
```

# Installation
1. ``git clone https://github.com/suyamat/ScalingVisionEncoder``
2. ``cd ScalingVisionEncoder``
3. ``conda create -n scaling_vis_enc python==3.8``
4. ``conda activate scaling_vis_enc``
5. ``pip insall -r requirements.txt``
6. ``echo -e 'DATA_DIR=data\nPYTHONPATH=./' > .env ``

# Acknowledgement
Our codes are built upon these repository. We would like to thank the contributors of these great codebases.
> https://github.com/huggingface/pytorch-image-models

> https://github.com/OFA-Sys/ONE-PEACE

> https://github.com/OpenGVLab/InternImage

> https://github.com/gallantlab/himalaya