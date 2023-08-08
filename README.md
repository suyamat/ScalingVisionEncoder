# Applicability of scaling laws to vision encoding models
This repository provides codes for solving The Algonauts Project 2023 Challenge. For more information, see our solution paper and the challenge's page.

[[Our paper](https://arxiv.org/abs/2308.00678)]
[[Challenge website](http://algonauts.csail.mit.edu/)]

# Support models

<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh" align="center" rowspan="2">model</th>
    <th class="tg-0lax" align="center" rowspan="2">Size name</th>
    <th class="tg-baqh" align="center" rowspan="2">Num. of paremeters</th>
    <th class="tg-0lax" align="center" rowspan="2">Link</th>
  </tr>
  <tr>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh" align="center">EVA01-CLIP</td>
    <td class="tg-0lax" align="center">huge
    <td class="tg-baqh" align="center">1.0B</td>
    <td class="tg-0lax" align="center" rowspan="4"><a href="https://arxiv.org/abs/2303.15389">Paper</a></td>
  </tr>
  <tr>
  <td class="tg-baqh" align="center" rowspan="3">EVA02-CLIP</td>
  <td class="tg-0lax" align="center">base
  <td class="tg-baqh" align="center">0.086B </td>
  </tr>
  <tr>
  <td class="tg-0lax" align="center">large
  <td class="tg-baqh" align="center">0.3B </td>
  </tr>
  <tr>
  <td class="tg-0lax" align="center">enormous
  <td class="tg-baqh" align="center">4.4B </td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">ConvNext</td>
    <td class="tg-0lax" align="center">xxlarge
    <td class="tg-baqh" align="center">0.85B</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2201.03545">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">ONE-PEACE</td>
    <td class="tg-0lax" align="center">N / A
    <td class="tg-baqh" align="center">1.5B</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2305.11172">Paper</a></td>
  </tr>
  <tr>
    <td class="tg-baqh" align="center">InternImage</td>
    <td class="tg-0lax" align="center">giant
    <td class="tg-baqh" align="center">3.0B</td>
    <td class="tg-0lax" align="center"><a href="https://arxiv.org/abs/2211.05778">Paper</a></td>
  </tr>
</tbody>
</table>


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