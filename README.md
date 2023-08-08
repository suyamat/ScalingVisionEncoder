# Applicability of scaling laws to vision encoding models
This repository provides code for participating in The Algonauts Project 2023 Challenge. For more details, please refer to our solution paper and the challenge's official page.

[[Our paper](https://arxiv.org/abs/2308.00678)]
[[The challenge's official page](http://algonauts.csail.mit.edu/)]

# Supported Models

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
```
git clone https://github.com/suyamat/ScalingVisionEncoder
cd ScalingVisionEncoder
conda create -n scaling_vis_enc python==3.8
conda activate scaling_vis_enc
pip insall -r requirements.txt
echo -e 'DATA_DIR=data\nPYTHONPATH=./' > .env 
```

# Data Preparation
Place the challenge's data in the data directory, following this structure: data/resp/subj01/...

# Usage
## Extract Vision Models' Features
For instance, to extract features from "EVA02-CLIP-large" with 4 GPUs (1 node x 4 GPUs), you can run
```
python -m encoding.scripts.extract_features \
    --model_name "eva02-clip-large" \
    --subject_name "all" \
    --skip "1" \
    --n_device "4" \
    --batch_size "128"
```

## Search for Hyper-Parameters
For instance, to search for the optimal combination of layer and kernel size for max pooling, utilizing all layers, all kernel sizes, and 100% of the samples, you can run
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

## Final Predictions
For example, to make final predictions using EVA02-CLIP-large, you can run
```
python -m encoding.scripts.predict \
    --model_name "eva02-clip-large" \
    --subject_name "all" \
```

# Acknowledgement
Our code is built upon the following repositories. We would like to extend our gratitude to the contributors of these excellent codebases.

> https://github.com/huggingface/pytorch-image-models

> https://github.com/OFA-Sys/ONE-PEACE

> https://github.com/OpenGVLab/InternImage

> https://github.com/gallantlab/himalaya