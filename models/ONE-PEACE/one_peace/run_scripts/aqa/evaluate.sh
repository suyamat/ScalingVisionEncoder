#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=1,2,3
export GPUS_PER_NODE=3

config_dir=../../run_scripts
path=../../checkpoints/finetune_aqa.pt
task_name=aqa
model_name=one_peace_classify
selected_cols=uniq_id,audio,text,duration
results_path=../../results/aqa

data=../../dataset/avqa/avqa_val.tsv
gen_subset='val'
torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    --config-dir=${config_dir} \
    --config-name=evaluate \
    common_eval.path=${path} \
    common_eval.results_path=${results_path} \
    task._name=${task_name} \
    model._name=${model_name} \
    dataset.gen_subset=${gen_subset} \
    dataset.batch_size=4 \
    common_eval.model_overrides="{'task': {'_name': '${task_name}', 'data': '${data}', 'selected_cols': '${selected_cols}', 'bpe_dir': '../../utils/BPE'}}"
