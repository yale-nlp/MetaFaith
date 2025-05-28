#!/bin/bash

dataset_names=(popqa selfaware simpleqa halueval math umwp sciq arc_challenge mmlu superglue)
model_name=$1
hedge_name=$2
SPECIAL_PROMPT=$3

for dataset_name in "${dataset_names[@]}"; do
    echo $dataset_name
    bash ./scripts/score_cmd_vllm.sh  $dataset_name  $model_name  $hedge_name  $SPECIAL_PROMPT
done
