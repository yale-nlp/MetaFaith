#!/bin/bash

dataset_names=(popqa selfaware simpleqa halueval math umwp sciq arc_challenge mmlu superglue)
model_name=$1
hedge_name=$2
SPECIAL_PROMPT=$3


for dataset_name in "${dataset_names[@]}"; do
    bash ./scripts/score_cmd_proprietary.sh  $dataset_name  $model_name  $hedge_name  $SPECIAL_PROMPT
done
