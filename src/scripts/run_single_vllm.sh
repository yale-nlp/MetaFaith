#!/bin/bash

# Command-line arguments
MODEL_NAME=$1
HEDGE_PROMPT=$2
GMU=${3:-0.9} # VLLM argument gpu_mem_utilization: default to 0.9 if not passed 
TPS=${4:-1}  # VLLM argument tensor_parallel_size: default to 1 if not passed
SPECIAL_PROMPT=$5 # Identifier for MetaFaith calibration prompt or advanced prompt strategy; see names in src/prompts/__init__.py

# Pass the special prompt name if specified, else leave blank
SPECIAL_PROMPT_ARG=""
if [ -n "$SPECIAL_PROMPT" ]; then
  SPECIAL_PROMPT_ARG="--sys_prompt=$SPECIAL_PROMPT"
fi

### QA Datasets

python run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="popqa"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="qa"  --task_prompt="qa_short"  --output_dir="../results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU --max_output_tokens=250  $SPECIAL_PROMPT_ARG

python run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="selfaware"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="qa"  --task_prompt="qa_short_answerability"  --output_dir="../results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU --max_output_tokens=250  $SPECIAL_PROMPT_ARG

python run_exp_vllm.py --model_name="$MODEL_NAME"  --use_vllm  --dataset_name="simpleqa"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="qa"  --task_prompt="qa_short"  --output_dir="../results" --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU --max_output_tokens=250  $SPECIAL_PROMPT_ARG

### Hallucination Detection

python run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="halueval"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="hd"  --task_prompt="hd" --output_dir="../results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU  --max_output_tokens=250  $SPECIAL_PROMPT_ARG

### Mathematics & STEM Challenges

python run_exp_vllm.py --model_name="$MODEL_NAME"  --use_vllm  --dataset_name="math"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="qa"  --task_prompt="math" --output_dir="../results" --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU --max_output_tokens=250  $SPECIAL_PROMPT_ARG

python run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="umwp"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="qa"  --task_prompt="umwp" --output_dir="../results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU  --max_output_tokens=250  $SPECIAL_PROMPT_ARG

python run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="sciq"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="mcq"  --task_prompt="mcq_unique" --output_dir="../results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU  --max_output_tokens=250  $SPECIAL_PROMPT_ARG

python run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="arc_challenge"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="mcq"  --task_prompt="mcq_unique_letters" --output_dir="../results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU  --max_output_tokens=250  $SPECIAL_PROMPT_ARG

### General Task Suites

python run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="mmlu"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="mcq"  --task_prompt="mcq_unique" --output_dir="../results"  --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU  --max_output_tokens=250  $SPECIAL_PROMPT_ARG

python run_exp_vllm.py --model_name="$MODEL_NAME" --use_vllm  --dataset_name="superglue"  --num_samples=1000  --num_candidates=20  --hedge_prompt="$HEDGE_PROMPT"  --input_prompt="qa"  --task_prompt="superglue" --output_dir="../results" --tensor_parallel_size=$TPS --gpu_mem_utilization=$GMU  --max_output_tokens=250  $SPECIAL_PROMPT_ARG

