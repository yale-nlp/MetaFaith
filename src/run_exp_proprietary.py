import argparse
import pandas as pd
import json
import re
import ast 
import os
import time

from src.prompts.input_prompts import *
from src.prompts.task_prompts import *
from src.utilities.utils import *
from src.tasks._task import Task

from src.tasks import TASK_REGISTRY, TASK_DEFAULTS, MODEL_DEFAULTS
from src.prompts import *


def parse_args():
    """
    Define and parse script arguments.
    """

    def add_arg(parser, arg, defaults, **kwargs):
        if defaults is None:
            default = None
        else:
            default = defaults.get(arg)
        parser.add_argument(f"{arg}", default=default, **kwargs)

    parser = argparse.ArgumentParser()

    ### Set model args
    add_arg(
        parser, 
        "--model_name", 
        MODEL_DEFAULTS, 
        type=str, 
        help="Name of model (e.g. in HF hub)"
    )
    parser.add_argument(
        "--use_vllm", 
        action='store_true',
        default=False,
        help="Whether to use VLLM for inference for HF model"
    )


    ### Set inference args
    parser.add_argument(
        "--start_idx", 
        type=int, 
        default=-1,
        help="Index of sample to start on"
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=100,
        help="Max tokens model can generate at inference time",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for model inference",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top_p for model inference",
    )


    ### Set faithfulness/experiment args
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=20,
        help="Number of sampled responses to use when scoring uncertainty/confidence of the model",
    )
    parser.add_argument(
        "--input_prompt",
        default="qa",
        required=True,
        help="Lowercase string name of the input prompt to use; full list is in src/prompts/__init__.py",
    )
    parser.add_argument(
        "--hedge_prompt",
        default="blank",
        required=True,
        help="Lowercase string name of the uncertainty elicitation (hedge) prompt to use; full list is in src/prompts/__init__.py",
    )
    parser.add_argument(
        "--task_prompt",
        default="qa_short",
        required=True,
        help="Lowercase string name of the task prompt to use; full list is in src/prompts/__init__.py",
    )
    parser.add_argument(
        "--sys_prompt",
        default=None,
        help="System or special prompt to use",
    )
    
    ### Set dataset args
    parser.add_argument(
        "--dataset_name",
        type=str,
        # nargs="+",
        required=True,
        choices=list(TASK_REGISTRY.keys()),
        help="Name of the dataset/task to use, full list is in src/tasks/__init__.py",
    )
    parser.add_argument(
        "--num_samples",
        default=None,
        type=int,
        help="Max number of instances to run inference on",
    )
    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Seed for sampling when `num_samples` is smaller than dataset size",
    )

        
    ### Set saving/output args
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Directory for output files"
    )
    parser.add_argument(
        "--score_only", 
        action='store_true',
        default=False,
        help="Score existing saved results."
    )

    args = parser.parse_args()
    return args


def load_model(args):
    """
    Load model as specified in args.
    """

    # Prepare Gemini model
    if "gemini" in args.model_name:
        tokenizer = None
        if args.sys_prompt and args.sys_prompt!="rr" and args.sys_prompt!="filler":
            from google import genai
            model = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))    # gemini api key
        else: 
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))     # gemini api key
            model = genai.GenerativeModel(args.model_name)

    # Prepare GPT model
    elif "gpt" in args.model_name:
        tokenizer = None
        model = None

    return tokenizer, model


def get_identifier(args):
    """
    Get identifying string for the present experiment to use in saving.
    """

    num_samples, dataset_name, input_prompt, task_prompt, hedge_prompt = args.num_samples, args.dataset_name, args.input_prompt, args.task_prompt.lower(), args.hedge_prompt.lower()

    task_identifier = f"{dataset_name}__{input_prompt}__{task_prompt}__{hedge_prompt}__{num_samples}_samps"

    if args.sys_prompt:
        task_identifier += f"__prompt_{args.sys_prompt}"

    return task_identifier


def get_model_identifier(args):

    model_identifier = args.model_name.replace("-", "_").replace("/", "_")

    if args.temperature:
        model_identifier += f"__temp_{args.temperature}"
    if args.top_p:
        model_identifier += f"__temp_{args.top_p}"

    return model_identifier


def apply_prompts(args, data_df):
    """
    Format each input into the task and hedge prompts and return updated data_df.
    """

    def extract_placeholders(template):
        """
        Extract placeholders from a template string.
        """
        return re.findall(r"\{(\w+)\}", template)
    
    def promptify(row):
        """
        Format input args into task prompt and then hedge prompt.
        """
        placeholders = extract_placeholders(input_prompt)

        num_input_args = len(row["input_args"]) 
        num_placeholders = len(placeholders)
        if num_input_args != num_placeholders:
            raise ValueError(f"Mismatch between number of template arguments {num_placeholders} and number of provided arguments {num_input_args}")
        format_args = dict(zip(placeholders, row["input_args"]))
        task_input = input_prompt.format(**format_args)
        hedged_task_input = task_prompt.format(
            task_input=task_input,
            hedge_prompt=hedge_prompt,
        )
        return (task_input, hedged_task_input)

    # Access task and hedge prompts
    if args.input_prompt not in INPUT_PROMPT_REGISTRY:
        raise ValueError("Invalid task prompt provided")
    if args.task_prompt.lower() not in TASK_PROMPT_REGISTRY:
        raise ValueError("Invalid hedge prompt provided")
    if args.hedge_prompt.lower() not in HEDGE_PROMPT_REGISTRY:
        raise ValueError("Invalid hedge direction provided")
    input_prompt = INPUT_PROMPT_REGISTRY[args.input_prompt]
    task_prompt = TASK_PROMPT_REGISTRY[args.task_prompt.lower()]
    hedge_prompt = HEDGE_PROMPT_REGISTRY[args.hedge_prompt.lower()]
    if args.sys_prompt=="filler":
        special_prompt = SPECIAL_PROMPT_REGISTRY["filler"]
        hedge_prompt += f" {special_prompt}"

    # Apply task and hedge prompt formatting
    data_df["inputs"] = data_df.apply(promptify, axis=1)

    return data_df


def run(args):
    """
    Run full inference experiment and save predictions results.
    """

    start_time = time.time()


    # Prepare output directories
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    candidate_str = f"{args.num_candidates}_cands"
    task_identifier = get_identifier(args)
    model_identifier = get_model_identifier(args)
    task_dir = os.path.join(output_dir, task_identifier)
    save_dir = os.path.join(output_dir, task_identifier, candidate_str, model_identifier)
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Do nothing if already completed
    done = False
    metrics_file = os.path.join(save_dir, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f) 
        if "predxn_runtime_seconds" or "runtime_seconds" in metrics.keys():
            done = True 

    if done:     
        print(f"Already completed predictions; skipping {task_identifier}...")

    else:       

        print("\n\n"+"#"*50)
        print(f"Starting predictions experiment {task_identifier} for model {model_identifier}...")

        # Save arguments
        if not args.score_only:
            args_file = os.path.join(save_dir, "args.json")
            with open(args_file, "w") as f:
                json.dump(vars(args), f, indent=4, default=serialize)       # save args


        # Load task
        if args.dataset_name not in TASK_REGISTRY:               
            raise ValueError("Invalid dataset/task provided")

        task: Task = TASK_REGISTRY[args.dataset_name](args=args)
        

        # Get data samples for inference
        inputs_file = os.path.join(task_dir, "formatted_inputs.csv")   # if existent
        if os.path.exists(inputs_file):
            data_df = pd.read_csv(inputs_file, index_col=0)
            data_df["input_args"] = data_df["input_args"].apply(ast.literal_eval)
            data_df["inputs"] = data_df["inputs"].apply(ast.literal_eval)
            if task.task_type=="qa":
                try:
                    data_df["targets"] = data_df["targets"].apply(ast.literal_eval)
                except: 
                    pass
            task.set_data_df(data_df)

        else:                                                        # otherwise
            data_df = task.get_data_df()
            if type(data_df.input_args.iloc[0])!=tuple:
                data_df["input_args"] = data_df["input_args"].apply(ast.literal_eval)
            # data_df = prepare_input_args(data_df)                   # literal eval
            
            # Preprocess with task and hedging prompts
            data_df = apply_prompts(args, data_df)

            # Save formatted inputs, if not already saved
            data_df.to_csv(inputs_file)


        # Load model and tokenizer, if applicable
        tokenizer, model = load_model(args)
        inference_args = args, tokenizer, model 
    

        # Create results files
        results_file = os.path.join(save_dir, "results.csv")
        sampled_outputs_file = os.path.join(save_dir, "outputs_sampled.json")

        if os.path.exists(results_file):
            if args.start_idx != -1:        # start from some specified index
                results_df = pd.read_csv(results_file, index_col=0)
                try:
                    results_df["targets"] = results_df["targets"].apply(ast.literal_eval)
                except: 
                    pass
                results_df["outputs"] = results_df["outputs"].apply(pd.to_numeric, errors='ignore') 
                with open(sampled_outputs_file) as f:
                    sampled_outputs_dict = json.load(f)
                start_idx = args.start_idx
            else:                           # start from unfinished index
                results_df = pd.read_csv(results_file, index_col=0)
                try:
                    results_df["targets"] = results_df["targets"].apply(ast.literal_eval)
                except: 
                    pass
                results_df["outputs"] = results_df["outputs"].apply(pd.to_numeric, errors='ignore') 
                with open(sampled_outputs_file) as f:
                    sampled_outputs_dict = json.load(f) 
                try:
                    start_idx = results_df[results_df['outputs'] == -2].index[0]
                except IndexError:
                    print("Finished evaluating all samples; skipping all...")
                    start_idx = results_df.shape[0]
                except Exception as e: 
                    print(f"An unexpected error occurred: {e}")
                    raise
            
        else: 
            results_df = pd.DataFrame() 
            results_df['outputs'] = np.ones(len(data_df)) * -2.
            results_df['faithfulness'] = np.ones(len(data_df)) * -2.
            results_df['confidence'] = np.ones(len(data_df)) * -2.
            results_df['decisiveness'] = np.ones(len(data_df)) * -2.
            sampled_outputs_dict = {}
            start_idx = 0


        # Run model on samples in dataset
        for idx, example in enumerate(data_df['inputs']): 

            if idx < start_idx:
                print(f"Skipping index {idx}")
                continue

            print(f"On index {idx}")


            # Obtain input
            unhedged_input, hedged_input = example

            # Get output
            try:                     # gpt case
                outputs = get_response(
                    inference_args=inference_args, 
                    prompt=hedged_input, 
                    num_responses=1,
                ) 
                output = outputs[0]
            except:                 # gemini case
                output = get_response(
                    inference_args=inference_args,
                    prompt=hedged_input,
                    num_responses=1,
                )[0]

            # Get sampled_outputs
            try: 
                sampled_outputs = get_response(
                    inference_args,
                    hedged_input,
                )
            except: 
                print("Proprietary model output generation error")
                sampled_outputs = [""]*args.num_candidates
            
            
            # Record
            results_df.loc[idx, 'outputs'] = output
            sampled_outputs_dict[f"sample_{str(idx)}"] = sampled_outputs

            results_df['targets'] = data_df['targets'].tolist()
            results_df = results_df[['targets', 'outputs'] + [col for col in results_df.columns if col!='targets' and col!='outputs']]

            # Save results intermediately
            results_df.to_csv(results_file)
            with open(sampled_outputs_file, 'w') as file:
                json.dump(sampled_outputs_dict, file, indent=4)
        
        results_df['targets'] = data_df['targets'].tolist()
        results_df = results_df[['targets', 'outputs'] + [col for col in results_df.columns if col!='targets' and col!='outputs']]


        # Get runtime
        metrics = {}
        processing_time = time.time() - start_time  
        print(f"\nCompleted evaluation in {processing_time:.2f} seconds")
        metrics['predxn_runtime_seconds'] = processing_time

        # Save results
        print(f"\nSaving predictions and runtime to {save_dir}...")
        results_df.to_csv(results_file)
        with open(sampled_outputs_file, 'w') as file:
            json.dump(sampled_outputs_dict, file, indent=4)
        with open(metrics_file, 'w') as file:
            json.dump(metrics, file, indent=4)


if __name__ == "__main__":
    args = parse_args()
    run(args)

