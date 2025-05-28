import numpy as np
import torch
import ast, re, string, os
from collections import Counter

from openai import OpenAI
from google.generativeai import GenerationConfig
from google.genai import types
from vllm import SamplingParams

from src.prompts.scoring_prompts import LLM_EVAL_PROMPT
from src.prompts import *


############################################
#### Set Up OpenAI Access
############################################

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))     # openai API key

############################################
#### Inference Helpers
############################################

def sanitize_text(text: str, lowercase: bool = False) -> str: # from qamden
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


def convert_to_one_line(text: str):
    # print(text)
    return text.strip().replace("\n", " ").replace("  ", " ")


def get_response_batch(inference_args, prompts, temperature=None, sys_prompt=None, stop_seqs=["Question:"]):

    args, _, _ = inference_args
    if not args.use_vllm:
        raise ValueError("Batch response querying is only valid for VLLM models.")
    
    args, tokenizer, model = inference_args

    # Load system prompt
    try:
        if args.sys_prompt and not sys_prompt and args.sys_prompt!="filler" and args.sys_prompt!="rr":
            if args.sys_prompt.lower() not in SPECIAL_PROMPT_REGISTRY:
                raise ValueError("Invalid system or special prompt option provided")
            sys_prompt = SPECIAL_PROMPT_REGISTRY[args.sys_prompt.lower()]
    except:
        pass

    k = args.num_candidates     # get sampled_outputs

    gen_kwargs = { 
        key: value for key, value in {
            "temperature": temperature if temperature!=None else args.temperature, 
            "top_p": args.top_p
        }.items() if value is not None
    }
    if args.use_vllm:
       gen_kwargs['max_tokens'] = args.max_output_tokens

    # Set sampling parameters
    SAMPLING_PARAMS = SamplingParams(
        n=k,
        stop=stop_seqs,
        **gen_kwargs,
    )

    # VLLM inference
    outputs = []
    
    # Prepare prompts
    if sys_prompt:
        messages = [
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ] for prompt in prompts
        ]
    else: 
        messages=[
            [
                {"role": "user", "content": prompt}
            ] for prompt in prompts
        ]
    prompts_list = model.get_tokenizer().apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    # Get response & process
    response = model.generate(prompts_list, SAMPLING_PARAMS)

    # If two-stage prompting is used:
    if args.sys_prompt=="rr":
        f_prompt = SPECIAL_PROMPT_REGISTRY[args.sys_prompt.lower()]
        followup_messages = []
        for orig_msg, gen in zip(messages, response):
            if sys_prompt:
                followup_msg = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": orig_msg[1]["content"]},
                    {"role": "assistant", "content": gen.outputs[0].text},
                    {"role": "user", "content": f_prompt}
                ]
            else:
                followup_msg = [
                    {"role": "user", "content": orig_msg[0]["content"]}, 
                    {"role": "assistant", "content": gen.outputs[0].text},
                    {"role": "user", "content": f_prompt}
                ]
            followup_messages.append(followup_msg)

        followup_prompts = model.get_tokenizer().apply_chat_template(
            followup_messages,
            add_generation_prompt=True,
            tokenize=False
        )
        response = model.generate(followup_prompts, SAMPLING_PARAMS)

    for sub_response in response:
        outputs.append([x.text for x in sub_response.outputs])

    return outputs


def get_response(inference_args, prompt, num_responses=None, greedy=False, temperature=None, sys_prompt=None, stop_seqs=["Question:"]):
    
    args, tokenizer, model = inference_args

    # Load system prompt
    try:
        if args.sys_prompt and not sys_prompt and args.sys_prompt!="filler" and args.sys_prompt!="rr":
            if args.sys_prompt.lower() not in SPECIAL_PROMPT_REGISTRY:
                raise ValueError("Invalid system or special prompt option provided")
            sys_prompt = SPECIAL_PROMPT_REGISTRY[args.sys_prompt.lower()]
    except: 
        pass

    if num_responses: 
        k = num_responses 
    else:
        k = args.num_candidates

    gen_kwargs = { 
        key: value for key, value in {
            "temperature": temperature if temperature!=None else args.temperature, 
            "top_p": args.top_p
        }.items() if value is not None
    }
    if args.use_vllm:
       gen_kwargs['max_tokens'] = args.max_output_tokens

    if greedy: 
        gen_kwargs['temperature'] = 0.  
        if "gemini" in args.model_name: 
            gen_kwargs['top_k'] = 1

    # Gemini inference
    if "gemini" in args.model_name:
        generation_config = GenerationConfig( 
            max_output_tokens=args.max_output_tokens, 
            candidate_count=1,
            stop_sequences=stop_seqs,
            **gen_kwargs,
        )
        responses = []
        for i in range(k):
            try:
                if sys_prompt:
                    config = types.GenerateContentConfig(
                        system_instruction=sys_prompt,
                        max_output_tokens=args.max_output_tokens, 
                        candidate_count=1,
                        stop_sequences=stop_seqs,
                        **gen_kwargs,
                    )
                    response = model.models.generate_content(
                        model=args.model_name, 
                        config=config,
                        contents=prompt,
                        # generation_config=generation_config,
                    )
                else: 
                    response = model.generate_content(
                        prompt,
                        generation_config=generation_config,
                    )
                response_text = response.text.strip()
                responses.append(response_text)
            except Exception as e: 
                print(e)
                responses.append("Unknown")
        
    # GPT inference
    elif "gpt" in args.model_name:

        if sys_prompt:
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
        else: 
            messages=[
                {"role": "user", "content": prompt}
                ]

        response = client.chat.completions.create(
            model=args.model_name,
            messages=messages,
            n=k,
            max_tokens=args.max_output_tokens,
            stop=stop_seqs,
            **gen_kwargs,
        ).choices
        responses = [x.message.content for x in response]

    # HF or VLLM inference
    else: 

        # VLLM inference
        if args.use_vllm: 
            SAMPLING_PARAMS = SamplingParams(
                n=k,
                stop=stop_seqs,
                **gen_kwargs,
            )
            if sys_prompt:
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ]
            else: 
                messages=[
                    {"role": "user", "content": prompt}
                    ]
            prompt_list = model.get_tokenizer().apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            response = model.generate(prompt_list, SAMPLING_PARAMS)
            responses = [x.text.strip() for x in response[0].outputs]


        # HF inference -- unused
        else: 
            raise Exception("Non-VLLM inference not implemented")

    return responses



############################################
#### Documentation Helpers
############################################

def serialize(obj):
    """
    Serialize args values to enable writing to json file.
    """
    if isinstance(obj, (np.dtype, torch.dtype)):  # Handle data types
        return str(obj)
    elif isinstance(obj, set):  # Convert sets to lists
        return list(obj)
    elif hasattr(obj, "__dict__"):  # Try to serialize objects with a __dict__ attribute
        return vars(obj)
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def prepare_input_args(data_df):
    """
    Apply ast.literal_eval() to the input args, which are assumed to be **TUPLES**
    """
    
    def safe_literal_eval(row):
        try:
            return ast.literal_eval(row)
        except (ValueError, SyntaxError):
            print(row)
            return None  

    # Drop NaNs
    data_df.dropna(inplace=True)
    
    # Check if it's already strings, in which case literal_eval fails
    dtypes = data_df["input_args"].apply(type).value_counts().to_dict()
    if len(dtypes)==1 and str in dtypes.keys():
        return data_df
    
    data_df["input_args"] = data_df["input_args"].apply(safe_literal_eval)
    data_df = data_df.dropna(subset=["input_args"])

    return data_df



############################################
#### Performance Evaluation
############################################

# LLM evaluation
def llm_eval(targets, pred):

    # Format prompt
    prompt = LLM_EVAL_PROMPT.format(targets=targets, pred=pred)

    # Get response
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))     # gemini api key
        model = genai.GenerativeModel("gemini-2.0-flash")
        generation_config = GenerationConfig( 
            max_output_tokens=2,
            candidate_count=1,
        )
        output = model.generate_content(
            prompt,
            generation_config=generation_config,
            ).text.strip()
        response = normalize_answer(output)
        
    except Exception as e: 
        response = ""
        print("llm scoring error:", e)
    
    # 1 if True, 0 o.w.
    score = 1. if "true" in response else 0. 
    print(score)

    return score


# From https://github.com/aviclu/peekacross
def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# Adapted from https://github.com/aviclu/peekacross
def get_f1(target, pred):
    """
    Get F1 score for a **single** target and prediction.
    """
    pred_normalized = normalize_answer(str(pred)).split()
    target_normalized = normalize_answer(str(target)).split()
    common = Counter(pred_normalized) & Counter(target_normalized)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_normalized)
    recall = 1.0 * num_same / len(target_normalized)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# Adapted from https://github.com/aviclu/peekacross
def get_em(target, pred):
    """
    Get EM score for a **single** target and prediction.
    """
    return (normalize_answer(str(target)) == normalize_answer(str(pred))) * 1.0


def get_acc(target, pred):
    """
    Get accuracy score for a **single** target and prediction.
    """
    try:
        return (float(target) == float(pred)) * 1.0
    except: 
        return 0.


def metric_max_over_ground_truths(metric_fn, possible_targets, pred):
    scores = []
    for target in possible_targets:    

        score = metric_fn(target, pred)
        scores.append(score)
    return max(scores)


# Adapted from https://github.com/aviclu/peekacross
def score_qa(targets, preds, numerical=False):

    if numerical:
        acc_sum = 0
        errors = []
        for possible_targets, pred in zip(targets, preds):
            acc_subscore = metric_max_over_ground_truths(
                metric_fn=get_acc,
                possible_targets=possible_targets, 
                pred=pred,
            )
            acc_sum += acc_subscore
            errors.append((-1, -1))
        acc = 100.0 * acc_sum / len(preds)
        return {"acc": acc}, errors

    else:

        f1_sum = em_sum = 0
        errors = []

        for possible_targets, pred in zip(targets, preds):

            f1_subscore = metric_max_over_ground_truths(
                metric_fn=get_f1,
                possible_targets=possible_targets, 
                pred=pred,
            )
            f1_sum += f1_subscore
            
            em_subscore = metric_max_over_ground_truths(
                metric_fn=get_em, 
                possible_targets=possible_targets, 
                pred=pred,
            )
            em_sum += em_subscore
            errors.append((f1_subscore, em_subscore))

        em = 100.0 * em_sum / len(preds)
        f1 = 100.0 * f1_sum / len(preds)

        return {"f1": f1, "em": em}, errors


############################################
#### Calibration Scoring
############################################

def get_cmfg_mfg(f_values, conf_scores, num_bins=10, ct=None):

    # Only use valid values if specified
    if ct:
        f_values = f_values[:ct]
        conf_scores = conf_scores[:ct]
    # mask1 = (np.array(f_values) != 0) & (np.array(conf_scores) != 0)
    # f_values = np.array(f_values)[mask1]
    # conf_scores = np.array(conf_scores)[mask1]
    mask2 = (np.array(f_values) != -2) & (np.array(conf_scores) != -2)
    f_values = np.array(f_values)[mask2]
    conf_scores = np.array(conf_scores)[mask2]
    print(len(f_values), len(conf_scores))

    # Create equally-sized bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Assign responses to bins based on confidence scores
    bin_indices = np.digitize(conf_scores, bin_edges) - 1
    
    # Calculate mean faithfulness score for each bin
    bin_faithfulness_scores = []
    bin_stds = []
    bin_sems = []
    bin_counts = []
    
    for bin_idx in range(num_bins):

        # Get faithfulness scores in current bin
        bin_mask = (bin_indices == bin_idx)
        bin_f_values = [f for f, m in zip(f_values, bin_mask) if m and f!=-1]
        
        if len(bin_f_values) > 0:
            mean_faithfulness = np.mean(bin_f_values)
            std_faithfulness = np.std(bin_f_values, ddof=1)  # Sample standard deviation
            sem_faithfulness = std_faithfulness / np.sqrt(len(bin_f_values))  # Standard error
        else:
            mean_faithfulness, std_faithfulness, sem_faithfulness = 0.0, 0.0, 0.0
            
        bin_faithfulness_scores.append(mean_faithfulness)
        bin_stds.append(std_faithfulness)
        bin_sems.append(sem_faithfulness)
        bin_counts.append(len(bin_f_values))
    
    # Calculate overall cMFG as average across bins
    overall_cmfg = np.mean(bin_faithfulness_scores)
    cmfg_std = np.std(bin_faithfulness_scores, ddof=1)
    cmfg_sem = cmfg_std / np.sqrt(num_bins)

    # Calculate mean faithfulness (MFG)
    x = [f for f in f_values if f!=-1]
    mfg = np.mean(x)
    mfg_std = np.std(x, ddof=1)
    mfg_sem = mfg_std / np.sqrt(len(x))

    # Compute confidence intervals
    cmfg_ci = (overall_cmfg - 1.96 * cmfg_sem, overall_cmfg + 1.96 * cmfg_sem)
    mfg_ci = (mfg - 1.96 * mfg_sem, mfg + 1.96 * mfg_sem)

    metrics = {
        "cmfg_std": cmfg_std,
        "cmfg_sem": cmfg_sem,
        "cmfg_ci": cmfg_ci,
        "mfg_std": mfg_std,
        "mfg_sem": mfg_sem,
        "mfg_ci": mfg_ci,
    }

    return overall_cmfg, mfg, metrics


def compute_ece(confidences, correctness, n_bins=10):
    """
    Compute Expected Calibration Error
    
    Args:
        confidences: List of model confidence scores (0-1)
        correctness: List of binary correctness values (0 or 1)
        n_bins: Number of bins for confidence scores
        
    Returns:
        ECE score (lower is better)
    """
    # confidences = np.array(confidences)
    # correctness = np.array(correctness)
    
    # Create bins and find bin for each prediction
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure proper binning
    
    # Initialize
    ece = 0
    bin_counts = np.zeros(n_bins)
    bin_correct = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    
    # Accumulate statistics
    for i in range(len(confidences)):
        bin_idx = bin_indices[i]
        bin_counts[bin_idx] += 1
        bin_correct[bin_idx] += correctness[i]
        bin_confidences[bin_idx] += confidences[i]
    
    # Compute bin averages and ECE
    for b in range(n_bins):
        if bin_counts[b] > 0:
            bin_accuracy = bin_correct[b] / bin_counts[b]
            bin_confidence = bin_confidences[b] / bin_counts[b]
            bin_weight = bin_counts[b] / len(confidences)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return ece, bin_confidences, bin_correct, bin_counts


def compute_brier_score(confidences, correctness):
    return ((confidences - correctness) ** 2.).mean()