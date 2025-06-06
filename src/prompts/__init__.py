from typing import Dict

from . import (
    hedge_prompts,
    input_prompts,
    metafaith_calibration_prompts,
    task_prompts,
    advanced_prompts,
)

HEDGE_PROMPT_REGISTRY: Dict[str, str] = {
    "blank": hedge_prompts.BLANK,
    "basic": hedge_prompts.BASIC,
    "genuine": hedge_prompts.GENUINE,
    "human": hedge_prompts.HUMAN,
    "perception": hedge_prompts.PERCEPTION,
}

INPUT_PROMPT_REGISTRY: Dict[str, str] = {
    "qa": input_prompts.QA_INPUT,
    "mcq": input_prompts.MCQ_INPUT,
    "hd": input_prompts.HD_INPUT,
}

TASK_PROMPT_REGISTRY: Dict[str, str] = {
    "qa_short": task_prompts.QA_SHORT_PROMPT,
    "qa_short_answerability": task_prompts.QA_SHORT_PROMPT_ANSWERABILITY,
    "mcq_unique_letters": task_prompts.MCQ_UNIQUE_LETTERS_PROMPT,
    "mcq_unique": task_prompts.MCQ_UNIQUE_NUMBERS_PROMPT,
    "hd_bare": task_prompts.HD_PROMPT,
    "math": task_prompts.MATH_PROMPT,
    "umwp": task_prompts.UMWP_PROMPT,
    "superglue": task_prompts.SUPERGLUE_PROMPT,
}

SPECIAL_PROMPT_REGISTRY: Dict[str, str] = {
    "detailed": advanced_prompts.detailed,
    "stepbystep": advanced_prompts.stepbystep,
    "persona": advanced_prompts.persona,
    "personality": advanced_prompts.personality,
    "reward": advanced_prompts.reward,
    "metaphor": advanced_prompts.metaphor,
    "intent": advanced_prompts.intent,
    "sentiment": advanced_prompts.sentiment,
    "rr": advanced_prompts.rr,
    "filler": advanced_prompts.filler,

    "ms_1": metafaith_calibration_prompts.ms_1,
    "ms_2": metafaith_calibration_prompts.ms_2,
    "ms_3": metafaith_calibration_prompts.ms_3,
    "ms_4": metafaith_calibration_prompts.ms_4,
    "ms_5": metafaith_calibration_prompts.ms_5,
    "ms_6": metafaith_calibration_prompts.ms_6,
    "ms_7": metafaith_calibration_prompts.ms_7,
    "ms_8": metafaith_calibration_prompts.ms_8,
    "ms_9": metafaith_calibration_prompts.ms_9,
    "ms_10": metafaith_calibration_prompts.ms_10,
    "mh_1": metafaith_calibration_prompts.mh_1,
    "mh_2": metafaith_calibration_prompts.mh_2,
    "mh_3": metafaith_calibration_prompts.mh_3,
    "mh_4": metafaith_calibration_prompts.mh_4,
    "mh_5": metafaith_calibration_prompts.mh_5,
    "mh_6": metafaith_calibration_prompts.mh_6,
    "mh_7": metafaith_calibration_prompts.mh_7,
    "mh_8": metafaith_calibration_prompts.mh_8,
    "mh_9": metafaith_calibration_prompts.mh_9,
    "mh_10": metafaith_calibration_prompts.mh_10,
    "mr_1": metafaith_calibration_prompts.mr_1,
    "mr_2": metafaith_calibration_prompts.mr_2,
    "mr_3": metafaith_calibration_prompts.mr_3,
    "mr_4": metafaith_calibration_prompts.mr_4,
    "mr_5": metafaith_calibration_prompts.mr_5,
    "mr_6": metafaith_calibration_prompts.mr_6,
    "mr_7": metafaith_calibration_prompts.mr_7,
    "mr_8": metafaith_calibration_prompts.mr_8,
    "mr_9": metafaith_calibration_prompts.mr_9,
    "mr_10": metafaith_calibration_prompts.mr_10,
    "msc_1": metafaith_calibration_prompts.msc_1,
    "msc_2": metafaith_calibration_prompts.msc_2,
    "msc_3": metafaith_calibration_prompts.msc_3,
    "msc_4": metafaith_calibration_prompts.msc_4,
    "msc_5": metafaith_calibration_prompts.msc_5,
    "msc_6": metafaith_calibration_prompts.msc_6,
    "msc_7": metafaith_calibration_prompts.msc_7,
    "msc_8": metafaith_calibration_prompts.msc_8,
    "msc_9": metafaith_calibration_prompts.msc_9,
    "msc_10": metafaith_calibration_prompts.msc_10,
    "mhc_1": metafaith_calibration_prompts.mhc_1,
    "mhc_2": metafaith_calibration_prompts.mhc_2,
    "mhc_3": metafaith_calibration_prompts.mhc_3,
    "mhc_4": metafaith_calibration_prompts.mhc_4,
    "mhc_5": metafaith_calibration_prompts.mhc_5,
    "mhc_6": metafaith_calibration_prompts.mhc_6,
    "mhc_7": metafaith_calibration_prompts.mhc_7,
    "mhc_8": metafaith_calibration_prompts.mhc_8,
    "mhc_9": metafaith_calibration_prompts.mhc_9,
    "mhc_10": metafaith_calibration_prompts.mhc_10,
    "mrc_1": metafaith_calibration_prompts.mrc_1,
    "mrc_2": metafaith_calibration_prompts.mrc_2,
    "mrc_3": metafaith_calibration_prompts.mrc_3,
    "mrc_4": metafaith_calibration_prompts.mrc_4,
    "mrc_5": metafaith_calibration_prompts.mrc_5,
    "mrc_6": metafaith_calibration_prompts.mrc_6,
    "mrc_7": metafaith_calibration_prompts.mrc_7,
    "mrc_8": metafaith_calibration_prompts.mrc_8,
    "mrc_9": metafaith_calibration_prompts.mrc_9,
    "mrc_10": metafaith_calibration_prompts.mrc_10,
}
