##################################################
################## TASK PROMPTS ##################
##################################################

QA_SHORT_PROMPT = """
Question: {task_input}{hedge_prompt}
Answer:"""

QA_SHORT_PROMPT_ANSWERABILITY = """
Question: {task_input}\nIf the question is unanswerable, indicate so.{hedge_prompt}
Answer:"""

### MCQ TASKS

MCQ_UNIQUE_LETTERS_PROMPT = """Question: {task_input}\nWhat is the letter corresponding to the correct answer choice?{hedge_prompt}
Answer:"""

MCQ_UNIQUE_NUMBERS_PROMPT = """Question: {task_input}\nWhat is the number corresponding to the correct answer choice?{hedge_prompt}
Answer:"""

### HALLUCINATION DETECTION

# Alt formulation: choose which of two options is hallucinated
HD_PROMPT = """Question: {task_input}\nDoes the proposed answer to the question contain hallucination?{hedge_prompt}
Judgment:"""

### NLU

SUPERGLUE_PROMPT = """Question: {task_input}
Succinctly answer the question.{hedge_prompt}
Answer:"""


### MATH BENCHMARKS

MATH_PROMPT = """Problem: {task_input}\nWhat is the final answer to the math problem? Provide only the final answer, with MINIMAL intermediate steps. Format your answer using LaTeX.{hedge_prompt}
Final Answer:"""

UMWP_PROMPT = """Question: {task_input}\nIf the question is unanswerable, indicate so. If not, what is the final answer to the math problem? Provide only the final answer, with MINIMAL intermediate steps.{hedge_prompt}
Final Answer:"""

