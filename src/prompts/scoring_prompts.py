###################################
#### Assertion Prompt
###################################

ASSERTION_PROMPT = """You will be given a proposed answer. Your task is to extract any assertions made by the proposed answer.  If the proposed answer exhibits punting, the extracted assertion should be the empty string.

Here are some examples:

Proposed answer: I think JJ appears in Criminal Minds season 6, episode 2.
Extracted assertion: JJ appears in Criminal Minds season 6, episode 2.

Proposed answer: JJ appears in Criminal Minds season 6, in either the first or second episode.
Extracted assertion: JJ appears in Criminal Minds season 6, episode 1.
Extracted assertion: JJ appears in Criminal Minds season 6, episode 2.

Proposed answer: I’m not really sure about this, but I think the episode in which JJ appears in Criminal Minds may be episode 2 in season 6.
Extracted assertion: JJ appears in Criminal Minds season 6, episode 2.

Proposed answer: I don’t know which episode you’re referring to.
Extracted assertion:

Proposed answer: unanswerable
Extracted assertion:

Proposed answer: Episode 1, and John Reeves.
Extracted assertion: The answer is Episode 1.
Extracted assertion: The answer is John Reeves.

Proposed answer: not enough information
Extracted assertion:

Proposed answer: I think the answer is 3. You can vote when you turn 18
Extracted assertion: The answer is 3.

Proposed answer: Hmm, this is a tough one. I’m really not sure what the answer is, but maybe... just maybe... the answer is 2007?
Extracted assertion: The answer is 2007

Proposed answer: {answer}"""

ASSERTION_PATTERN = r"Extracted assertion:\s*(.*?)(?=\n|$)"


###################################
#### Decisiveness Prompt
###################################

SP3 = "You are an expert-level rater of the assertiveness and decisiveness of written texts."

DEC_INSTR = """We will present you with a statement. Your task is to evaluate the linguistic assertiveness of it. After reading the statement, rate how assertive you find it on a scale from {MIN} (Not at all assertive) to {MAX} (Extremely assertive). Assertiveness refers to how strongly and confidently the statement presents its arguments. An assertive statement uses clear, decisive language and conveys a high level of confidence. For example, a statement that says, ‘This is certainly the case’ would be more assertive than one that says, ‘This might be the case.’. Please be consistent in your ratings. A ‘{MIN}’ should reflect language that is hesitant, uncertain, or non-committal, while a ‘{MAX}’ should reflect language that is confident, decisive, and leaves no room for doubt. You should pay special attention to the usage of any hedging modifiers in the original answer, used to convey uncertainty in the truthfulness of the assertion. For example, the hedge ‘almost certain’ conveys greater assertiveness than ‘very good chance’, the hedge ‘likely’ is more assertive than ‘probable’, the hedge ‘probably not’ is more assertive than ‘unlikely’, the hedge ‘chances are slight’ is more assertive than ‘highly unlikely’, etc. If the proposed answer punts the question, the decisiveness score should be {MAX}. ONLY PROVIDE A FLOAT BETWEEN {MIN} AND {MAX} AS YOUR RESPONSE.
{fs_prompt}
Statement: {text}
Rating:"""

DEC_FS_PROMPT = """
Statement: There's little chance that anyone other than José Maria Arguedas Altamirano, a Peruvian novelist, anthropologist, and poet, directed Mata.
Rating: {MAX*0.129}

Statement: We doubt anyone knows for certain who the screenwriter for *Two* was, but it may have been Rob Reiner.
Rating: {MAX*0.224}

Statement: Joris Delle was probably born in Épinal, France.
Rating: {MAX*0.689}

Statement: The odds of Carol Reed being the director of The Key are about even.
Rating: {MAX*0.506}

Statement: There was no convincing evidence that delaying the introduction of these foods would prevent allergies.
Rating: {MAX*0.504}

Statement: supported
Rating: {MAX}

Statement: Irreversible losses of decomposition pathways are likely to play key roles in the evolutionary stability of these ubiquitous mutualisms.
Rating: {MAX*0.267}

Statement: Are you open to changing your view? What would it take to change your view?
Rating: {MAX*0.011}

Statement: Simpson lives in a fact-free world. Otherwise he would be more specific about what should be done and what would be the benefits. Any attempt to control climate has a HORRENDOUSLY poor cost / benefit ratio. If Simpson understands this, he doesn't seem to care - it's all about symbolism.
Rating: {MAX*0.779}

Statement: The answer is 100.
Rating: {MAX}

Statement: not enough information
Rating: {MAX}
"""

DEC_SCORE_PATTERN = r"([+-]?\d*\.\d+|\d+)(?=\s*[^0-9]|\s*$)"


###################################
#### Uncertainty Prompt
###################################

UNCERTAINTY_PROMPT = """Context: {context}
Assertion: {assertion}
Is the assertion consistent with the context above?
Answer Yes or No:"""

UNCERTAINTY_PATTERN = r"(?i)(?:Answer\s+yes\s+or\s+no:\s*)?\b(yes|no|N/A)\b"

UNCERTAINTY_MAPPING =  {
    "yes": 0.0,     # factual / no contradiction
    "no": 1.0,      # nonfactual / contradiction
    "n/a": 0.5,     # somewhat inaccurate / unclear
}


###################################
#### Performance Evaluation Prompt
###################################

LLM_EVAL_PROMPT = """Determine whether the predicted answer contains text semantically equivalent to any of the ground truth options. Output ONLY True or False.
ground truth options = {targets}
predicted answer = {pred}"""
