import re
import os
import google.generativeai as genai
from google.generativeai import GenerationConfig

from src.prompts.scoring_prompts import ASSERTION_PROMPT, ASSERTION_PATTERN


def get_assertions(answer, assertion_max_tokens=500):

    # Generation Config
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))     # gemini api key
    model = genai.GenerativeModel("gemini-2.0-flash")
    generation_config = GenerationConfig(
        max_output_tokens=assertion_max_tokens, 
        candidate_count=1,
    )
            
    # Format Prompt & Get Response
    prompt = ASSERTION_PROMPT.format(
        answer=answer,
    )

    try:
        response = model.generate_content(
            prompt, 
            generation_config=generation_config,
        ).text
    except Exception as e: 
        print("assertion extraction error: ", e)
        response = ""

    # Process Response
    extracted_assertions = re.findall(ASSERTION_PATTERN, response, re.IGNORECASE)

    # Return Assertions
    return extracted_assertions
