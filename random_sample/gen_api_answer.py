from openai import OpenAI
import anthropic
from together import Together
import os       
from atla import Atla
from dotenv import load_dotenv
from .prompts import (
    JUDGE_SYSTEM_PROMPT
)

load_dotenv()

# Initialize clients
anthropic_client = anthropic.Anthropic()
openai_client = OpenAI()
together_client = Together()
hf_api_key = os.getenv("HF_API_KEY")

atla_client = Atla()

def get_openai_response(model_name, prompt, system_prompt=JUDGE_SYSTEM_PROMPT, max_tokens=500, temperature=0):
    """Get response from OpenAI API"""
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with OpenAI model {model_name}: {str(e)}"

def get_anthropic_response(model_name, prompt, system_prompt=JUDGE_SYSTEM_PROMPT, max_tokens=500, temperature=0):
    """Get response from Anthropic API"""
    try:
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        return response.content[0].text
    except Exception as e:
        return f"Error with Anthropic model {model_name}: {str(e)}"


def get_atla_response(model_name, prompt, system_prompt=None, max_tokens=500, temperature=0.01):
    """Get response from Atla API"""
    try:
        # Extract components from the prompt data
        model_input = prompt.get('human_input', '')
        model_output = prompt.get('ai_response', '')
        expected_output = prompt.get('ground_truth')
        evaluation_criteria = prompt.get('eval_criteria', '')

        response = atla_client.evaluation.create(
            model_id=model_name,  # Will be either "atla-selene" or "atla-selene-mini"
            model_input=model_input,
            model_output=model_output,
            expected_model_output=expected_output if expected_output else None,
            evaluation_criteria=evaluation_criteria,
        )
        
        # Return the score and critique directly
        return {
            "score": response.result.evaluation.score,
            "critique": response.result.evaluation.critique
        }
    except Exception as e:
        return f"Error with Atla model {model_name}: {str(e)}"