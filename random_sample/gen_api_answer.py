from openai import OpenAI
import anthropic
from together import Together
import os       
from atla import Atla
from dotenv import load_dotenv
from .prompts import (
    JUDGE_SYSTEM_PROMPT
)
from transformers import AutoTokenizer
import requests
import json
import re

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
            model_id="atla-selene",
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

def get_selene_mini_response(model_name, prompt, system_prompt=None, max_tokens=500, temperature=0.01):
    """Get response from HF endpoint for Atla model"""
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {hf_api_key}",
            "Content-Type": "application/json"
        }
        
        # Create messages list for chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        model_id = "AtlaAI/Selene-1-Mini-Llama-3.1-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_api_key)
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "return_full_text": False,
                "temperature": temperature,
                "seed": 42,
                "add_generation_prompt": True
            }
        }
        
        response = requests.post(
            "https://bkp9p28gri93egqh.us-east-1.aws.endpoints.huggingface.cloud",
            headers=headers,
            json=payload
        )
        return response.json()[0]["generated_text"]
    except Exception as e:
        return f"Error with Atla model {model_name}: {str(e)}"

def parse_selene_mini_response(response_text):
    """Parse the response from Selene Mini to extract score and critique"""
    try:
        # Clean up the response text
        response_text = response_text.strip()
        
        # More flexible regex patterns
        reasoning_pattern = r'\*\*Reasoning:?\*\*\s*(.*?)(?=\*\*Result|$)'
        result_pattern = r'\*\*Result:?\*\*\s*(\d+)'
        
        reasoning_match = re.search(reasoning_pattern, response_text, re.DOTALL | re.IGNORECASE)
        result_match = re.search(result_pattern, response_text, re.IGNORECASE)
        
        if reasoning_match and result_match:
            critique = reasoning_match.group(1).strip()
            score = result_match.group(1)
            return {"score": score, "critique": critique}
        else:
            # If we can't parse it properly, let's return the raw response as critique
            return {
                "score": "Error",
                "critique": f"Failed to parse response. Raw response:\n{response_text}"
            }
    except Exception as e:
        return {
            "score": "Error",
            "critique": f"Error parsing response: {str(e)}\nRaw response:\n{response_text}"
        }