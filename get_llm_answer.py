# get_llm_answer.py

from openai import OpenAI
import anthropic
from together import Together
import json
import re
from atla import atla

from dotenv import load_dotenv
load_dotenv()

# Initialize clients
anthropic_client = anthropic.Anthropic()
openai_client = OpenAI()
together_client = Together()
atla_client = atla.Atla()

SYSTEM_PROMPT = """Please act as an impartial judge and evaluate based on the user's instruction. Your output format should strictly adhere to JSON as follows: {"feedback": "<write feedback>", "result": <numerical score>}. Ensure the output is valid JSON, without additional formatting or explanations."""

def get_openai_response(model_name, prompt):
    """Get response from OpenAI API"""
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with OpenAI model {model_name}: {str(e)}"


def get_anthropic_response(model_name, prompt):
    """Get response from Anthropic API"""
    try:
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=1000,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        return response.content[0].text
    except Exception as e:
        return f"Error with Anthropic model {model_name}: {str(e)}"


def get_together_response(model_name, prompt):
    """Get response from Together API"""
    try:
        response = together_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with Together model {model_name}: {str(e)}"


def get_atla_response(model_name, model_input, model_output, model_context, expected_output, evaluation_criteria):
    """Get response from Atla API"""
    try:
        response = atla_client.evaluation.create(
            model_id=model_name,
            model_input=model_input,
            model_output=model_output,
            model_context=model_context,
            expected_model_output=expected_output,
            evaluation_criteria=evaluation_criteria,
        )
        # Return the score and critique directly from the evaluation result
        return {
            "score": response.result.evaluation.score,
            "critique": response.result.evaluation.critique
        }
    except Exception as e:
        return f"Error with Atla model {model_name}: {str(e)}"


def get_model_response(model_name, model_info, prompt=None, **kwargs):
    """Get response from appropriate API based on model organization"""
    if not model_info:
        return "Model not found or unsupported."

    api_model = model_info["api_model"]
    organization = model_info["organization"]

    try:
        if organization == "Atla":
            return get_atla_response(
                api_model,
                kwargs.get('model_input'),
                kwargs.get('model_output'),
                kwargs.get('model_context'),
                kwargs.get('expected_output'),
                kwargs.get('evaluation_criteria')
            )
        elif organization == "OpenAI":
            return get_openai_response(api_model, prompt)
        elif organization == "Anthropic":
            return get_anthropic_response(api_model, prompt)
        else:
            # All other organizations use Together API
            return get_together_response(api_model, prompt)
    except Exception as e:
        return f"Error with {organization} model {model_name}: {str(e)}"


def parse_model_response(response):
    try:
        # Debug print
        print(f"Raw model response: {response}")

        # First try to parse the entire response as JSON
        try:
            data = json.loads(response)
            return str(data.get("result", "N/A")), data.get("feedback", "N/A")
        except json.JSONDecodeError:
            # If that fails (typically for smaller models), try to find JSON within the response
            json_match = re.search(r"{.*}", response)
            if json_match:
                data = json.loads(json_match.group(0))
                return str(data.get("result", "N/A")), data.get("feedback", "N/A")
            else:
                return "Error", f"Failed to parse response: {response}"

    except Exception as e:
        # Debug print for error case
        print(f"Failed to parse response: {str(e)}")
        return "Error", f"Failed to parse response: {response}"