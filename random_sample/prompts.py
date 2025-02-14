# Default values for compatible mode
DEFAULT_EVAL_CRITERIA = """Does the model provide relevant and useful responses to the user's needs or questions?

Scoring Rubric:
Score 1: The model's responses are irrelevant or unhelpful to the user's needs or queries.
Score 2: The model sometimes provides helpful information, but often fails to address the user's actual needs or questions.
Score 3: The model generally provides helpful responses that address the user's needs, though it may occasionally miss the mark.
Score 4: The model regularly provides helpful responses that are well-aligned with the user's inquiries, with only rare inaccuracies.
Score 5: The model consistently offers highly relevant and useful responses that perfectly cater to the user's needs and inquiries."""

# Default Eval Prompt
DEFAULT_EVAL_PROMPT = """Does the model provide relevant and useful responses to the user's needs or questions?

Scoring Rubric:
Score 1: The model's responses are irrelevant or unhelpful to the user's needs or queries.
Score 2: The model sometimes provides helpful information, but often fails to address the user's actual needs or questions.
Score 3: The model generally provides helpful responses that address the user's needs, though it may occasionally miss the mark.
Score 4: The model regularly provides helpful responses that are well-aligned with the user's inquiries, with only rare inaccuracies.
Score 5: The model consistently offers highly relevant and useful responses that perfectly cater to the user's needs and inquiries.

[User Query]: {{input}}

[AI Response]: {{response}}"""

# Split the eval prompt into editable and fixed parts
DEFAULT_EVAL_PROMPT_EDITABLE = """Does the model provide relevant and useful responses to the user's needs or questions?

Scoring Rubric:
Score 1: The model's responses are irrelevant or unhelpful to the user's needs or queries.
Score 2: The model sometimes provides helpful information, but often fails to address the user's actual needs or questions.
Score 3: The model generally provides helpful responses that address the user's needs, though it may occasionally miss the mark.
Score 4: The model regularly provides helpful responses that are well-aligned with the user's inquiries, with only rare inaccuracies.
Score 5: The model consistently offers highly relevant and useful responses that perfectly cater to the user's needs and inquiries."""

# Fixed suffix that will always be appended
FIXED_EVAL_SUFFIX = """
[User Query]: {{human_input}}

[AI Response]: {{ai_response}}"""

ATLA_PROMPT = """You are tasked with evaluating a response based on a given instruction (which may contain an Input) and a scoring rubric that serve as the evaluation standard. Provide a comprehensive feedback on the response quality strictly adhering to the scoring rubric, without any general evaluation. Follow this with a score integer, referring to the scoring rubric. Avoid generating any additional opening, closing, or explanations.
  Here are some rules of the evaluation:
  (1) You should prioritize evaluating whether the response satisfies the provided rubric. The basis of your score should depend exactly on the rubric. However, the response does not need to explicitly address points raised in the rubric. Rather, evaluate the response based on the criteria outlined in the rubric.

  Your reply should strictly follow this format:
  **Reasoning:** <Your feedback>

  **Result:** <Your score>

  Here is the data:

  Instruction:
  ```
  {human_input}
  ```

  Response:
  ```
  {ai_response}
  ```

  Score Rubrics:
  {eval_criteria}"""

ATLA_PROMPT_WITH_REFERENCE = """You are tasked with evaluating a response based on a given instruction (which may contain an Input) and a scoring rubric and reference answer that serve as the evaluation standard. Provide a comprehensive feedback on the response quality strictly adhering to the scoring rubric, without any general evaluation. Follow this with a score, referring to the scoring rubric. Avoid generating any additional opening, closing, or explanations.

  Here are some rules of the evaluation:
  (1) You should prioritize evaluating whether the response satisfies the provided rubric. The basis of your score should depend exactly on the rubric. However, the response does not need to explicitly address points raised in the rubric. Rather, evaluate the response based on the criteria outlined in the rubric.

  Your reply should strictly follow this format:
  **Reasoning:** <Your feedback>

  **Result:** <Your score>

  Here is the data:

  Instruction:
  ```
  {human_input}
  ```

  Response:
  ```
  {ai_response}
  ```

  Score Rubrics:
  {eval_criteria}

  Reference answer:
  {ground_truth_input}"""

# Judge system prompt for non-Prometheus models
JUDGE_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate based on the user's instruction. Your output format should strictly adhere to JSON as follows: {"feedback": "<write feedback>", "result": <numerical score>}. Ensure the output is valid JSON, without additional formatting or explanations.""" 