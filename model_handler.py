# model_handler.py

import gradio as gr
import json
import os
import re
from get_llm_answer import get_model_response, parse_model_response, get_atla_response
from jinja2 import Template

def select_evaluators(criteria_group, df_state, prompt_state, save_prompt_button):
    with gr.Group(visible=True) as model_selection_group:
        select_evaluators_button = gr.Button("Select Evaluators", visible=False)

        # Load the model_data from JSONL
        def load_model_data():
            model_data = {}
            try:
                script_dir = os.path.dirname(__file__)
                file_path = os.path.join(script_dir, "models.jsonl")
                with open(file_path, "r") as f:
                    for line in f:
                        model = json.loads(line)
                        model_data[model["name"]] = {
                            "organization": model["organization"],
                            "license": model["license"],
                            "api_model": model["api_model"],
                        }
            except FileNotFoundError:
                print("Warning: models.jsonl not found")
                return {}
            return model_data


        model_data = load_model_data()
        model_choices = list(model_data.keys())

        # Define dropdowns using model choices
        with gr.Row(visible=False) as evaluator_row:
               judge_a_dropdown = gr.Dropdown(
                   choices=["Atla Selene"], label="Judge A", value="Atla Selene", interactive=False
               )
               judge_b_dropdown = gr.Dropdown(
                   choices=model_choices, label="Judge B", value="Claude 3.5 Sonnet"
               )

        run_evaluation_button = gr.Button("Run Evaluation", visible=False)
        loading_spinner = gr.Markdown("Evaluation in progress...", visible=False)
        analyze_results_button = gr.Button("Analyze Results", visible=False)

        # Show evaluator selection UI
        def show_evaluator_selection():
            return {
                criteria_group: gr.update(visible=False),
                save_prompt_button: gr.update(visible=False),
                evaluator_row: gr.update(visible=True),
                run_evaluation_button: gr.update(visible=True),
            }

        # DONT FORGET TO RENAME THIS TO SAVE_PROMPT_BUTTON.Click
        save_prompt_button.click(
            fn=show_evaluator_selection,
            inputs=[],
            outputs=[
                #select_evaluators_button,
                save_prompt_button,
                criteria_group,
                evaluator_row,
                run_evaluation_button,
            ],
        )      

        # Run evaluation
        def run_evaluation(judge_a, judge_b):
            # Show loading spinner
            yield {loading_spinner: gr.update(visible=True)}
            
            # Get template and mappings from prompt state
            template_str = prompt_state.value['template']
            mappings = prompt_state.value['mappings']
            evaluation_criteria = mappings.get('evaluation_criteria')
            
            # Create Jinja template for Judge B only
            template = Template(template_str)
            
            # Submit prompt to chosen models
            for index, row in df_state.value.iterrows():
                # Create a context dictionary for this row
                context = {}
                model_context = None
                expected_output = None
                
                for key, column in mappings.items():
                    if key == 'evaluation_criteria':
                        continue  # Skip as we handle it separately
                    elif column and column != 'None':
                        context[key] = str(row[column])
                        if column == 'model_context':
                            model_context = str(row[column])
                        elif column == 'expected_model_output':
                            expected_output = str(row[column])
                
                # For Judge B, render the template using Jinja
                current_prompt = template.render(**context)
                # For Judge A (Atla Selene), call get_atla_response directly
                response_a = get_atla_response(
                    "atla-selene",
                    model_input=context.get('model_input'),
                    model_output=context.get('model_output'),
                    model_context=model_context,
                    expected_output=expected_output,
                    evaluation_criteria=evaluation_criteria
                )
                response_b = get_model_response(
                    judge_b, 
                    model_data.get(judge_b), 
                    current_prompt
                )
                
                # Parse the responses - handle Atla response differently
                if isinstance(response_a, dict):  # Atla response
                    score_a, critique_a = response_a['score'], response_a['critique']
                else:  # Error case
                    score_a, critique_a = "Error", response_a
                    
                score_b, critique_b = parse_model_response(response_b)
                
                df_state.value.loc[index, 'score_a'] = score_a
                df_state.value.loc[index, 'critique_a'] = critique_a
                df_state.value.loc[index, 'score_b'] = score_b
                df_state.value.loc[index, 'critique_b'] = critique_b
            
            import time
            time.sleep(2)
            
            # Hide loading spinner
            yield {loading_spinner: gr.update(visible=False)}
            
            # Start Generation Here
            markdown_table = df_state.value.to_string()
            yield {
                loading_spinner: gr.update(
                    value=f"### Evaluation Complete\n\n```\n{markdown_table}\n```",
                    visible=True
                ),
                analyze_results_button: gr.update(visible=True),
            }
            return response_a, response_b

        run_evaluation_button.click(
            fn=run_evaluation,
            inputs=[judge_a_dropdown, judge_b_dropdown],
            outputs=[loading_spinner, analyze_results_button],
        )



    return model_selection_group, df_state, analyze_results_button