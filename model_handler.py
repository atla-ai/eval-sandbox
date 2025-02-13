# model_handler.py

import gradio as gr
import json
import os
import re
from get_llm_answer import get_model_response, parse_model_response

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
                   choices=model_choices, label="Judge A", value="GPT-4o"
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
            # Submit prompt to chosen models
            prompt_template = prompt_state.value
            for index, row in df_state.value.iterrows():
                variable_values_dict = row.to_dict()
                prompt_state.value = prompt_template
                for key, value in variable_values_dict.items():
                    prompt_state.value = prompt_state.value.replace(f"{{{{{key}}}}}", str(value))
                response_a = get_model_response(judge_a, model_data.get(judge_a), prompt_state.value)
                response_b = get_model_response(judge_b, model_data.get(judge_b), prompt_state.value)
                # Parse the responses
                score_a, critique_a = parse_model_response(response_a)
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

        # Make "Select Evaluators" button visible
        def make_select_button_visible(*args):
            return gr.update(visible=True)

        df_state.change(
            fn=make_select_button_visible,
            inputs=[],
            #outputs=select_evaluators_button,
            #outputs=save_prompt_button,
        )

    return model_selection_group, df_state, analyze_results_button