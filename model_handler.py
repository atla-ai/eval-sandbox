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

        with gr.Row(visible=False) as evaluator_row:
               judge_a_dropdown = gr.Dropdown(
                   choices=["Selene"], label="Judge A", value="Selene", interactive=False
               )
               judge_b_dropdown = gr.Dropdown(
                   choices=model_choices, label="Judge B", value="Claude 3.5 Sonnet"
               )

        loading_spinner = gr.Markdown("Evaluation in progress...", visible=False)

        evaluation_result_df = gr.Dataframe(
            visible=False,
            label="Evaluation Results",
            elem_classes=["truncate_cells"]
        )

        with gr.Row(visible=False) as evaluation_nav_row:
            back_to_criteria_button = gr.Button("← Back to Criteria", visible=False)
            run_evaluation_button = gr.Button("Run Evaluation", visible=False)
            analyze_results_button = gr.Button("Analyze Results", visible=False)

        def show_evaluator_selection(current_df):
            updates = {
                criteria_group: gr.update(visible=False),
                save_prompt_button: gr.update(visible=False),
                evaluator_row: gr.update(visible=True),
                evaluation_nav_row: gr.update(visible=True),
                run_evaluation_button: gr.update(visible=True),
                back_to_criteria_button: gr.update(visible=True),
                analyze_results_button: gr.update(visible=False),
                evaluation_result_df: gr.update(visible=False),
            }
            if (
                current_df.value is not None
                and hasattr(current_df.value, "attrs")
                and current_df.value.attrs.get("eval_done")
            ):
                updates[loading_spinner] = gr.update(value="### Evaluation Complete", visible=True)
                updates[evaluation_result_df] = gr.update(value=current_df.value, visible=True)
                updates[analyze_results_button] = gr.update(visible=True)

            return updates

        save_prompt_button.click(
            fn=show_evaluator_selection,
            inputs=[df_state],
            outputs=[
                save_prompt_button,
                criteria_group,
                evaluator_row,
                evaluation_nav_row,
                run_evaluation_button,
                back_to_criteria_button,
                loading_spinner,
                analyze_results_button,
                evaluation_result_df,
            ],
        )

        def back_to_criteria():
            return {
                save_prompt_button: gr.update(visible=True),
                criteria_group: gr.update(visible=True),
                evaluator_row: gr.update(visible=False),
                evaluation_nav_row: gr.update(visible=False),
                run_evaluation_button: gr.update(visible=False),
                loading_spinner: gr.update(visible=False),
                analyze_results_button: gr.update(visible=False),
                evaluation_result_df: gr.update(visible=False),
            }

        back_to_criteria_button.click(
            fn=back_to_criteria,
            inputs=[],
            outputs=[
                save_prompt_button,
                criteria_group,
                evaluator_row,
                evaluation_nav_row,
                run_evaluation_button,
                loading_spinner,
                analyze_results_button,
                evaluation_result_df
            ],
        )

        # Run evaluation
        def run_evaluation(judge_a, judge_b):
            # 1) Immediately hide old results and disable navigation while running
            yield {
                loading_spinner: gr.update(value="Evaluation in progress...", visible=True),
                evaluation_result_df: gr.update(visible=False),
                analyze_results_button: gr.update(visible=False),
                run_evaluation_button: gr.update(interactive=False),
                back_to_criteria_button: gr.update(interactive=False),
            }

            # Perform the actual evaluation
            template_str = prompt_state.value['template']
            mappings = prompt_state.value['mappings']
            evaluation_criteria = mappings.get('evaluation_criteria')

            template = Template(template_str)

            for index, row in df_state.value.iterrows():
                context = {}
                model_context = None
                expected_output = None

                for key, column in mappings.items():
                    if key == 'evaluation_criteria':
                        continue
                    elif column and column != 'None':
                        context[key] = str(row[column])
                        if column == 'model_context':
                            model_context = str(row[column])
                        elif column == 'expected_model_output':
                            expected_output = str(row[column])

                # Render the template for Judge B
                current_prompt = template.render(**context)
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

                # Parse ATLA response
                if isinstance(response_a, dict):
                    score_a, critique_a = response_a['score'], response_a['critique']
                else:
                    score_a, critique_a = "Error", response_a

                score_b, critique_b = parse_model_response(response_b)

                df_state.value.loc[index, 'score_a'] = score_a
                df_state.value.loc[index, 'critique_a'] = critique_a
                df_state.value.loc[index, 'score_b'] = score_b
                df_state.value.loc[index, 'critique_b'] = critique_b

            import time
            time.sleep(2)  # simulating time-consuming operations

            # 2) Hide spinner
            yield {loading_spinner: gr.update(visible=False)}

            # 3) Show final results and re-enable buttons
            yield {
                loading_spinner: gr.update(value="### Evaluation Complete", visible=True),
                evaluation_result_df: gr.update(value=df_state.value, visible=True),
                analyze_results_button: gr.update(visible=True),
                run_evaluation_button: gr.update(interactive=True),
                back_to_criteria_button: gr.update(interactive=True),
            }

            if hasattr(df_state.value, "attrs"):
                df_state.value.attrs["eval_done"] = True

        # Include back_to_criteria_button & run_evaluation_button in outputs so we can update them
        run_evaluation_button.click(
            fn=run_evaluation,
            inputs=[judge_a_dropdown, judge_b_dropdown],
            outputs=[
                loading_spinner,
                evaluation_result_df,
                analyze_results_button,
                run_evaluation_button,
                back_to_criteria_button,
            ],
        )

    return model_selection_group, df_state, analyze_results_button