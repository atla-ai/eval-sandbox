# criteria_handler.py

import gradio as gr
import re
from eval_criteria_library import EXAMPLE_METRICS

def select_evaluation_criteria(data_upload_group, df_state, prompt_state):
    with gr.Group(visible=True) as criteria_group:
        select_eval_criteria_button = gr.Button("Select Evaluation Criteria", visible=False)

        criteria_dropdown = gr.Dropdown(
            choices=list(EXAMPLE_METRICS.keys()),
            label="Choose Evaluation Criteria",
            value=list(EXAMPLE_METRICS.keys())[0],
            visible=False
        )

        with gr.Row(visible=False) as mapping_row:
            with gr.Column():
                # Left column - Evaluation Criteria Editor
                prompt_editor = gr.Textbox(
                    label="Evaluation Criteria",
                    lines=15,
                    visible=False,
                    placeholder="Enter the evaluation criteria/rubric here..."
                )
            with gr.Column():
                # Right column - Required and Optional Variable Mapping
                # Required mappings
                input_mapping = gr.Dropdown(
                    choices=[], 
                    label="Map 'model_input' to column (Required)",
                    interactive=True, 
                    visible=False
                )
                output_mapping = gr.Dropdown(
                    choices=[], 
                    label="Map 'model_output' to column (Required)",
                    interactive=True, 
                    visible=False
                )
                # Optional mappings
                context_mapping = gr.Dropdown(
                    choices=[], 
                    label="Map 'model_context' to column (Optional)",
                    interactive=True, 
                    visible=False
                )
                expected_output_mapping = gr.Dropdown(
                    choices=[], 
                    label="Map 'expected_model_output' to column (Optional)",
                    interactive=True, 
                    visible=False
                )
        # We'll place the "Back to Data" and "Select Evaluators" within the same row:
        with gr.Row(visible=False) as nav_row:
            back_to_data_button = gr.Button("← Back to Data", visible=False)
            save_prompt_button = gr.Button("Select Evaluators", visible=False)

        def update_column_choices(df_state):
            df = df_state.value
            columns = df.columns.tolist() if df is not None else []
            return {
                input_mapping: gr.update(choices=columns, visible=True),
                output_mapping: gr.update(choices=columns, visible=True),
                context_mapping: gr.update(choices=['None'] + columns, visible=True),
                expected_output_mapping: gr.update(choices=['None'] + columns, visible=True)
            }

        def update_prompt(selected_criteria, df_state):
            if selected_criteria in EXAMPLE_METRICS:
                evaluation_criteria = EXAMPLE_METRICS[selected_criteria]['prompt']
            else:
                evaluation_criteria = ""
            updates = {prompt_editor: gr.update(value=evaluation_criteria, visible=True)}
            updates.update(update_column_choices(df_state))
            return updates

        def show_criteria_selection():
            default_criterion = list(EXAMPLE_METRICS.keys())[0]
            evaluation_criteria = EXAMPLE_METRICS[default_criterion]['prompt']
            updates = {
                select_eval_criteria_button: gr.update(visible=False),
                criteria_dropdown: gr.update(visible=True),
                prompt_editor: gr.update(value=evaluation_criteria, visible=True),
                data_upload_group: gr.update(visible=False),
                mapping_row: gr.update(visible=True),
                # Show the nav row and buttons
                nav_row: gr.update(visible=True),
                back_to_data_button: gr.update(visible=True),
                save_prompt_button: gr.update(visible=True),
            }
            updates.update(update_column_choices(df_state))
            return updates

        def save_prompt(evaluation_criteria, input_col, output_col, context_col, expected_output_col):
            # Use the actual Jinja template with proper Jinja syntax and raw JSON
            template = '''You are tasked with evaluating a response based on a given instruction (which may contain an Input) and a scoring rubric. Provide a comprehensive feedback on the response quality strictly adhering to the scoring rubric, without any general evaluation. Follow this with a score, referring to the scoring rubric. Avoid generating any additional opening, closing, or explanations.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the response satisfies the provided rubric. The basis of your score should depend exactly on the rubric. However, the response does not need to explicitly address points raised in the rubric. Rather, evaluate the response based on the criteria outlined in the rubric.

Your reply should strictly follow this format:
Your output format should strictly adhere to JSON as follows: {% raw %}{"feedback": "<write feedback>", "result": <numerical score>}{% endraw %}. Ensure the output is valid JSON, without additional formatting or explanations.

Here is the data.

{% if model_context is defined and model_context %}Context:
```
{{ model_context }}
```

{% endif %}Instruction:
```
{{ model_input }}
```

Response:
```
{{ model_output }}
```

Score Rubrics:
{{ evaluation_criteria }}

{% if expected_model_output is defined and expected_model_output %}Reference answer:
{{ expected_model_output }}{% endif %}'''
            
            # Create mapping dictionary
            mapping_dict = {
                'model_input': input_col,
                'model_output': output_col,
                'evaluation_criteria': evaluation_criteria
            }
            
            # Add optional mappings if selected
            if context_col != 'None':
                mapping_dict['model_context'] = context_col
            if expected_output_col != 'None':
                mapping_dict['expected_model_output'] = expected_output_col
                
            prompt_state.value = {
                'template': template,
                'mappings': mapping_dict
            }

        # Update event handlers
        select_eval_criteria_button.click(
            fn=show_criteria_selection,
            inputs=[],
            outputs=[
                
                select_eval_criteria_button,
                criteria_dropdown,
                prompt_editor,
               
                data_upload_group,
                mapping_row,
                nav_row,
                back_to_data_button,
                save_prompt_button
            ,
                input_mapping, output_mapping, context_mapping, expected_output_mapping
            ]
        )

        criteria_dropdown.change(
            fn=update_prompt,
            inputs=[criteria_dropdown, df_state],
            outputs=[prompt_editor, input_mapping, output_mapping, context_mapping, expected_output_mapping]
        )

        def make_select_button_visible(df_value):
            if df_value is not None:
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)

        df_state.change(
            fn=make_select_button_visible,
            inputs=df_state,
            outputs=select_eval_criteria_button
        )

        save_prompt_button.click(
            fn=save_prompt,
            inputs=[
                prompt_editor, input_mapping, output_mapping,
                context_mapping, expected_output_mapping
            ],
            outputs=[]
        )

        # BACK BUTTON: Hide the criteria UI, show the data upload UI
        def back_to_data():
            return {
                # show data upload group again
                data_upload_group: gr.update(visible=True),
                # hide the criteria group
                criteria_dropdown: gr.update(visible=False),
                prompt_editor: gr.update(visible=False),
                mapping_row: gr.update(visible=False),
                nav_row: gr.update(visible=False),
                # make "Select Evaluation Criteria" button visible again
                select_eval_criteria_button: gr.update(visible=True),
            }

        back_to_data_button.click(
            fn=back_to_data,
            inputs=[],
            outputs=[
                data_upload_group,
                criteria_dropdown,
                prompt_editor,
                mapping_row,
                nav_row,
                select_eval_criteria_button
            ]
        )

    # Return both the criteria rule group, the df_state, prompt_state, save_prompt_button
    return criteria_group, df_state, prompt_state, save_prompt_button
