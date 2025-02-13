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
                # Left column - Prompt Editor
                prompt_editor = gr.Textbox(
                    label="Evaluation Prompt",
                    lines=15,
                    visible=False
                )
            with gr.Column():
                # Right column - Variable Mapping
                variable_dropdowns = []
                for i in range(5):  # Assume up to 5 variables
                    dropdown = gr.Dropdown(choices=[], label=f"Variable {i+1}", interactive=True, visible=False)
                    variable_dropdowns.append(dropdown)

        # Now declare your save_prompt_button AFTER the dropdown
        save_prompt_button = gr.Button("Select Evaluators", visible=False)

        def extract_variables(prompt):
            return re.findall(r'\{\{(.*?)\}\}', prompt)

        def update_variable_mappings(prompt, df_state):
            df = df_state.value
            columns = df.columns.tolist() if df is not None else []
            variables = extract_variables(prompt)
            updates = {}
            for i, dropdown in enumerate(variable_dropdowns):
                if i < len(variables):
                    updates[dropdown] = gr.update(
                        choices=columns,
                        label=f"Map '{{{{{variables[i]}}}}}' to column",
                        visible=True
                    )
                else:
                    updates[dropdown] = gr.update(visible=False)
            return updates

        def update_prompt(selected_criteria, df_state):
            if selected_criteria in EXAMPLE_METRICS:
                prompt = EXAMPLE_METRICS[selected_criteria]['prompt']
            else:
                prompt = ""
            prompt_update = gr.update(value=prompt, visible=True)
            variable_updates = update_variable_mappings(prompt, df_state)
            updates = {prompt_editor: prompt_update}
            updates.update(variable_updates)
            return updates

        def show_criteria_selection():
            default_criterion = list(EXAMPLE_METRICS.keys())[0]
            prompt = EXAMPLE_METRICS[default_criterion]['prompt']
            prompt_update = gr.update(value=prompt, visible=True)
            variable_updates = update_variable_mappings(prompt, df_state)
            updates = {
                select_eval_criteria_button: gr.update(visible=False),
                criteria_dropdown: gr.update(visible=True),
                prompt_editor: prompt_update,
                data_upload_group: gr.update(visible=False),
                mapping_row: gr.update(visible=True),
                save_prompt_button: gr.update(visible=True),
            }
            updates.update(variable_updates)
            return updates

        select_eval_criteria_button.click(
            fn=show_criteria_selection,
            inputs=[],
            outputs=[select_eval_criteria_button, criteria_dropdown, prompt_editor, data_upload_group, mapping_row, save_prompt_button] + variable_dropdowns
        )

        criteria_dropdown.change(
            fn=update_prompt,
            inputs=[criteria_dropdown, df_state],
            outputs=[prompt_editor] + variable_dropdowns
        )

        prompt_editor.change(
            fn=update_variable_mappings,
            inputs=[prompt_editor, df_state],
            outputs=variable_dropdowns
        )

        def on_prompt_change(prompt):
            prompt_state.value = prompt  # Update prompt_state with the latest prompt

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

        def save_prompt(prompt, *variable_mappings):
            variables = extract_variables(prompt)
            mapping_dict = {var: mapping for var, mapping in zip(variables, variable_mappings) if mapping}
            # Replace variables with mapped column names
            for var in variables:
                if mapping_dict.get(var):
                    prompt = prompt.replace(f"{{{{{var}}}}}", f"{{{{{mapping_dict[var]}}}}}")
            prompt_state.value = prompt

        save_prompt_button.click(
            fn=save_prompt,
            inputs=[prompt_editor] + variable_dropdowns,
            outputs=[]
        )

    return criteria_group, df_state, prompt_state, save_prompt_button
