# sandbox_runner.py

import gradio as gr
from data_handler import upload_test_data
from criteria_handler import select_evaluation_criteria
from model_handler import select_evaluators
from score_handler import handle_analysis
from hello_world_tab import hello_world_tab

def run_sandbox():
    with gr.Blocks() as demo:
        gr.Markdown("# Atla Eval Sandbox testing")
        with gr.Tabs():
            # Sandbox tab
            with gr.TabItem("Sandbox"):
                # Initialize state object to track the DataFrame
                df_state = gr.State(value=None)
                # Initialize state object to track the prompt
                prompt_state = gr.State(value=None)
                # Initialize the evaluation_complete flag
                evaluation_complete = gr.State(value=None)

                # Data upload
                data_upload_group, df_state = upload_test_data(df_state)
                
                # Criteria selection
                criteria_group, df_state, prompt_state, save_prompt_button = \
                    select_evaluation_criteria(data_upload_group, df_state, prompt_state)

                # Models selection
                model_selection_group, df_state, analyze_results_button = \
                    select_evaluators(criteria_group, df_state, prompt_state, save_prompt_button)

                # Result analysis
                handle_analysis(df_state, model_selection_group, analyze_results_button)

            # Hello World tab
            hello_world_tab()

    demo.launch()

if __name__ == "__main__":
    run_sandbox()