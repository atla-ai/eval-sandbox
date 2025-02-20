# sandbox_runner.py

import gradio as gr
from data_handler import upload_test_data
from criteria_handler import select_evaluation_criteria
from model_handler import select_evaluators
from score_handler import handle_analysis
from random_sample_tab import random_sample_tab

def run_sandbox():
    with gr.Blocks(css="""
    .truncate_cells table {
        table-layout: fixed !important;
        width: 100% !important;
    }
    .truncate_cells table td,
    .truncate_cells table th {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 200px !important;
        text-align: left !important;
        vertical-align: top !important;
    }
    """) as demo:
        gr.Markdown("# Selene Playground")
        gr.Markdown("""Run evals with Selene and Selene-Mini in this interactive playground! 
                    <br>Check out Selene-Mini's [model card](https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B) or get started with the Selene API for free [here](https://www.atla-ai.com/sign-up?utm_source=huggingface&utm_medium=org_social&utm_campaign=SU_HF_atla_demo_sel1launch_).""")
        with gr.Tabs():
            # Random samples tab
            random_sample_tab()

            # Sandbox tab
            with gr.TabItem("Custom dataset"):
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

    demo.launch()

if __name__ == "__main__":
    run_sandbox()