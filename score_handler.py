import gradio as gr
import pandas as pd
import numpy as np
import tempfile
import os

def handle_analysis(df_state, model_selection_group, analyze_results_button):
    with gr.Group(visible=False) as analysis_group:
        gr.Markdown("## Analysis")

        # Dropdown to select the accuracy measurement
        accuracy_measurement_dropdown = gr.Dropdown(
            choices=['Accuracy', 'Pearson Correlation'],
            label='Select Evaluation Metric'
        )

        # We remove the two compare dropdowns and only keep ground truth
        with gr.Row():
            ground_truth_dropdown = gr.Dropdown(
                choices=[],
                label='Select True Label Column'
            )

        # Define two side-by-side boxes for results
        with gr.Row():
            judge_a_result = gr.Textbox(
                label="Judge A Results",
                lines=10,
                interactive=False,
                visible=False
            )
            judge_b_result = gr.Textbox(
                label="Judge B Results",
                lines=10,
                interactive=False,
                visible=False
            )

        # Move the JSON output below those textboxes and buttons
        json_output = gr.File(label="Results .json", interactive=False, visible=False)

        # Now place the row of buttons AFTER the json_output
        with gr.Row():
            back_to_results_button = gr.Button("← Back to Results")
            calculate_button = gr.Button("Calculate")
            download_button = gr.Button("Download Results as JSON")

    # Show analysis group
    def show_analysis_group():
        df = df_state.value
        if df is not None:
            columns = df.columns.tolist()
        else:
            columns = []
        # Now we only update ground_truth_dropdown
        return (
            gr.update(visible=True),         # analysis_group
            gr.update(visible=False),        # model_selection_group
            gr.update(choices=columns),      # ground_truth_dropdown
        )

    analyze_results_button.click(
        fn=show_analysis_group,
        inputs=[],
        outputs=[
            analysis_group,
            model_selection_group,
            ground_truth_dropdown  # only this one
        ]
    )

    def back_to_results():
        return (
            gr.update(visible=False),  # Hide analysis_group
            gr.update(visible=True),   # Show model_selection_group
        )

    back_to_results_button.click(
        fn=back_to_results,
        inputs=[],
        outputs=[analysis_group, model_selection_group]
    )

    def calculate_multiple_accuracies(measurement, ground_truth_col, df_state):
        # Hard-code 'score_a' and 'score_b' as the columns to compare
        col2_name = "score_a"
        col3_name = "score_b"
        df = df_state.value
        if df is None:
            # Return two "No DataFrame" messages
            return (
                gr.update(value="No DataFrame available.", visible=True),
                gr.update(value="No DataFrame available.", visible=True)
            )

        # Check if user-supplied ground_truth_col is valid
        missing_columns = [col for col in [ground_truth_col, col2_name, col3_name] if col not in df.columns]
        if missing_columns:
            msg = f"Selected columns not found in DataFrame: {', '.join(missing_columns)}."
            # Return same message in both boxes
            return (
                gr.update(value=msg, visible=True),
                gr.update(value=msg, visible=True)
            )

        # Compare ground_truth_col with score_a
        result1 = calculate_accuracy(
            measurement, ground_truth_col, col2_name,
            df_state, compare_to_ground_truth=True
        )
        text_a = f"Comparison: '{ground_truth_col}' vs. 'Judge A'\n{result1}"

        # Compare ground_truth_col with score_b
        result2 = calculate_accuracy(
            measurement, ground_truth_col, col3_name,
            df_state, compare_to_ground_truth=True
        )
        text_b = f"Comparison: '{ground_truth_col}' vs. 'Judge B'\n{result2}"

        # Return them separately, each is for a different Textbox
        return (
            gr.update(value=text_a, visible=True),
            gr.update(value=text_b, visible=True)
        )

    # Now the calculate_button only expects measurement, ground_truth_col, df_state
    calculate_button.click(
        fn=calculate_multiple_accuracies,
        inputs=[
            accuracy_measurement_dropdown,
            ground_truth_dropdown,
            df_state
        ],
        outputs=[judge_a_result, judge_b_result]
    )

    def create_json_download(df_state):
        if df_state.value is None:
            return gr.update(value=None, visible=True)
        
        json_str = df_state.value.to_json(orient='records', indent=2)
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, 'atla_custom_eval_results.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return gr.update(value=file_path, visible=True)

    download_button.click(
        fn=create_json_download,
        inputs=[df_state],
        outputs=[json_output]
    )

# Helper functions

def calculate_accuracy(measurement, col1, col2, df_state, compare_to_ground_truth=False):
    df = df_state.value
    # No changes here (function remains sacred as per your request)
    if df is None:
        return "No DataFrame available."
    if col1 not in df.columns or col2 not in df.columns:
        return "Selected columns not found in DataFrame."

    results_df = pd.DataFrame()
    if compare_to_ground_truth:
        results_df['ground_truth'] = df[col1]
        results_df['predicted'] = df[col2]
    else:
        results_df['extracted_winner'] = df[col1]
        results_df['truth_result'] = df[col2]

    if measurement == 'Accuracy':
        result = process_pairwise_accuracy(results_df, compare_to_ground_truth)
        output_text = (
            f"Overall Accuracy: {result['overall_accuracy']}\n"
            f"Number of NaNs: {result['num_extracted_nan']}"
        )
    elif measurement == 'Pearson Correlation':
        result = process_single_rating_pearson_correlation(results_df, compare_to_ground_truth)
        output_text = (
            f"Pearson Correlation: {result['overall_pearson_correlation']}\n"
            f"Number of NaNs: {result['num_extracted_nan']}"
        )
    else:
        output_text = "Unknown measurement selected."

    return output_text

def process_pairwise_accuracy(results_df: pd.DataFrame, compare_to_ground_truth=False) -> dict:
    # Compute 'results' column based on whether comparing to ground truth
    if compare_to_ground_truth:
        # NEW: convert both columns to float
        results_df['ground_truth'] = results_df['ground_truth'].apply(convert_to_float_or_nan)
        results_df['predicted'] = results_df['predicted'].apply(convert_to_float_or_nan)

        results_df['results'] = results_df['ground_truth'] == results_df['predicted']
        num_extracted_nan = int(results_df['predicted'].isna().sum())
    else:
        results_df['results'] = results_df['extracted_winner'] == results_df['truth_result']
        num_extracted_nan = int(results_df['extracted_winner'].isna().sum())

    overall_accuracy = results_df['results'].mean()

    return {
        "overall_accuracy": overall_accuracy,
        "num_extracted_nan": num_extracted_nan,
    }

def process_single_rating_pearson_correlation(
    results_df: pd.DataFrame, compare_to_ground_truth=False
) -> dict:
    if compare_to_ground_truth:
        pred_col = 'predicted'
        truth_col = 'ground_truth'
    else:
        pred_col = 'extracted_winner'
        truth_col = 'truth_result'

    results_df[pred_col] = results_df[pred_col].apply(convert_to_float_or_nan)
    results_df[truth_col] = results_df[truth_col].apply(convert_to_float_or_nan)

    numerical_results = results_df.dropna(subset=[pred_col, truth_col])

    if len(numerical_results) == 0:
        pearson_corr = np.nan
    else:
        pearson_corr = numerical_results[pred_col].corr(numerical_results[truth_col])

    num_extracted_nan = int(results_df[pred_col].isna().sum())

    return {
        "overall_pearson_correlation": pearson_corr if not pd.isna(pearson_corr) else 0.0,
        "num_extracted_nan": num_extracted_nan,
    }

def convert_to_float_or_nan(extracted_input):
    if extracted_input is None or pd.isna(extracted_input):
        return np.nan
    try:
        return float(extracted_input)
    except ValueError:
        return np.nan