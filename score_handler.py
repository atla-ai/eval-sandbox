import gradio as gr
import pandas as pd
import numpy as np

def handle_analysis(df_state, model_selection_group, analyze_results_button):
    with gr.Group(visible=False) as analysis_group:
        gr.Markdown("## Analysis")

        # Dropdown to select the accuracy measurement
        accuracy_measurement_dropdown = gr.Dropdown(
            choices=['Pairwise Accuracy', 'Pearson Correlation'],
            label='Select Accuracy Measurement'
        )

        # Dropdowns to select columns to compare
        with gr.Row():
            # First dropdown for ground truth labels
            ground_truth_dropdown = gr.Dropdown(
                choices=[],
                label='Select Ground Truth Labels'
            )
            # Second dropdown for first comparison column
            compare_column1_dropdown = gr.Dropdown(
                choices=[],
                label='Select First Column to Compare'
            )
            # Third dropdown for second comparison column
            compare_column2_dropdown = gr.Dropdown(
                choices=[],
                label='Select Second Column to Compare'
            )

        # Button to trigger calculation
        calculate_button = gr.Button('Calculate')

        # Output display
        result_output = gr.Textbox(label='Result', lines=10, interactive=False)

    # Event handler connections

    # When "Analyze Results" button is clicked, show analysis group and hide model_selection_group
    def show_analysis_group():
        df = df_state.value
        if df is not None:
            columns = df.columns.tolist()
        else:
            columns = []
        return (
            gr.update(visible=True),                 # Show analysis_group
            gr.update(visible=False),                # Hide model_selection_group
            gr.update(choices=columns),              # Update ground_truth_dropdown
            gr.update(choices=columns),              # Update compare_column1_dropdown
            gr.update(choices=columns),              # Update compare_column2_dropdown
        )

    analyze_results_button.click(
        fn=show_analysis_group,
        inputs=[],
        outputs=[
            analysis_group,
            model_selection_group,
            ground_truth_dropdown,
            compare_column1_dropdown,
            compare_column2_dropdown
        ]
    )

    # Calculate button click event
    def calculate_multiple_accuracies(measurement, ground_truth_col, col2_name, col3_name, df_state):
        df = df_state.value
        if df is None:
            return "No DataFrame available."

        missing_columns = [
            col for col in [ground_truth_col, col2_name, col3_name] if col not in df.columns
        ]
        if missing_columns:
            return f"Selected columns not found in DataFrame: {', '.join(missing_columns)}."

        # Prepare results for both comparisons
        output_texts = []

        # Compare ground_truth_col with col2_name
        result1 = calculate_accuracy(
            measurement, ground_truth_col, col2_name, df_state, compare_to_ground_truth=True
        )
        output_texts.append(
            f"Comparison between '{ground_truth_col}' and '{col2_name}':\n{result1}"
        )

        # Compare ground_truth_col with col3_name
        result2 = calculate_accuracy(
            measurement, ground_truth_col, col3_name, df_state, compare_to_ground_truth=True
        )
        output_texts.append(
            f"\nComparison between '{ground_truth_col}' and '{col3_name}':\n{result2}"
        )

        # Combine the results
        return "\n".join(output_texts)

    calculate_button.click(
        fn=calculate_multiple_accuracies,
        inputs=[
            accuracy_measurement_dropdown,
            ground_truth_dropdown,
            compare_column1_dropdown,
            compare_column2_dropdown,
            df_state
        ],
        outputs=result_output
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

    if measurement == 'Pairwise Accuracy':
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