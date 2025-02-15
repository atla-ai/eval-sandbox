# data_handler.py

import gradio as gr
import pandas as pd
import json
def upload_test_data(df_state):
    with gr.Group() as data_upload_group:
        file_upload = gr.File(
            label="Upload JSON with test data incl. true labels as integers",
            file_types=[".json"],
        )
        import_button = gr.Button("Import Data", visible=False)
        # Show exactly 5 rows, no scrolling
        df_display = gr.Dataframe(
            visible=False,
            elem_classes=["truncate_cells"],
            label="Uploaded Data"
        )
        error_display = gr.Textbox(visible=False)

    def display_file_info(file):
        if file is not None:
            return {
                import_button: gr.update(visible=True),
                error_display: gr.update(visible=False)  # Hide previous errors
            }
        else:
            return {
                import_button: gr.update(visible=False),
                df_display: gr.update(visible=False),
                error_display: gr.update(visible=False)  # Hide previous errors
            }

    def import_data(file):
        if file is not None:
            try:
                df_state.value = pd.json_normalize(json.load(open(file.name)))

                return {
                    df_display: gr.update(value=df_state.value, visible=True),
                    import_button: gr.update(visible=False),
                    df_state: df_state,
                    error_display: gr.update(visible=False)  # Hide previous errors
                }
            except json.JSONDecodeError as e:
                return {
                    df_display: gr.update(visible=False),
                    error_display: gr.update(value="**Error:** Invalid JSON file. Please upload a valid JSON file.", visible=True),
                    import_button: gr.update(visible=True),
                    df_state: None
                }
            except Exception as e:
                return {
                    df_display: gr.update(visible=False),
                    error_display: gr.update(value=f"**Error:** {str(e)}", visible=True),
                    import_button: gr.update(visible=True),
                    df_state: None
                }
        else:
            return {
                df_display: gr.update(visible=False),
                import_button: gr.update(visible=True),
                df_state: None
            }

    file_upload.change(
        fn=display_file_info,
        inputs=file_upload,
        outputs=[import_button, df_display, error_display]
    )
    import_button.click(
        fn=import_data,
        inputs=file_upload,
        outputs=[df_display, import_button, df_state, error_display]
    )

    return data_upload_group, df_state
