import json
import re
import gradio as gr

from dotenv import load_dotenv
load_dotenv()

from .gen_api_answer import (
    get_atla_response
)

from .prompts import (
    DEFAULT_EVAL_CRITERIA,
    DEFAULT_EVAL_PROMPT,
    DEFAULT_EVAL_PROMPT_EDITABLE,
    FIXED_EVAL_SUFFIX
)

from .random_sample_generation import (
    get_random_human_ai_pair,
    get_random_human_ai_ground_truth_pair,
    generate_ai_response
)   

from common import CSS_STYLES, MAIN_TITLE, HOW_IT_WORKS

def parse_variables(prompt):
    # Extract variables enclosed in double curly braces
    variables = re.findall(r"{{(.*?)}}", prompt)
    # Remove duplicates while preserving order
    seen = set()
    variables = [
        x.strip() for x in variables if not (x.strip() in seen or seen.add(x.strip()))
    ]
    return variables


def get_final_prompt(eval_prompt, variable_values):
    # Replace variables in the eval prompt with their values
    for var, val in variable_values.items():
        eval_prompt = eval_prompt.replace("{{" + var + "}}", val)
    return eval_prompt


def populate_random_example(request: gr.Request, compatible_mode: bool):
    """Generate a random human-AI conversation example and reset judge outputs."""
    if compatible_mode:
        human_msg, ai_msg, ground_truth_msg = get_random_human_ai_ground_truth_pair()
    else:
        human_msg, ai_msg = get_random_human_ai_pair()
        ground_truth_msg = ""
    
    return [
        gr.update(value=human_msg),
        gr.update(value=ai_msg),
        gr.update(value="🎲", variant="secondary"),
        gr.update(value=""),  # Clear score
        gr.update(value=""),  # Clear critique
        gr.update(value=ground_truth_msg, visible=compatible_mode),  # Set ground truth and visibility
    ]


def create_arena_interface():
    with gr.Blocks(theme="default", css=CSS_STYLES) as interface:
        # Hidden eval prompt that will always contain DEFAULT_EVAL_PROMPT
        eval_prompt = gr.Textbox(
            value=DEFAULT_EVAL_PROMPT,
            visible=False
        )
        with gr.Row():
            # Left side - Input section
            with gr.Column(scale=1):
                with gr.Group():
                    human_input = gr.TextArea(
                        label="👩 User Input",
                        lines=5,
                        placeholder="Enter the human message here..."
                    )
                    with gr.Row():
                        generate_btn = gr.Button(
                            "Generate AI Response",
                            size="sm",
                            interactive=False
                        )
                    
                    ai_response = gr.TextArea(
                        label="🤖 AI Response", 
                        lines=10,
                        placeholder="Enter the AI response here..."
                    )
                    
                    # Ground truth response (initially hidden)
                    ground_truth = gr.TextArea(
                        label="🎯 Ground truth response",
                        lines=10,
                        placeholder="Enter the ground truth response here...",
                        visible=False
                    )
                    
                with gr.Row():
                    random_btn = gr.Button("🎲", scale=2)
                    send_btn = gr.Button(
                        value="Run evaluation",
                        variant="primary",
                        size="lg",
                        scale=8
                    )

            # Right side - Model outputs
            with gr.Column(scale=1):
                gr.Markdown("## 👩‍⚖️ Selene-Mini Evaluation")
                with gr.Group():
                    with gr.Row():
                        score = gr.Textbox(label="Score", lines=1, interactive=False)
                    critique = gr.TextArea(label="Critique", lines=12, interactive=False)
        
        gr.Markdown("<br>")
        

        # Replace the "Edit Judge Prompt" Accordion section with:
        with gr.Accordion("📝 Edit Judge Prompt", open=False) as prompt_accordion:
            gr.Markdown("<br>")
            use_reference_toggle = gr.Checkbox(
                label="Use a reference response",
                value=False
            )
            
            # Hide the default prompt editor
            with gr.Column(visible=False) as default_prompt_editor:
                eval_prompt_editable = gr.TextArea(
                    value=DEFAULT_EVAL_PROMPT_EDITABLE,
                    label="Evaluation Criteria",
                    lines=12
                )

                with gr.Row(visible=False) as edit_buttons_row:
                    cancel_prompt_btn = gr.Button("Cancel")
                    save_prompt_btn = gr.Button("Save", variant="primary")
            
            # Show the compatible mode editor
            with gr.Column(visible=True) as compatible_prompt_editor:
                eval_criteria_text = gr.TextArea(
                    label="Evaluation Criteria",
                    lines=12,
                    value=DEFAULT_EVAL_CRITERIA,
                    placeholder="Enter the complete evaluation criteria and scoring rubric..."
                )
                with gr.Row(visible=False) as compatible_edit_buttons_row:
                    compatible_cancel_btn = gr.Button("Cancel")
                    compatible_save_btn = gr.Button("Save", variant="primary")
            
            gr.Markdown("*The sample being evaluated is always appended as:*")
            gr.Markdown(f"```{FIXED_EVAL_SUFFIX}")

        eval_prompt_previous = gr.State(value=DEFAULT_EVAL_PROMPT_EDITABLE)  # Initialize with default value
        is_editing = gr.State(False)  # Track editing state
        compatible_mode_state = gr.State(False)  # Track compatible mode state

        # Update model names after responses are generated
        def update_model_names(model_a, model_b):
            return gr.update(value=f"*Model: {model_a}*"), gr.update(
                value=f"*Model: {model_b}*"
            )

        # Store the last submitted prompt and variables for comparison
        last_submission = gr.State({})

        # Update the save/cancel buttons section in the compatible prompt editor
        def save_criteria(new_criteria, previous_criteria):
            return [
                gr.update(value=new_criteria),  # Update the criteria
                new_criteria,  # Update the previous criteria state
                gr.update(visible=False)  # Hide the buttons
            ]

        def cancel_criteria(previous_criteria):
            return [
                gr.update(value=previous_criteria),  # Revert to previous criteria
                previous_criteria,  # Keep the previous criteria state
                gr.update(visible=False)  # Hide the buttons
            ]

        def show_criteria_edit_buttons(current_value, previous_value):
            # Show buttons only if the current value differs from the previous value
            return gr.update(visible=current_value != previous_value)

        # Add handlers for save/cancel buttons and criteria changes
        compatible_save_btn.click(
            fn=save_criteria,
            inputs=[eval_criteria_text, eval_prompt_previous],
            outputs=[eval_criteria_text, eval_prompt_previous, compatible_edit_buttons_row]
        )

        compatible_cancel_btn.click(
            fn=cancel_criteria,
            inputs=[eval_prompt_previous],
            outputs=[eval_criteria_text, eval_prompt_previous, compatible_edit_buttons_row]
        )

        eval_criteria_text.change(
            fn=show_criteria_edit_buttons,
            inputs=[eval_criteria_text, eval_prompt_previous],
            outputs=compatible_edit_buttons_row
        )

        # Function to toggle visibility based on compatible mode
        def toggle_use_reference(checked):
            if checked:
                human_msg, ai_msg, ground_truth_msg = get_random_human_ai_ground_truth_pair()
                return {
                    ground_truth: gr.update(visible=True, value=ground_truth_msg),
                    human_input: gr.update(value=human_msg),
                    ai_response: gr.update(value=ai_msg),
                    score: gr.update(value=""),
                    critique: gr.update(value=""),
                    random_btn: gr.update(value="🎲", variant="secondary"),
                }
            else:
                return {
                    ground_truth: gr.update(visible=False)
                }

        # Update the change handler to include all necessary outputs
        use_reference_toggle.change(
            fn=toggle_use_reference,
            inputs=[use_reference_toggle],
            outputs=[
                ground_truth,
                human_input,
                ai_response,
                score,
                critique,
                random_btn,
            ]
        )

        # Add a new state variable to track first game
        first_game_state = gr.State(True)  # Initialize as True

        # Update the submit function to parse the evaluation criteria
        def submit_and_store(
            use_reference,
            eval_criteria_text,
            human_input,
            ai_response,
            ground_truth_input,
        ):
            # Build prompt data dictionary
            prompt_data = {
                'human_input': human_input,
                'ai_response': ai_response,
                'ground_truth_input': ground_truth_input if use_reference else None,
                'eval_criteria': eval_criteria_text,
            }

            # Get response from Atla
            response = get_atla_response(
                model_name="AtlaAI/Selene-1-Mini-Llama-3.1-8B",
                prompt=prompt_data,
                max_tokens=500,
                temperature=0.01
            )

            # Response now contains score and critique directly
            if isinstance(response, dict) and 'score' in response and 'critique' in response:
                score = str(response['score'])
                critique = response['critique']
            else:
                # Handle error case
                score = "Error"
                critique = str(response)

            return [
                score,
                critique,
                gr.update(value="Regenerate evaluation", variant="secondary", interactive=True),
                gr.update(value="🎲"),
            ]

        # Update the click handler to use False for is_first_game after first submission
        def create_submit_handler():
            first_game = True
            
            def handler(*args):
                nonlocal first_game
                result = submit_and_store(*args)
                first_game = False  # Set to False after first submission
                return result
            
            return handler

        # Update the send_btn click handler
        send_btn.click(
            fn=submit_and_store,
            inputs=[
                use_reference_toggle,
                eval_criteria_text,
                human_input,
                ai_response,
                ground_truth,
            ],
            outputs=[
                score,
                critique,
                send_btn,
                random_btn,
            ],
        )

        # Add random button handler
        random_btn.click(
            fn=populate_random_example,
            inputs=[use_reference_toggle],
            outputs=[
                human_input,
                ai_response,
                random_btn,
                score,
                critique,
                ground_truth,
            ]
        )

        # Add input change handlers
        def handle_input_change():
            """Reset UI state when inputs are changed"""
            return [
                gr.update(value="Run evaluation", variant="primary"),  # send_btn
                gr.update(value="🎲", variant="secondary"),  # random_btn
            ]

        # Update the change handlers for inputs
        human_input.change(
            fn=handle_input_change,
            inputs=[],
            outputs=[send_btn, random_btn]
        )

        ai_response.change(
            fn=handle_input_change,
            inputs=[],
            outputs=[send_btn, random_btn]
        )

        generate_btn.click(
            fn=lambda msg: (
                generate_ai_response(msg)[0],  # Only take the response text
                gr.update(
                    value="Generate AI Response",  # Keep the label
                    interactive=False  # Disable the button
                )
            ),
            inputs=[human_input],
            outputs=[ai_response, generate_btn]
        )

        human_input.change(
            fn=lambda x: gr.update(interactive=bool(x.strip())),
            inputs=[human_input],
            outputs=[generate_btn]
        )

        # Update the demo.load to include the random example population
        interface.load(
            fn=lambda: populate_random_example(None, False),  # Pass False for initial compatible_mode
            inputs=[],
            outputs=[
                human_input,
                ai_response,
                random_btn,
                score,
                critique,
                ground_truth,
            ]
        )

    return interface

if __name__ == "__main__":
    demo = create_arena_interface()
    demo.launch()