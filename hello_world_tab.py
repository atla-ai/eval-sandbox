# hello_world_tab.py

import gradio as gr

def hello_world_tab():
    with gr.TabItem("Hello World"):
        gr.Markdown("## Hello, World!")
        gr.Textbox("This is the Hello World tab.", interactive=False)