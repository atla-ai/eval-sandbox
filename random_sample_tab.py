import gradio as gr
from random_sample.arena_interface import create_arena_interface

def random_sample_tab():
    with gr.TabItem("Random samples"):
        return create_arena_interface() 