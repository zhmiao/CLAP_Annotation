import gradio as gr
import time
def test():
    progress = gr.Progress()
    progress(0, desc="Starting...")
    time.sleep(1)
    for i in progress.tqdm(range(100)):
        time.sleep(0.1)

def my_function(x):
    test()
    return x
gr.Interface(my_function, gr.Textbox(), gr.Textbox()).queue().launch()