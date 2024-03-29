import gradio as gr

from utils import AnnLogger, ValLogger
from blocks import * 

prompt_logger = PromptLogger()
ann_logger = AnnLogger()
val_logger = ValLogger()
wel_block = Welcome()
# prompt_block = PromptTest(prompt_logger, max_sample=prompt_logger.max_sample)
prompt_block = PromptTest(prompt_logger)
ann_block = Annotation(ann_logger)
val_block = Validation(val_logger)

demo = gr.TabbedInterface([wel_block, prompt_block, ann_block, val_block],
                          tab_names=["Welcome", "Prompt Testing", "Annotation", "Validation"])

demo.launch()
