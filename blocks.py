
import gradio as gr
from utils import *
from species_list import SPECIES_LIST

MAX_OUTPUT_ROWS = 100

class Welcome(gr.Blocks):
    def __init__(self):
        super().__init__()
        self.build_blocks()

    def build_blocks(self):
        with self:
            gr.Markdown("# Bioacoustics Annotation Tool Powered by CLAP!")
            gr.Markdown("## Please choose a tab to start!")
    
class Annotation(gr.Blocks):

    def __init__(self, ann_logger):
        super().__init__()
        self.ann_logger = ann_logger
        self.build_blocks()
    
    def build_blocks(self):
        with self:
            # Annotation tab config
            with gr.Accordion("Configurations", open=True) as load_acc:

                ann_root_path = gr.Text("./demo_data", label="Please type in the directory of the dataset root:", interactive=True)
                ann_data_fetch_but = gr.Button("Get data from the root directory.")
                # Gradio components to show the available files and the path to save the annotation file
                with gr.Column(visible=False) as ann_model_loading_col:
                    with gr.Accordion("Open to see all available files.", open=False):
                        ann_file_list = gr.Text("", lines=5, label="Available files:")
                    ann_path = gr.Text("", label="Annotation file will be saved to:", info="Please change the directory here if necessary!")
                    # Model selection
                    det_drop = gr.Dropdown(
                        ["CLAP_Jan23"],
                        label="Detection model",
                        info="Will add more detection models in the future!",
                        value="CLAP_Jan23"
                    )
                    load_but = gr.Button("Click to load CLAP")
                    load_out = gr.Text("CLAP is not loaded yet!!", label="Loaded CLAP model:")
                # Name registering
                with gr.Column(visible=False) as register_col:
                    with gr.Row():
                        ann_name = gr.Textbox(lines=1, label="Please put your name here and register before annotation:", interactive=True)
                        ann_name_reg_but = gr.Button("Register Your Name", scale=0.5)
                    start_but = gr.Button("Start annotations!", visible=False)

            # Starting the annotation task
            with gr.Column(visible=False) as ann_col:
                # Audio file path
                cur_file_path = gr.Text(visible=False)
                cur_file = gr.Audio()

                # Positive and negative sample selection
                with gr.Column(visible=True) as det_config_col:
                    with gr.Row():
                        det_pos_prompt = gr.Text("birds chirping;", label="Positive Prompts:", interactive=True)
                        det_neg_prompt = gr.Text("noise;", label="Negative Prompts:", interactive=True)
                    # Detection confidence threshold
                    det_conf = gr.Slider(0, 1, label="Detection Confidence Threshold", value=0.5)
                    det_but = gr.Button("Detect!")
                # Detection output
                with gr.Column(visible=True) as det_output_col:
                    det_txt_summ = gr.Text("Click Detect button for predictions.",
                                           label="Number of detectect positive sounds:")
                    with gr.Column(visible=False) as det_output_col_2:
                        det_out = gr.Image(label="Detection Predictions:")
                        ann_but = gr.Button("Annotate!")
                    # Start annotation
                    with gr.Column():
                        set_info = gr.Text("Please click the Annotate button for segment annotations.",
                                           label="Instruction:", visible=False)
                        with gr.Column(visible=False) as ann_seg_col:
                            with gr.Row():
                                spec_seg = gr.Image(label="Spectrogram Seg:", height=200)
                                aud_seg = gr.Audio(label="Audio Seg:")
                            ann_drop = gr.Dropdown(SPECIES_LIST, label="Select a species:")
                            # Go to next segment if available
                            seg_next_but = gr.Button("Next Segment")
                # Go to next audio file
                with gr.Column():
                    next_but = gr.Button(value="Next Audio", visible=True)
                    submit_but = gr.Button("Submit", visible=False)

            # Annotation end text
            with gr.Column(visible=False) as end_col:
                end_text = gr.Text(label="Annotation saved to the following path, please close the annotation app.")

            # %% # Annotation buttons and actions
            # Load the data
            ann_data_fetch_but.click(self.ann_logger.load_data, 
                                     inputs=ann_root_path, 
                                     outputs=[ann_file_list, ann_model_loading_col, ann_path])
            # Load the model
            load_but.click(load_models, 
                           inputs=det_drop, 
                           outputs=[load_out, register_col])
            # Register the name
            ann_name_reg_but.click(self.ann_logger.register_ann_file, 
                                   inputs=[ann_path, ann_name], 
                                   outputs=[ann_name, start_but])
            # Start the annotation
            start_but.click(self.ann_logger.start_annotation, 
                            outputs=[ann_col, load_acc, cur_file_path, cur_file])
            # Detection button
            det_but.click(single_audio_detection, 
                          inputs=[cur_file_path, det_neg_prompt, det_pos_prompt, det_conf],
                          outputs=[det_txt_summ, det_out, det_output_col_2, set_info])
            # Annotation button
            ann_but.click(self.ann_logger.start_seg_annotation, 
                        #   inputs=cur_file_path,
                          outputs=[spec_seg, aud_seg, ann_seg_col, ann_drop, set_info])
            # Next segment button
            seg_next_but.click(self.ann_logger.next_segment, 
                               inputs=ann_drop,
                               outputs=[spec_seg, aud_seg, ann_drop, ann_seg_col, set_info])
            # Next audio button
            next_but.click(self.ann_logger.next_audio_file, 
                           outputs=[cur_file_path, cur_file, det_txt_summ, det_output_col_2,
                                    det_config_col, det_output_col, next_but, submit_but, set_info])
            # Submit button
            submit_but.click(self.ann_logger.end_annotation, 
                             outputs=[ann_col, end_col, end_text])


class Validation(gr.Blocks):

    def __init__(self, val_logger):
        super().__init__()
        self.val_logger = val_logger
        self.build_blocks()

    def build_blocks(self):
        with self:
            with gr.Accordion("Configurations", open=True) as load_acc_val:
                # Dataset to load the annotation
                val_root_path = gr.Text("./demo_data", label="Please type in the directory of the dataset root:", interactive=True)
                val_data_fetch_but = gr.Button("Get data from the root directory.")
                # Gradio components to show the available files and the path to save the annotation file
                with gr.Column(visible=False) as val_model_loading_col:
                    with gr.Accordion("Open to see all available files.", open=False):
                        val_file_list = gr.Text("", lines=5, label="Available files:")
                    ann_path_val = gr.Text("", label="Annotation path:", info="Please change the directory here if necessary!")
                    get_ann_val_but = gr.Button("Get Annotation Files")
                # Name registering
                with gr.Column(visible=False) as register_col_val:
                    ann_val_drop = gr.Dropdown(choices=[], label="Select an annotation file to validate:") 
                    with gr.Row():
                        ann_name_val = gr.Textbox(lines=1, label="Please put your name here and register before annotation:", interactive=True)
                        ann_name_reg_but_val = gr.Button("Register Your Name", scale=0.5)

                    start_val_but = gr.Button("Start Validation!", visible=False)
            # Starting the validation task
            with gr.Column(visible=False) as val_col:
                cur_file_path_val = gr.Text(visible=False)
                # Show the audio and spectrogram
                with gr.Column(visible=True) as val_col_2:
                    cur_aud_val = gr.Audio()
                    cur_spec_val = gr.Image(label="Detection Predictions:")
                    val_txt_summ = gr.Text(label="Number of annotated segments:")
                    # Load annotations from file
                    get_ann_output_val = gr.Button("Get Annotations")
                # Change label
                set_info_val = gr.Text("Please use the dropdown menu to change the catgegory.",
                                       label="Instruction:", visible=False)

                with gr.Column(visible=False) as val_seg_col:
                    with gr.Row():
                        spec_seg_val = gr.Image(label="Spectrogram Seg:", height=200)
                        aud_seg_val = gr.Audio(label="Audio Seg:")
                    val_drop = gr.Dropdown(SPECIES_LIST, label="Select a species:")
                    seg_next_but_val = gr.Button("Next Segment")
                # Go to next audio file
                with gr.Column():
                    next_but_val = gr.Button(value="Next Audio", visible=True)
                    submit_but_val = gr.Button("Submit", visible=False)

            # Load the data 
            val_data_fetch_but.click(self.val_logger.load_data, 
                                     inputs=val_root_path, 
                                     outputs=[val_file_list, val_model_loading_col, ann_path_val])
            # Register the name
            ann_name_reg_but_val.click(self.val_logger.register_val_file,
                                       inputs=ann_name_val,
                                       outputs=[ann_name_val, start_val_but])
            # Load the annotation
            get_ann_val_but.click(self.val_logger.register_ann_path,
                                  inputs=ann_path_val,
                                  outputs=[ann_val_drop, register_col_val])
            # Start the validation
            start_val_but.click(self.val_logger.start_validation,
                                inputs=ann_val_drop,
                                outputs=[load_acc_val, val_col, cur_file_path_val, cur_spec_val,
                                         cur_aud_val, val_txt_summ])
            # Get the annotation
            get_ann_output_val.click(self.val_logger.start_segment_validation, 
                                     outputs=[spec_seg_val, aud_seg_val, val_drop, set_info_val, val_seg_col])
            # Next segment button
            seg_next_but_val.click(self.val_logger.next_segment, 
                                   inputs=val_drop, 
                                   outputs=[spec_seg_val, aud_seg_val, val_drop, val_seg_col, set_info_val])
            # Next audio button
            next_but_val.click(self.val_logger.next_audio, 
                               outputs=[val_col_2, set_info_val, cur_file_path_val,
                                        cur_spec_val, cur_aud_val, val_txt_summ,
                                        next_but_val, submit_but_val])
            # Submit button
            submit_but_val.click(self.val_logger.end_validation, 
                                 outputs=[val_col_2, val_seg_col, set_info_val])