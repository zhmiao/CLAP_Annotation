import gradio as gr
from app.utils import *
from app.species_list import SPECIES_LIST

import random

MAX_OUTPUT_ROWS = 100

class Welcome(gr.Blocks):
    def __init__(self):
        super().__init__()
        self.build_blocks()

    def build_blocks(self):
        with self:
            gr.Markdown("# Bioacoustics Annotation Tool Powered by CLAP!")
            gr.Markdown("## Please choose a tab to start!")


class PromptTest(gr.Blocks):
    def __init__(self, prompt_logger, max_sample=10):
        super().__init__()
        self.prompt_logger = prompt_logger
        self.max_sample = max_sample
        self.build_blocks()
    
    def build_blocks(self):
        with self:
            with gr.Column(visible=True) as conf_col:
                with gr.Accordion("Configurations", open=True) as load_acc:

                    prompt_data_root_path = gr.Text(os.path.join(".", "demo_data"), label="Please type in the directory of where images for prompting are stored:",
                                                    interactive=True)
                    file_extension = gr.Text("WAV", label="File extension for the data.", info="Please change accordingly", interactive=True)
                    data_fetch_but = gr.Button("Get data from the root directory.")

                    # Gradio components to show the available files and the path to save the annotation file

                    with gr.Column(visible=False) as data_config_col:
                        with gr.Accordion("Open to see all available files.", open=False):
                            data_file_list = gr.Text("", lines=5, label="Available files:")
                        with gr.Row():
                            num_file = gr.Text("2", label="Number of randomly selected file for testing prompts (Max 10 files):", interactive=True)
                            random_seed = gr.Text("0", label="Random seed:", interactive=True)

                        sample_file_list = gr.Text("Click the button to get sample files.", lines=1, label="Sampled files for prompting:")

                        random_data_fetch_but = gr.Button("Get sample data for prompting.")

                    with gr.Column(visible=False) as model_loading_col:
                        # Model selection
                        det_drop = gr.Dropdown(
                            ["CLAP_Jan23"],
                            label="Select an audio-language model",
                            info="Will add more detection models in the future!",
                            value="CLAP_Jan23"
                        )
                        load_but = gr.Button("Click to load CLAP")
                        load_out = gr.Text("CLAP is not loaded yet!!", label="Loaded CLAP model:")

                    with gr.Column(visible=False) as register_col:
                        with gr.Row():
                            prompt_name = gr.Textbox(lines=1, label="Please put your name here and register before prompting:",
                                                     interactive=True)
                            prompt_name_reg_but = gr.Button("Register Your Name", scale=0.5)
                        start_but = gr.Button("Start prompting!", visible=False)

            with gr.Column(visible=False) as prompt_col:
                with gr.Accordion("Prompting results.", open=True) as prompt_acc:
                    # Audio file path
                    with gr.Row():
                        with gr.Column():
                            spec_list = []
                            for _ in range(self.max_sample):
                                spec_list.append(gr.Image(visible=False))
                        with gr.Column():
                            audio_list = []
                            for _ in range(self.max_sample):
                                audio_list.append(gr.Audio(visible=False))
                        sample_file_list.change(self.prompt_logger.populate_audio, sample_file_list, audio_list)
                        sample_file_list.change(self.prompt_logger.populate_image, sample_file_list, spec_list)

                    # Positive and negative sample selectio
                    det_pos_prompt = gr.Text("birds chirping;", label="Positive Prompts:", interactive=True)
                    det_neg_prompt = gr.Text("noise;", label="Negative Prompts:", interactive=True)
                    # Detection confidence threshold
                    det_conf = gr.Slider(0, 1, label="Detection Confidence Threshold", value=0.7)

                    det_but = gr.Button("Detect!")

                    submit_but = gr.Button("Submit prompts!")
                
            with gr.Column(visible=False) as prompt_final_col:

                data_path = gr.Text("", label="Data directory for batch detection:", info="Please change the directory here if necessary!")
                batch_extension = gr.Text("", label="File extension for batch detection", info="Please change if necessary!")
                data_confirm_but = gr.Button("Confirm data root.")

                with gr.Column(visible=False) as prompt_final_col_2:
                    with gr.Accordion("", open=False) as batch_data_acc:
                        batch_data_file_list = gr.Text("", lines=5, label="Available files for batch detection:")
                    prompt_path = gr.Text("", label="Prompt information file will be saved to:", info="Please change the directory here if necessary!")
                    ann_path = gr.Text("", label="Batch detection file will be saved to:", info="Please change the directory here if necessary!")
                    prompt_sess_id = gr.Text("0", label="Prompt testing session id for reference:", info="Please change the session number here if necessary!")
                    command_text = gr.Text("Click Generate Detection Command button to generate batch detection commands.")
                    command_but = gr.Button("Generate Detection Command!")

            with gr.Column(visible=False) as prompt_finish_col:
                finish_text = gr.Text("", visible=False)
                finish_but = gr.Button("Finish!")
                batch_det_but = gr.Button(visible=False)

            # %% # Annotation buttons and actions
            # Load the data
            data_fetch_but.click(self.prompt_logger.load_data, 
                                 inputs=[prompt_data_root_path, file_extension], 
                                 outputs=[data_file_list, data_config_col, prompt_path])
            # Get sample data for prompting
            random_data_fetch_but.click(self.prompt_logger.load_sample_data, 
                                         inputs=[num_file, random_seed], 
                                         outputs=[sample_file_list, model_loading_col])
            # Load the model
            load_but.click(self.prompt_logger.load_models, 
                           inputs=det_drop, 
                           outputs=[load_out, register_col])
            # Register the name
            prompt_name_reg_but.click(self.prompt_logger.register_prompt_file, 
                                   inputs=[prompt_path, prompt_name], 
                                   outputs=[prompt_name, start_but])
            # Start prompting
            start_but.click(self.prompt_logger.start_prompting, 
                            outputs=[prompt_col, load_acc])

            det_but.click(self.prompt_logger.update_image, 
                          inputs=[sample_file_list, det_neg_prompt, det_pos_prompt, det_conf],
                          outputs=spec_list)

            submit_but.click(self.prompt_logger.submission,
                             inputs=prompt_data_root_path,
                             outputs=[prompt_acc, prompt_final_col, data_path, batch_extension])

            data_confirm_but.click(self.prompt_logger.path_confirm,
                                   inputs=[data_path, batch_extension],
                                   outputs=[prompt_final_col_2, prompt_finish_col, batch_data_acc, batch_data_file_list,
                                            prompt_path, ann_path])

            command_but.click(self.prompt_logger.command_gen,
                              inputs=[data_path, batch_extension, ann_path, prompt_path, prompt_sess_id, random_seed],
                              outputs=command_text)

            finish_but.click(self.prompt_logger.finish,
                             inputs=[command_text, det_neg_prompt, det_pos_prompt, det_conf],
                             outputs=[conf_col, prompt_col, prompt_final_col, finish_text, batch_det_but, finish_but])

            batch_det_but.click(self.prompt_logger.batch_detection,
                                inputs=[det_neg_prompt, det_pos_prompt, det_conf,
                                        ann_path, batch_extension, prompt_sess_id, random_seed],
                                outputs=finish_text)

    
class Annotation(gr.Blocks):

    def __init__(self, ann_logger):
        super().__init__()
        self.ann_logger = ann_logger
        self.build_blocks()
    
    def build_blocks(self):
        with self:
            # Annotation tab config
            with gr.Accordion("Configurations", open=True) as load_acc:

                root_path = gr.Text(os.path.join(".", "demo_data"), label="Please type in the directory of the dataset root:", interactive=True)
                file_extension = gr.Text("WAV", label="File extension for the data.", info="Please change accordingly", interactive=True)
                data_fetch_but = gr.Button("Get data from the root directory.")

                # Gradio components to show the available files and the path to save the annotation file
                with gr.Column(visible=False) as path_col:
                    ann_path = gr.Text("", label="Annotation file will be saved to:", info="Please change the directory here if necessary!")
                    det_path = gr.Text("", label="Directory where batch detection results are saved to", info="Please change the directory here if necessary!")
                    get_det_file_but = gr.Button("Get Detection Results Files")

                # Name registering
                with gr.Column(visible=False) as register_col:
                    det_drop = gr.Dropdown(choices=[], label="Select a batch detection result file to annotate:") 
                    get_ann_file_but = gr.Button("Load all files with detected segments.")
                    with gr.Accordion("", visible=False, open=False) as file_acc:
                        file_list = gr.Text("", lines=5, label="Available files:")
                    with gr.Row(visible=False) as name_row:
                        ann_name = gr.Textbox(lines=1, label="Please put your name here and register before annotation:", interactive=True)
                        ann_name_reg_but = gr.Button("Register Your Name", scale=0.5)
                    start_but = gr.Button("Start annotations!", visible=False)

            # Starting the annotation task
            with gr.Column(visible=False) as ann_col:
                cur_file_path = gr.Text(visible=False)
                # Show the audio and spectrogram
                with gr.Column(visible=True) as ann_col_2:
                    cur_aud = gr.Audio()
                    cur_spec = gr.Image(label="Detection predictions:")
                    txt_summ = gr.Text(label="Number of detected segments:")
                    # Load annotations from file
                    get_det_output = gr.Button("Get Detections")
                # Change label
                set_info = gr.Text("Click the Get Detection button to get detected segments.",
                                   label="Instruction:", visible=True)

                with gr.Column(visible=False) as ann_seg_col:
                    with gr.Row():
                        with gr.Column():
                            spec_seg = gr.Image(label="Spectrogram Seg:", height=200)
                        with gr.Column():
                            aud_seg = gr.Audio(label="Audio Seg:")
                    ann_drop = gr.Dropdown(SPECIES_LIST, label="Select a species:", value=None)
                    seg_next_but = gr.Button("Next Segment")
                # Go to next audio file
                with gr.Column():
                    next_but = gr.Button(value="Next Audio", visible=True)
                    submit_but = gr.Button("Submit", visible=False)

            # %% # Annotation buttons and actions
            # Load the data
            data_fetch_but.click(self.ann_logger.load_data, 
                                 inputs=[root_path, file_extension], 
                                 outputs=[path_col, ann_path, det_path])
            
            # Load the annotation
            get_det_file_but.click(self.ann_logger.register_ann_path,
                                  inputs=[ann_path, det_path],
                                  outputs=[det_drop, register_col])

            get_ann_file_but.click(self.ann_logger.load_detected_files,
                                   inputs=det_drop,
                                   outputs=[file_acc, file_list, name_row])
            # Register the name
            ann_name_reg_but.click(self.ann_logger.register_ann_file, 
                                   inputs=ann_name, 
                                   outputs=[ann_name, start_but])
            # Start the validation
            start_but.click(self.ann_logger.start_annotation,
                            inputs=None,
                            outputs=[load_acc, ann_col, cur_file_path, cur_spec,
                                     cur_aud, txt_summ])
            # Get the annotation
            get_det_output.click(self.ann_logger.start_segment_annotation, 
                                 outputs=[spec_seg, aud_seg, set_info, ann_seg_col])
            # Next segment button
            seg_next_but.click(self.ann_logger.next_segment, 
                               inputs=ann_drop, 
                               outputs=[spec_seg, aud_seg, ann_seg_col, set_info, ann_drop])
            # Next audio button
            next_but.click(self.ann_logger.next_audio, 
                           outputs=[ann_col_2, set_info, cur_file_path,
                                    cur_spec, cur_aud, txt_summ,
                                    next_but, submit_but, ann_drop])
            # Submit button
            submit_but.click(self.ann_logger.end_annotation, 
                             outputs=[ann_col_2, ann_seg_col, set_info])


class Validation(gr.Blocks):

    def __init__(self, val_logger):
        super().__init__(css=".seg_images {height: 2px;}")
        self.val_logger = val_logger
        self.build_blocks()

    def build_blocks(self):
        with self:
            with gr.Accordion("Configurations", open=True) as load_acc:
                # Dataset to load the annotation
                root_path = gr.Text(os.path.join(".", "demo_data"), label="Please type in the directory of the dataset root:", interactive=True)
                file_extension = gr.Text("WAV", label="File extension for the data.", info="Please change accordingly", interactive=True)
                data_fetch_but = gr.Button("Get data from the root directory.")

                # Gradio components to show the available files and the path to save the annotation file
                with gr.Column(visible=False) as path_col:
                    ann_path = gr.Text("", label="Default path where annotation and validation files are saved:",
                                       info="Please change the directory here if necessary!")
                    get_ann_but = gr.Button("Get Annotation Files")

                # Name registering
                with gr.Column(visible=False) as register_col:
                    ann_drop = gr.Dropdown(choices=[], label="Select an annotation file to validate:") 
                    get_ann_file_but = gr.Button("Load all files with annotated segments.")
                    with gr.Accordion("", visible=False, open=False) as file_acc:
                        file_list = gr.Text("", lines=5, label="")
                    with gr.Row(visible=False) as name_row:
                        ann_name_val = gr.Textbox(lines=1, label="Please put your name here and register before annotation:", interactive=True)
                        ann_name_reg_but_val = gr.Button("Register Your Name", scale=0.5)

                    start_val_but = gr.Button("Start Validation!", visible=False)

            with gr.Accordion("Category selection", visible=False, open=False) as cat_sel_acc:
                cat_sel_text = gr.Markdown("")
                with gr.Column(visible=True) as fetch_col:
                    cat_drop = gr.Dropdown(choices=[])
                    start_cat_val_but = gr.Button("Fetch segments.")
                get_seg_spec_but = gr.Button("Next", visible=False)

            # Starting the validation task
            with gr.Column(visible=False) as val_col:
                # Change label
                seg_info_val = gr.Text("Please use the dropdown menu to change the catgegory.",
                                       label="Instruction:", visible=False)

                # Show the audio and spectrogram
                with gr.Column(visible=True) as val_seg_col:

                    with gr.Column(visible=False) as col_1:
                        seg_info_1 = gr.Markdown("")
                        with gr.Row():
                            with gr.Column():
                                spec_1 = gr.Image()
                            with gr.Column():
                                aud_1 = gr.Audio()
                            with gr.Column():
                                drop_1 = gr.Dropdown(SPECIES_LIST, value=None, label="Select a different species if needed.", min_width=50)
                        
                    with gr.Column(visible=False) as col_2:
                        seg_info_2 = gr.Markdown("")
                        with gr.Row():
                            with gr.Column():
                                spec_2 = gr.Image()
                            with gr.Column():
                                aud_2 = gr.Audio()
                            with gr.Column():
                                drop_2 = gr.Dropdown(SPECIES_LIST, value=None, label="Select a different species if needed.")

                    with gr.Column(visible=False) as col_3:
                        seg_info_3 = gr.Markdown("")
                        with gr.Row():
                            with gr.Column():
                                spec_3 = gr.Image()
                            with gr.Column():
                                aud_3 = gr.Audio()
                            with gr.Column():
                                drop_3 = gr.Dropdown(SPECIES_LIST, value=None, label="Select a different species if needed.")

                    with gr.Column(visible=False) as col_4:
                        seg_info_4 = gr.Markdown("")
                        with gr.Row():
                            with gr.Column():
                                spec_4 = gr.Image()
                            with gr.Column():
                                aud_4 = gr.Audio()
                            with gr.Column():
                                drop_4 = gr.Dropdown(SPECIES_LIST, value=None, label="Select a different species if needed.")

                    with gr.Column(visible=False) as col_5:
                        seg_info_5 = gr.Markdown("")
                        with gr.Row():
                            with gr.Column():
                                spec_5 = gr.Image()
                            with gr.Column():
                                aud_5 = gr.Audio()
                            with gr.Column():
                                drop_5 = gr.Dropdown(SPECIES_LIST, value=None, label="Select a different species if needed.")
                    
                    batch_next_but = gr.Button("Next Batch")

                    new_cat_but = gr.Button(value="Save and select a new category", visible=False)

            submit_but = gr.Button("Submit", visible=False)

            # Load the data 
            data_fetch_but.click(self.val_logger.load_data, 
                                 inputs=[root_path, file_extension], 
                                 outputs=[path_col, ann_path])
            # Load the annotation
            get_ann_but.click(self.val_logger.register_ann_path,
                              inputs=ann_path,
                              outputs=[ann_drop, register_col])

            get_ann_file_but.click(self.val_logger.load_ann_files,
                                   inputs=ann_drop,
                                   outputs=[file_acc, file_list, name_row])
            # Register the name
            ann_name_reg_but_val.click(self.val_logger.register_val_file,
                                       inputs=ann_name_val,
                                       outputs=[ann_name_val, start_val_but])
            # Start the validation
            start_val_but.click(self.val_logger.start_validation,
                                inputs=ann_drop,
                                outputs=[load_acc, cat_sel_acc, cat_drop, cat_sel_text])

            start_cat_val_but.click(self.val_logger.fetch_segments,
                                    inputs=cat_drop,
                                    outputs=[cat_sel_text, fetch_col, get_seg_spec_but])
            
            get_seg_spec_but.click(self.val_logger.populate_segments,
                                   inputs=None,
                                   outputs=[val_col, cat_sel_acc, val_seg_col, seg_info_val, batch_next_but, new_cat_but,
                                            col_1, spec_1, aud_1, drop_1, seg_info_1,
                                            col_2, spec_2, aud_2, drop_2, seg_info_2,
                                            col_3, spec_3, aud_3, drop_3, seg_info_3,
                                            col_4, spec_4, aud_4, drop_4, seg_info_4,
                                            col_5, spec_5, aud_5, drop_5, seg_info_5])

            batch_next_but.click(self.val_logger.batch_update,
                                 inputs=[drop_1,drop_2,drop_3,drop_4,drop_5],
                                 outputs=[val_col, cat_sel_acc, val_seg_col, seg_info_val, batch_next_but, new_cat_but,
                                          col_1, spec_1, aud_1, drop_1, seg_info_1,
                                          col_2, spec_2, aud_2, drop_2, seg_info_2,
                                          col_3, spec_3, aud_3, drop_3, seg_info_3,
                                          col_4, spec_4, aud_4, drop_4, seg_info_4,
                                          col_5, spec_5, aud_5, drop_5, seg_info_5])

            new_cat_but.click(self.val_logger.new_cat,
                              inputs=None,
                              outputs=[cat_sel_acc, fetch_col, val_col, cat_drop, cat_sel_text,
                                       get_seg_spec_but, submit_but])

            # Submit button
            submit_but.click(self.val_logger.end_validation, 
                             outputs=[val_col, cat_sel_acc, val_seg_col, seg_info_val])