
import gradio as gr
from utils import *
from species_list import SPECIES_LIST

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

class PromptLogger():
    """
    A logger class for handling the annotation process of bioacoustic data.

    Attributes:
    - tgt_files: List of target files available for annotation.
    - current_file: Currently selected file for annotation.
    - img_segs: List of image segment paths.
    - aud_segs: List of audio segment paths.
    - ann_file: File path for saving annotations.
    - annotations: File handle for the annotation file.
    """
    def __init__(self):
        """
        Initializes the annotation logger with a counter and segment path.

        Args:
        - initial_counter (int): Initial value of the file counter. Defaults to 0.
        - seg_path (str): Path to the directory where segments are stored. Defaults to "./temp/segs".
        """
        self.tgt_files = None
        self.current_file = None 

        self.prompt_path = None 
        self.img_segs = []
        self.aud_segs = []
        self.ann_file = None
        self.annotations = None

    def load_data(self, data_root_path):
        """
        Loads the data from the specified root path and prepares it for annotation.

        Args:
        - data_root_path (str): The root path where the data (audio files) is located.

        Returns:
        - List containing Gradio components to display available files and annotation path.
        """
        # Load the files from the directory
        tgt_files = glob(os.path.join(data_root_path, "**/*.wav"), recursive=True)
        self.tgt_files = tgt_files
        # Display the available files and annotation path.
        return [gr.Text("\n".join(self.tgt_files), lines=5, label="Available files:"),
                gr.Column(visible=True),
                gr.Text("{}/annotations".format(data_root_path), label="Annotation file will be saved to:", info="Please change the directory here if necessary!")]

    def load_sample_data(self, sample_num, seed=0):
        random.seed(int(seed))
        self.sample_files = random.sample(self.tgt_files, int(sample_num))
        return [gr.Text("\n".join(self.sample_files), lines=5, label="Available files:"),
                gr.Column(visible=True)]

    def register_prompt_file(self, prompt_path, name):
        """
        Registers an annotation file to save the annotations.

        Args:
        - ann_path (str): Path where the annotation file will be saved.
        - name (str): Name of the annotator.

        Returns:
        - List containing Gradio components to indicate successful registration.
        """
        # Set the annotation path and create the directory if it doesn't exist.
        self.prompt_path = prompt_path
        os.makedirs(self.prompt_path, exist_ok=True)
        self.ann_file = os.path.join(self.prompt_path, "prompt_{}.csv".format(name))
        self.prompts = {"positive": "", "negative": "", "sample_files": self.sample_files} 
        return [gr.Text("Name Registered as {}.".format(name), label="Registration Successful!"),
                gr.Button("Start prompting!", visible=True)]

    def start_annotation(self):
        """
        Prepares and displays the Gradio interface for starting the annotation of the current audio file.

        Returns:
        - A list of Gradio components that constitute the interface for annotating the current audio file.
          This includes a column for the annotation workspace, an accordion for configuration details, 
          and the audio player for the current file.
        """
        return [gr.Column(visible=True),
                gr.Accordion("Loading configurations", open=False),
                gr.Text(self.current_file, visible=False),
                gr.Audio(self.current_file, label=self.current_file, visible=True)]
    
    def start_seg_annotation(self, current_file):
        """
        Begins annotation for a specific segment of the current audio file.

        Args:
        - current_file (str): The path to the current audio file being annotated.

        Returns:
        - A list of Gradio components for segment annotation including an image of the spectrogram 
          segment, the corresponding audio segment, and a dropdown for species selection.
        """
        # Open the annotation file for writing and set the segment counter to 0.
        try:
            self.annotations = open(self.ann_file, "a")
            self.seg_counter = 0
            # Retrieve the image and audio segments for the current file.
            self.img_segs = sorted(glob(os.path.join(self.seg_path, "*.png")))
            self.aud_segs = sorted(glob(os.path.join(self.seg_path, "*.wav")))
            # Load the first segment for annotation.
            img_seg = self.img_segs[self.seg_counter]
            aud_seg = self.aud_segs[self.seg_counter]
            # Return the Gradio components for segment annotation.
            return [gr.Image(img_seg, label="Spectrogram Seg:"),
                    gr.Audio(aud_seg, label="Audio Seg:"),
                    gr.Column(visible=True),
                    gr.Dropdown(SPECIES_LIST, label="Select a species:", value=None),
                    gr.Text("Please select a category in the dropdown menu.", label="Instruction:")]
        except:
            return [gr.Image(),
                    gr.Audio(),
                    gr.Column(visible=False),
                    gr.Dropdown(),
                    gr.Text("NO DETECTED SEGMENTS. PLEASE MOVE ON TO THE NEXT AUDIO.", label="INPORTANT INFO:")]

    def end_annotation(self):
        """
        Finalizes the annotation process for the current file.

        Returns:
        - A list of Gradio components indicating the completion of the annotation process 
          and the path where annotations are saved.
        """
        return [gr.Column(visible=False),
                gr.Column(visible=True),
                gr.Text(self.ann_file, 
                        label="Annotation saved to the following path, please close the annotation app.",
                        visible=True)]

    def next_audio_file(self):
        """
        Proceeds to the next audio file for annotation.

        Returns:
        - A list of Gradio components for the next audio file, or a message indicating no more files.
        """
        try:
            self.counter += 1
            self.current_file = self.tgt_files[self.counter]
            return [gr.Text(self.current_file, visible=False),
                    gr.Audio(self.current_file, label=self.current_file),
                    gr.Text("Click Detect button for predictions.",
                            label="Number of detectect positive sounds:"),
                    gr.Column(visible=False),
                    gr.Column(visible=True), gr.Column(visible=True),
                    gr.Button("Next Audio", visible=True), gr.Button(visible=False),
                    gr.Text("Please click the Detect button for possible call detections.", label="Instruction:", visible=False)]
        except:
            return [gr.Text("No more files to annotate!", visible=True, label="IMPORTANT INFO:"),
                    gr.Audio(visible=False),
                    gr.Text(visible=False),
                    gr.Column(visible=False), 
                    gr.Column(visible=False), 
                    gr.Column(visible=False),
                    gr.Button(visible=False),
                    gr.Button("Submit", visible=True),
                    gr.Text(visible=False)]

    def next_segment(self, category):
        """
        Proceeds to the next segment for annotation within the current audio file.

        Args:
        - category (str): The selected category for the current segment.

        Returns:
        - A list of Gradio components for the next segment, or a message indicating no more segments.
        """
        try:
            st, ed, conf = self.img_segs[self.seg_counter].replace(".png",'').split('_')[-3:]
            self.annotations.write("{},{},{},{},{}\n".format(self.current_file, st, ed, category, conf))
            self.seg_counter += 1
            img_seg = self.img_segs[self.seg_counter]
            aud_seg = self.aud_segs[self.seg_counter]
            return [gr.Image(img_seg, label="Spectrogram Seg:"),
                    gr.Audio(aud_seg, label="Audio Seg:"),
                    gr.Dropdown(SPECIES_LIST, label="Select a species:", value=None),
                    gr.Column(visible=True),
                    gr.Text("Please select a category in the dropdown menu.", label="Instruction:")]
        except:
            self.seg_counter = 0
            # self.current_file = None
            self.annotations.close()
            return [gr.Image(),
                    gr.Audio(),
                    gr.Dropdown(),
                    gr.Column(visible=False),
                    gr.Text("NO MORE DETECTED SEGMENTS. PLEASE MOVE ON TO THE NEXT AUDIO.", label="INPORTANT INFO:")]
    
class PromptTest(gr.Blocks):
    def __init__(self, prompt_logger):
        super().__init__()
        self.prompt_logger = prompt_logger
        self.build_blocks()
    
    def build_blocks(self):
        with self:
            with gr.Accordion("Configurations", open=True) as load_acc:

                data_root_path = gr.Text("./demo_data", label="Please type in the directory of the dataset root:", interactive=True)
                data_fetch_but = gr.Button("Get data from the root directory.")

                # Gradio components to show the available files and the path to save the annotation file

                with gr.Column(visible=False) as data_config_col:
                    with gr.Accordion("Open to see all available files.", open=False):
                        data_file_list = gr.Text("", lines=5, label="Available files:")

                    prompt_path = gr.Text("", label="Initial prompt file will be saved to:", info="Please change the directory here if necessary!")

                    with gr.Row():
                        num_file = gr.Text("3", label="Number of randomly selected file for testing prompts (Max 10 files):", interactive=True)
                        random_seed = gr.Text("0", label="Random seed:", interactive=True)

                    random_data_fetch_but = gr.Button("Get sample data for prompting.")

                with gr.Column(visible=False) as model_loading_col:
                    sample_file_list = gr.Text("", lines=5, label="Sampled files for prompting:")
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

            # %% # Annotation buttons and actions
            # Load the data
            data_fetch_but.click(self.prompt_logger.load_data, 
                                 inputs=data_root_path, 
                                 outputs=[data_file_list, data_config_col, prompt_path])

            # Get sample data for prompting
            random_data_fetch_but.click(self.prompt_logger.load_sample_data, 
                                         inputs=[num_file, random_seed], 
                                         outputs=[sample_file_list, model_loading_col])
            # Load the model
            load_but.click(load_models, 
                           inputs=det_drop, 
                           outputs=[load_out, register_col])
            # Register the name
            prompt_name_reg_but.click(self.prompt_logger.register_prompt_file, 
                                   inputs=[prompt_path, prompt_name], 
                                   outputs=[prompt_name, start_but])
            # Start the annotation
            # start_but.click(self.prompt_logger.start_prompting, 
            #                 outputs=[ann_col, load_acc, cur_file_path, cur_file])

    
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