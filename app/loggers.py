
# %%
import os
import shutil
import random
import json
import copy
from glob import glob
import pandas as pd
import gradio as gr
import torchaudio

from app.utils import *
from app.species_list import SPECIES_LIST

# %%
class PromptLogger():
    """
    A logger class for handling the prompt testing process of bioacoustic data.

    Attributes:
    """
    def __init__(self, max_sample=10):
        """
        Initializes the annotation logger with a counter and segment path.

        Args:
        """
        

        self.extension = None
        self.tgt_files = None
        self.current_file = None 

        self.prompt_path = None 
        self.max_sample = max_sample
        self.sample_num = 0

        self.prompt_file = None
        self.annotator = None

        self.batch_file_list = []

    def load_data(self, data_root_path, extension):
        """
        Loads the data from the specified root path and prepares it for annotation.

        Args:

        Returns:
        """
        # Load the files from the directory
        self.extension = extension
        tgt_files = glob(os.path.join(data_root_path, "**", "*.{}".format(self.extension)), recursive=True)
        self.tgt_files = tgt_files
        # Display the available files and annotation path.
        return [gr.Text("\n".join(self.tgt_files), lines=5, label="Available files:"),
                gr.Column(visible=True),
                gr.Text(os.path.join("{}".format(data_root_path), "annotations"), label="Prompt information of this round will be saved to:",
                        info="Please change the directory here if necessary!", interactive=True)]

    def load_sample_data(self, sample_num, seed=0):
        self.sample_num = sample_num
        random.seed(int(seed))
        self.sample_files = random.sample(self.tgt_files, int(sample_num))
        save_path = os.path.join(".", "temp", "prompting_sample_files")
        shutil.rmtree(save_path, ignore_errors=True) 
        os.makedirs(save_path, exist_ok=True)
        for f in self.sample_files:
            wav, sr = torchaudio.load(f)
            wav = wav.reshape(-1)
            generate_spectrogram_results(wav, sr, prefix="{}".format(f.split(os.sep)[-1].replace(".{}".format(self.extension), '')),
                                         save_path=save_path)
        return [gr.Text("\n".join(self.sample_files), lines=5, label="Available files:"),
                gr.Column(visible=True)]

    def load_models(self, model_name):
        self.weights_path = load_models(model_name)
        return ['Loaded CLAP model from {}'.format(self.weights_path),
                gr.Column(visible=True)]

    def register_prompt_file(self, prompt_path, name):
        """
        Registers an annotation file to save the annotations.

        Args:

        Returns:
        """
        # Set the annotation path and create the directory if it doesn't exist.
        self.annotator = name
        self.prompt_path = prompt_path
        os.makedirs(self.prompt_path, exist_ok=True)
        self.prompts = {"positive": "", "negative": "", "sample_files": self.sample_files} 
        return [gr.Text("Name Registered as {}.".format(name), label="Registration Successful!"),
                gr.Button("Start prompting!", visible=True)]

    def start_prompting(self):
        """
        Prepares and displays the Gradio interface for starting the annotation of the current audio file.

        Returns:
        """
        return [gr.Column(visible=True),
                gr.Accordion("Loading configurations.", open=False)]

    def populate_audio(self, file_list):
        file_list = file_list.split("\n")
        populated_path = [gr.Audio(p, visible=True, label=p, min_width=1200) for p in file_list]
        return populated_path + [gr.Audio(visible=False)] * (self.max_sample - len(file_list))

    def populate_image(self, file_list):
        file_list = file_list.split("\n")
        
        populated_path = [gr.Image(os.path.join(".", "temp", "prompting_sample_files",
                                                p.split(os.sep)[-1].replace(".{}".format(self.extension), '') + "_spec.jpg"),
                                   visible=True, label=p, interactive=False, scale=0.4, sources=[])
                          for p in file_list]
        return populated_path + [gr.Image(visible=False)] * (self.max_sample - len(file_list))
    
    def update_image(self, wav_list, neg_prompts, pos_prompts, theta):
        batch_audio_detection(wav_list=wav_list, neg_prompts=neg_prompts, pos_prompts=pos_prompts, theta=theta, 
                              output_spec=True, output_det=False,
                              save_path=os.path.join(".", "temp", "prompting_sample_files"),
                              input_ext=self.extension)
        return self.populate_image(wav_list)

    def submission(self, data_root_path):
        return [gr.Accordion("Prompting results.", open=False),
                gr.Column(visible=True),
                gr.Text("{}".format(data_root_path), label="Data directory for batch detection:",
                        info="Please change the directory here if necessary!", interactive=True),
                gr.Text("{}".format(self.extension), label="File extension for batch detection:",
                        info="Please change if necessary!", interactive=True)]

    def path_confirm(self, data_root_path, extension):

        self.batch_file_list = [os.path.abspath(p) 
                                for p in glob(os.path.join(data_root_path, "**", "*.{}".format(extension)), recursive=True)]

        return [gr.Column(visible=True),
                gr.Column(visible=True),
                gr.Accordion("There are {} files for batch detection. Open to see all the data:".format(len(self.batch_file_list)),
                             open=False),
                gr.Text("\n".join(self.batch_file_list), lines=5, label="Available files:"),
                gr.Text(os.path.join("{}".format(data_root_path), "annotations"), label="Prompt information of this round will be saved to:",
                        info="Please change the directory here if necessary!", interactive=True),
                gr.Text(os.path.join("{}".format(data_root_path), "annotations"), label="Annotation file will be saved to:",
                        info="Please change the directory here if necessary!", interactive=True)]

    def command_gen(self, data_root, extension, ann_path, prompt_path, sess_id, seed):
        os.makedirs(prompt_path, exist_ok=True)
        os.makedirs(ann_path, exist_ok=True)
        self.prompt_file = os.path.join(prompt_path, "prompt_{}_id_{}_seed_{}.json".format(self.annotator, sess_id, seed))
        return gr.Text("python batch_detection.py --data_root {} --extension {} --prompt {} --det_out {}".format(data_root, extension, self.prompt_file, ann_path),
                       label="If you are happy with that, please click on the Finish button and run the command in your terminal!",
                       interactive=False)

    def finish(self, command, neg_prompts, pos_prompts, theta):
        prompt_info = {"pos_prompts": pos_prompts,
                       "neg_prompts": neg_prompts, 
                       "theta": theta}

        with open(self.prompt_file, "w") as f:
            json.dump(prompt_info, f, indent=4)

        return [gr.Column(visible=False),
                gr.Column(visible=False),
                gr.Column(visible=False),
                gr.Text(command,
                        label="Prompt file is saved to `{}`.".format(self.prompt_file) +\
                              "Please paste the following command or " +\
                              "click the Run Batch Detection button to run batch detection.",
                        visible=True),
                gr.Button("Run Batch Detection!", visible=True),
                gr.Button(visible=False)]

    def batch_detection(self, neg_prompt, pos_prompt, theta, ann_path, extension, sess_id, seed):
        det_file = "det_{}_id_{}_seed_{}.csv".format(self.annotator, sess_id, seed)
        batch_audio_detection(wav_list=self.batch_file_list, neg_prompts=neg_prompt, pos_prompts=pos_prompt,
                              theta=theta, output_spec=False, output_det=True, 
                              save_path=ann_path, det_file=det_file,
                              progress="gr", 
                              input_ext=extension)
        return gr.Text("Detection file saved to `{}`".format(os.path.join(ann_path, det_file)), label="Batch detection finished. You can start annotation.")

# %%
class AnnLogger():
    """
    A logger class for handling the validation process of bioacoustic data annotations.

    Attributes:
    - ann_path: Path where annotation files are located.
    - temp_path: Temporary path for processing.
    - seg_path: Path where segments for validation are stored.
    - ann_files: List of annotation files available for validation.
    - file_ann: DataFrame for storing annotations of the current file.
    - file_cat: Categories of annotations for the current file.
    - aud_counter: Counter for the current audio file being validated.
    - seg_counter: Counter for the current segment being validated.
    - aud_files: List of audio files available for validation.
    - current_file: Currently selected audio file for validation.
    - ann_df: DataFrame containing all annotations for validation.
    - validator_name: Name of the validator.
    - val_file_path: Path to save the validation file.
    """
    def __init__(self, temp_path=os.path.join(".", "temp")):

        self.extension = None
        self.ann_path = None
        self.det_path = None
        self.det_file = None
        self.annotator_name = None
        self.ann_file_path = None
        self.det_df = None

        self.temp_path = temp_path
        self.seg_path = os.path.join(self.temp_path, "segs")
        self.det_file_list = []
        self.file_ann = None
        self.file_cat = []
        self.aud_counter = 0
        self.seg_counter = 0
        self.aud_files = []
        self.current_file = None

    def load_data(self, data_root_path, extension):
        """
        Loads the data from the specified root path for annotation.

        Args:
        - data_root_path (str): The root path where the data (audio files) is located.

        Returns:
        - A list of Gradio components to display available audio files and annotation path.
        """
        # Load the files from the directory
        self.extension = extension
        return [gr.Column(visible=True),
                gr.Text(os.path.join("{}".format(data_root_path), "annotations"), label="Annotation file will be saved to:", info="Please change the directory here if necessary!"),
                gr.Text(os.path.join("{}".format(data_root_path), "annotations"), label="Default detection results can be found here:", info="Please change the directory here if necessary!")]

    def register_ann_path(self, ann_path, det_path):
        """
        Registers the path of annotation files for annotation.

        Args:
        - ann_path (str): Path where the annotation files are located.

        Returns:
        - A list of Gradio components to allow the selection of an annotation file for annotation.
        """
        self.ann_path = ann_path
        self.det_path = det_path
        self.det_file_list = glob(os.path.join(self.det_path, "**", "det_*.csv"), recursive=True)

        return [gr.Dropdown(choices=self.det_file_list, label="Please select a detection file to annotate:"),
                gr.Column(visible=True)]

    def load_detected_files(self, det_file):
        self.det_file = det_file
        self.det_df = pd.read_csv(self.det_file)
        # Load the annotation file and prepare for annotation.
        self.aud_files = self.det_df["filename"].unique()
        return [
            gr.Accordion("There are {} detected files in total. Open to see all:".format(len(self.aud_files)),
                         visible=True, open=False),
            gr.Text("\n".join(list(self.aud_files)), lines=5, label="Available files:"),
            gr.Row(visible=True)
        ]

    def register_ann_file(self, name):
        """
        Registers the name of the annotator and prepares for the annotation process.

        Args:
        - name (str): Name of the annotator.

        Returns:
        - A list of Gradio components indicating successful registration and the option to start annotation.
        """
        self.annotator_name = name
        # Set the annotation file path and create the annotation file path.
        self.ann_file_path = self.det_file.replace(self.det_path, self.ann_path)\
                                          .replace("det_", "ann_{}_det_".format(self.annotator_name))
        return [gr.Text("Annotator name registered as {}.".format(name), label="Registration Successful!"),
                gr.Button("Start Annotation!", visible=True)]

    def start_annotation(self):
        """
        Starts the annotation process for the selected annotation file.

        Args:
        - ann_file (str): Path to the selected annotation file.

        Returns:
        - A list of Gradio components for initiating the annotation process.
        """

        self.current_file = self.aud_files[self.aud_counter]
        self.file_ann = self.det_df.loc[self.det_df["filename"] == self.current_file]
        self.file_cat = []

        # Generate the annotation segments for the current file.
        generate_segs(self.current_file, self.file_ann, prefix="ann")

        return [gr.Accordion("Configurations", open=False),
                gr.Column(visible=True),
                gr.Text(self.current_file),
                gr.Image(os.path.join(self.temp_path, "ann_full_spec.jpg"),
                         label="Detection Predictions:"),
                gr.Audio(self.current_file, label=self.current_file),
                gr.Text("There are {} annotated segments.".format(len(self.file_ann)),
                        label="Number of annotated segments:")]

    def end_annotation(self):
        """
        Ends the annotation process.

        Returns:
        - A list of Gradio components indicating the completion of annotation.
        """
        return [gr.Column(visible=False),
                gr.Column(visible=False),
                gr.Text(self.ann_file_path, 
                        label="Annotation saved to the following path, please close the app or move on the validation.")]

    def start_segment_annotation(self):
        """
        Starts the annotation for the detected segments of the current audio file.

        Returns:
        - A list of Gradio components for annotating the first segment.
        """
        # Set the segment counter to 0 and load the first segment for annotation.
        self.seg_counter = 0
        self.img_segs = sorted(glob(os.path.join(self.seg_path, "ann_*.jpg")))
        self.aud_segs = sorted(glob(os.path.join(self.seg_path, "ann_*.wav")))
        img_seg = self.img_segs[self.seg_counter]
        aud_seg = self.aud_segs[self.seg_counter]
        seg_id = img_seg.replace(".jpg", '').split('_')[-1]
        seg_conf = float(img_seg.replace(".jpg", '').split('_')[-2])
        seg_st = int(img_seg.replace(".jpg", '').split('_')[-3])

        return [gr.Image(img_seg, label="Seg #{}; st: {}s; conf: {}%.".format(seg_id, seg_st, seg_conf*100),
                         interactive=True, sources=[]),
                gr.Audio(aud_seg, label="Seg #{}; st: {}s; conf: {}%.".format(seg_id, seg_st, seg_conf*100)),
                gr.Text("Please use the drop down menu to assign a catgegory.", visible=True),
                gr.Column(visible=True)]

    def next_audio(self):
        """
        Proceeds to the next audio file for annotation.

        Returns:
        - A list of Gradio components for the next audio file or a completion message.
        """
        # Proceed to the next audio file for annotation.
        self.det_df.to_csv(self.ann_file_path, index=False)
        try:
            self.aud_counter += 1
            self.current_file = self.aud_files[self.aud_counter]
            self.file_ann = self.det_df.loc[self.det_df["filename"] == self.current_file]
            self.file_cat = [] 

            generate_segs(self.current_file, self.file_ann, prefix="ann")

            return [gr.Column(visible=True),
                    gr.Text("Click the Get Detection button to get detected segments.", visible=True, label="Instruction:"),
                    gr.Text(self.current_file),
                    gr.Image(os.path.join(self.temp_path, "ann_full_spec.jpg"),
                             label="Detection Predictions:"),
                    gr.Audio(self.current_file, label=self.current_file),
                    gr.Text("There are {} detected segments.".format(len(self.file_ann)),
                            label="Number of detected segments:"),
                    gr.Button("Next Audio", visible=True),
                    gr.Button(visible=False),
                    gr.Dropdown(SPECIES_LIST, label="Select a species:", value=None)]
        except:
            return [gr.Column(visible=False),
                    gr.Text("No more detections to annotate!", visible=True, label="IMPORTANT INFO:"),
                    gr.Text(),
                    gr.Image(),
                    gr.Audio(),
                    gr.Text(),
                    gr.Button(visible=False),
                    gr.Button("Submit", visible=True),
                    gr.Dropdown(SPECIES_LIST, label="Select a species:", value=None)]

    def next_segment(self, category):
        """
        Proceeds to the next segment for annotation within the current audio file.

        Args:
        - category (str): The selected category for the current segment.

        Returns:
        - A list of Gradio components for the next segment or a completion message.
        """
        # Proceed to the next segment for annotation.
        try:
            self.file_cat.append(category)
            self.seg_counter += 1
            img_seg = self.img_segs[self.seg_counter]
            aud_seg = self.aud_segs[self.seg_counter]
            return [gr.Image(img_seg, label="Spectrogram Seg:", interactive=True, sources=[]),
                    gr.Audio(aud_seg, label="Audio Seg:", scale=0.8),
                    gr.Column(visible=True),
                    gr.Text("Please select a category in the dropdown menu.", label="Instruction:"),
                    gr.Dropdown(SPECIES_LIST, label="Select a species:", value=None)]
        except:
            self.det_df.loc[self.det_df["filename"] == self.current_file, "species"] = self.file_cat
            return [gr.Image(),
                    gr.Audio(),
                    gr.Column(visible=False),
                    gr.Text("NO MORE DETECTED SEGMENTS. PLEASE MOVE ON TO THE NEXT AUDIO.", label="INPORTANT INFO:"),
                    gr.Dropdown(SPECIES_LIST, label="Select a species:", value=None)]

# %%
class ValLogger():
    """
    A logger class for handling the validation process of bioacoustic data annotations.

    Attributes:
    - ann_path: Path where annotation files are located.
    - temp_path: Temporary path for processing.
    - seg_path: Path where segments for validation are stored.
    - ann_files: List of annotation files available for validation.
    - file_ann: DataFrame for storing annotations of the current file.
    - file_cat: Categories of annotations for the current file.
    - aud_counter: Counter for the current audio file being validated.
    - seg_counter: Counter for the current segment being validated.
    - aud_files: List of audio files available for validation.
    - current_file: Currently selected audio file for validation.
    - ann_df: DataFrame containing all annotations for validation.
    - validator_name: Name of the validator.
    - val_file_path: Path to save the validation file.
    """
    def __init__(self, temp_path=os.path.join(".", "temp")):
        self.extension = None
        self.data_root = None
        self.ann_path = None
        self.aud_files = []

        self.validator_name = None

        self.val_file_path = None
        self.ann_df = None
        self.unique_cat = []

        self.current_cat = None
        self.ann_df_cat = None
        self.aud_files_cat = None
        self.cat_segs = []

        self.seg_counter = 0
        self.seg_info = []

        self.temp_path = temp_path
        self.seg_path = os.path.join(self.temp_path, "segs")
        self.ann_files = []
        self.file_ann = None
        self.file_cat = None
        self.aud_counter = 0
        self.current_file = None

    def load_data(self, data_root_path, extension):
        """
        Loads the data from the specified root path for validation.

        Args:
        - data_root_path (str): The root path where the data (audio files) is located.

        Returns:
        - A list of Gradio components to display available audio files and annotation path.
        """
        # Load the files from the directory
        self.data_root = data_root_path
        self.extension = extension
        return [gr.Column(visible=True),
                gr.Text(os.path.join("{}".format(self.data_root), "annotations"), label="Default path where annotation and validation files are saved:",
                        info="Please change the directory here if necessary!")]

    def register_ann_path(self, ann_path):
        """
        Registers the path of annotation files for validation.

        Args:
        - ann_path (str): Path where the annotation files are located.

        Returns:
        - A list of Gradio components to allow the selection of an annotation file for validation.
        """
        self.ann_path = ann_path
        self.ann_files = glob(os.path.join(self.ann_path, "**", "ann_*.csv"), recursive=True)

        return [gr.Dropdown(choices=self.ann_files, label="Please select an annotation file to validate:"),
                gr.Column(visible=True)]

    def load_ann_files(self, ann_file):
        self.ann_df = pd.read_csv(ann_file)
        self.aud_files = self.ann_df["filename"].unique()

        if len(self.aud_files) > 1:
            actual_root = os.path.commonpath(list(self.aud_files))
        else:
            actual_root = self.aud_files[0].rsplit(os.sep, 1)[0]
        
        if self.data_root != actual_root:
            self.data_root = actual_root

        return [
            gr.Accordion("There are {} annotated files in total. Open to see all:".format(len(self.aud_files)),
                         visible=True, open=False),
            gr.Text("\n".join(list(self.aud_files)), lines=5, label="Available files:"),
            gr.Row(visible=True)
        ]

    def register_val_file(self, name):
        """
        Registers the name of the validator and prepares for the validation process.

        Args:
        - name (str): Name of the validator.

        Returns:
        - A list of Gradio components indicating successful registration and the option to start validation.
        """
        self.validator_name = name
        return [gr.Text("Validator Name Registered as {}.".format(name), label="Registration Successful!"),
                gr.Button("Start validation!", visible=True)]

    def start_validation(self, ann_file):
        """
        Starts the validation process for the selected annotation file.

        Args:
        - ann_file (str): Path to the selected annotation file.

        Returns:
        - A list of Gradio components for initiating the validation process.
        """
        # Set the annotation file path and create the validation file path.
        self.val_file_path = ann_file.replace("ann_", "val_{}_ann_".format(self.validator_name))
        # Load the annotation file and prepare for validation.
        self.ann_df = pd.read_csv(ann_file)
        self.val_df = copy.deepcopy(self.ann_df)
        self.unique_cat = list(self.ann_df["species"].unique())

        return [gr.Accordion("Configurations", open=False),
                gr.Accordion("Category selection", open=True, visible=True),
                gr.Dropdown(choices=self.unique_cat, label="Available categories:"),
                gr.Markdown("There are {} categories annotated. Please select one category in the dropdown menu to validate.".format(len(self.unique_cat)))]

    def fetch_segments(self, tgt_cat):

        self.current_cat = tgt_cat
        self.ann_df_cat = self.ann_df.loc[self.ann_df["species"] == self.current_cat]
        self.aud_files_cat = self.ann_df_cat["filename"].unique()

        # %%
        gr_progress = gr.Progress()
        gr_progress(0, desc="Starting")

        for i in gr_progress.tqdm(range(len(self.aud_files_cat)),
                                  desc="Fetching all segments for {}. This can take a while...".format(self.current_cat)):
            aud_f = self.aud_files_cat[i]
            ann_f = self.ann_df_cat.loc[self.ann_df_cat["filename"] == aud_f]
            generate_segs(aud_f, ann_f,
                          prefix=aud_f.replace(self.data_root+os.sep, '').replace(".{}".format(self.extension), ''),
                          clean_seg_folder=(i == 0), save_full=False)

        self.cat_segs = glob(os.path.join(".", "temp", "segs", "**", "*.jpg"), recursive=True)

        return [gr.Markdown("There are {} segments annotated for {}.".format(len(self.ann_df_cat), self.current_cat)),
                gr.Column(visible=False),
                gr.Button("Next", visible=True)]

    def populate_segments(self):

        outputs = [gr.Column(visible=True),
                   gr.Accordion("Category selection", open=False),
                   gr.Column(visible=True),
                   gr.Text("Please use the dropdown menu to change the catgegory.",
                           label="Instruction:", visible=True),
                   gr.Button("Next Batch", visible=True),
                   gr.Button(value="Save and select a new category", visible=False)]

        if len(self.cat_segs) - self.seg_counter >= 5:
            max_render = 5
        else:
            max_render = len(self.cat_segs) - self.seg_counter
        for i in range(self.seg_counter, (max_render + self.seg_counter)):
            seg = self.cat_segs[i]
            aud = seg.replace("ImgSeg", "AudSeg").replace(".jpg", ".wav")
            file = seg.replace(os.path.join(".", "temp", "segs"), self.data_root).split("_ImgSeg_")[0]+".{}".format(self.extension)
            seg_conf = float(seg.replace(".jpg", '').split('_')[-2])
            seg_st = int(seg.replace(".jpg", '').split('_')[-3])
            cat = self.ann_df["species"][(self.ann_df["filename"] == file) & (self.ann_df["start_time(s)"] == seg_st)].item()
            outputs += [gr.Column(visible=True),
                        gr.Image(seg, height=200, interactive=True, sources=[], label="Spectrogram:"),
                        gr.Audio(aud, label="Audio:"),
                        gr.Dropdown(SPECIES_LIST, value=cat, label="Select a different species if needed.", interactive=True, min_width=50),
                        gr.Markdown("{}; st: {}s; conf: {}%".format(file, seg_st, seg_conf*10))]

            self.seg_counter += 1
            self.seg_info.append((file, seg_st, self.current_cat))
        if max_render < 5:
            for i in range(5 - max_render):
                outputs += [gr.Column(visible=False),
                            gr.Image(),
                            gr.Audio(),
                            gr.Dropdown(SPECIES_LIST, value=None, label="Select a different species if needed."),
                            gr.Markdown()]
        return outputs

    def batch_update(self, drop_1, drop_2, drop_3, drop_4, drop_5):
        val_cats = [drop_1, drop_2, drop_3, drop_4, drop_5]
        for i in range(len(self.seg_info)):
            if self.seg_info[i][2] != val_cats[i]:
                self.val_df.loc[(self.ann_df["filename"] == self.seg_info[i][0]) &\
                                (self.ann_df["start_time(s)"] == self.seg_info[i][1]) &\
                                (self.ann_df["species"] == self.seg_info[i][2]), "species"] = val_cats[i]
        self.seg_info = []

        if len(self.cat_segs) > self.seg_counter:
            return self.populate_segments()
        else:
            self.seg_counter = 0
            self.unique_cat.remove(self.current_cat)
            self.val_df.to_csv(self.val_file_path, index=False)
            return [gr.Column(visible=True),
                    gr.Accordion("Category selection", open=False),
                    gr.Column(visible=True),
                    gr.Text("No more segments to validate for {}. Please select another category.".format(self.current_cat),
                            label="Instruction:", visible=True),
                    gr.Button(visible=False),
                    gr.Button(value="Save and select a new category", visible=True)] +\
                   [gr.Column(visible=False),
                    gr.Image(),
                    gr.Audio(),
                    gr.Dropdown(SPECIES_LIST, value=None, label="Select a different species if needed."),
                    gr.Markdown()] * 5

    def new_cat(self):

        self.current_cat = None

        return [gr.Accordion("Category selection", open=True, visible=True),
                gr.Column(visible=len(self.unique_cat) > 0),
                gr.Column(visible=False),
                gr.Dropdown(choices=self.unique_cat, label="Available categories:", value=None),
                gr.Markdown("There are {} categories left for validate. ".format(len(self.unique_cat)) +\
                            "Please select one category in the dropdown menu to validate." 
                            if len(self.unique_cat) > 0 else "Please submit"), 
                gr.Button(visible=False),
                gr.Button("Submit", visible=len(self.unique_cat) == 0)]

    def end_validation(self):
        """
        Ends the validation process.

        Returns:
        - A list of Gradio components indicating the completion of validation.
        """
        self.val_df.to_csv(self.val_file_path, index=False)
        return [gr.Column(visible=True),
                gr.Accordion(visible=False),
                gr.Column(visible=False),
                gr.Text(self.val_file_path, 
                        label="Validation file saved to the following path, please close the annotation app.",
                        visible=True)]
