
# %%
import os
import shutil
from glob import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from maad import sound, util

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

import torch
from torch.utils.data import DataLoader
import torchaudio

import gradio as gr

from CLAP.CLAPWrapper import CLAPWrapper
from CLAP.dataset.datasets import Inference_DS

from species_list import SPECIES_LIST

# %%
# Utility function
def softmax(x, T=1):
    e_x = np.exp(x / T)
    return e_x / e_x.sum(axis=1).reshape(-1, 1)

# %%
# Define the weights path and load the CLAP model. 
#weights_path = 'models/weights/jan23.pth'
clap = None
def load_models(model_name):
    """
    Loads the CLAP model from the weights path.
    Args:
        model_name (str): Name of the model to load.

    Returns:
        str: Message indicating that the model was loaded.    
    """
    # Set clap as a global variable
    global clap
    # Load the model
    weights_path = None
    weights_path = 'weights/{}.pth'.format(model_name)
    
    print("Loading CLAP model..")
    clap = CLAPWrapper(weights_path, use_cuda=True)
    # Load the weights
    clap.load_clap()
    print("CLAP model loaded!")
    return ['Loaded CLAP model from {}'.format(weights_path),
            gr.Column(visible=True)]

# %%
def compute_similarity(model, data_loader, class_prompts, neg_n=1, pos_n=1, theta=0.5):
    """
    Computes the similarity between audio embeddings and class text embeddings.

    This function takes audio data from a data loader, computes their embeddings using the given model, 
    and then calculates the similarity of these embeddings with the text embeddings of class prompts. 
    The function returns the total similarity scores and a prediction array based on a threshold.

    Args:
    - model: The CLAP model wrapper that provides methods for getting text embeddings, audio embeddings, and computing similarity.
    - data_loader: An iterable data loader that provides batches of audio data.
    - class_prompts: A list of class prompts (text) for which embeddings are computed.
    - neg_n (int, optional): The number of negative samples to consider. Defaults to 1.
    - pos_n (int, optional): The number of positive samples to consider. Defaults to 1.
    - theta (float, optional): Threshold for converting similarity scores to binary predictions. Defaults to 0.5.

    Returns:
    - total_scores (numpy.ndarray): An array of similarity scores between audio and text embeddings.
    - preds (numpy.ndarray): An array of binary predictions derived from the similarity scores and the threshold theta.
    """
    # Get the text embeddings using the model embeddings function. 
    text_embeddings = model.get_text_embeddings(class_prompts)

    total_scores = []
    # Iterate over the data loader to get audio embeddings and compute similarity.
    for x in data_loader:
        
        if torch.cuda.is_available():
            x = x.cuda()
        
        audio_embeddings = model._get_audio_embeddings(x)
        similarity = model.compute_similarity(audio_embeddings, text_embeddings)
        
        scores = []
        # Compute the similarity scores for each audio embedding with the class prompts.
        for i in range(len(similarity)):
            values, indices = similarity[i].topk(len(class_prompts))
            values, indices = values.detach().cpu().numpy(), indices.detach().cpu().numpy()
            neg_sc = 0
            pos_sc = 0
            for i in range(neg_n):
                neg_sc += values[indices == i].item()
            for i in range(neg_n, neg_n + pos_n):
                pos_sc += values[indices == i].item()
            scores.append((neg_sc, pos_sc))
        
        total_scores.append(np.array(scores))
    # Concatenate the total scores and apply softmax to get the final scores.
    total_scores = np.concatenate(total_scores, 0)
    total_scores = softmax(total_scores)
    positive_scores = total_scores[:, 1]
    preds = (positive_scores >= theta).astype(int)

    return total_scores, preds

# %%
def generate_prediction_results(seg_size, wav, sr, preds, total_scores,
                                nperseg=1024, noverlap=512, db_range=90,
                                cmap='gray', khz_lims=[0, 10], save_path="./temp"):
    """
    Generates and saves prediction results including spectrogram images and audio segments.

    This function takes waveform data, its sample rate, predictions, and total scores, and generates
    spectrogram images highlighting the segments where predictions are positive. It also saves 
    these identified segments as separate audio files. The function uses matplotlib for generating 
    spectrogram images and librosa for audio processing.

    Args:
    - seg_size (int): Size of each segment in the waveform data.
    - wav (numpy.ndarray): The waveform data as a numpy array.
    - sr (int): Sample rate of the waveform data.
    - preds (numpy.ndarray): An array of binary predictions for each segment.
    - total_scores (numpy.ndarray): An array of total scores corresponding to each segment.
    - save_path (str, optional): The base path where the spectrogram and segments are saved. Defaults to "./temp".

    The function creates a directory for segments, generates a full spectrogram of the waveform, 
    and for each segment where the prediction is positive, it overlays the segment on the spectrogram, 
    saves the segment as an image, and extracts and saves the corresponding audio segment as a WAV file.
    """

    # Create a directory for segments and remove any existing segments.
    seg_folder = os.path.join(save_path, "segs")
    shutil.rmtree(seg_folder, ignore_errors=True) 
    os.makedirs(seg_folder, exist_ok=True)

    matplotlib.use("Agg")

    wav = wav.numpy()
    # Compute the spectrogram of the waveform.
    Sxx, _, _, ext = sound.spectrogram(wav, sr, nperseg=nperseg, noverlap=noverlap,
                                       flims=[khz_lims[0] * 1000, khz_lims[1] * 1000])
    ext[-1] = khz_lims[1]
    Sxx_db = util.power2dB(Sxx, db_range, 0)

    fig, ax = plt.subplots(figsize=(30, 5))

    ax.imshow(Sxx_db, extent=ext, origin="lower", cmap=cmap, 
              vmin=None, vmax=None)

    ax.axis("tight")
    fig.tight_layout()

    # Iterate over the predictions and overlay the segments on the spectrogram.
    seg_counter = 0
    for i in range(len(preds)):
        p = preds[i]
        s = total_scores[i][1]
        if p == 1:
            ax.add_patch(Rectangle((seg_size * i, 0), seg_size, khz_lims[1], linewidth=2, facecolor="orange",
                                   edgecolor="red", alpha=0.25))
            ax.text(seg_size * i + 0.3, khz_lims[1] - 0.7, "Seg #{}".format(seg_counter), c="red", fontsize=15)
            ax.text(seg_size * i + 0.3, khz_lims[1] - 1.3, "Conf: {:.2f}".format(s*100), c="red", fontsize=15)
            bbox = Bbox([[seg_size * i, 0],[seg_size * (i + 1), khz_lims[1]]])
            bbox = bbox.transformed(ax.transData).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(seg_folder, "ImgSeg_{}_{}_{}_{:.2f}.png").format(seg_counter,
                                                                                      seg_size * i,
                                                                                      seg_size * (i + 1),
                                                                                      s*100),
                        bbox_inches=bbox)
            sf.write(os.path.join(seg_folder, "AudSeg_{}_{}_{}_{:.2f}.wav").format(seg_counter,
                                                                               seg_size * i, 
                                                                               seg_size * (i + 1),
                                                                               s*100),
                     wav[seg_size*i*sr : seg_size*(i+1)*sr], sr)
            seg_counter += 1

    ax.set(xlabel=None)
    ax.tick_params(axis='x', which="major", labelsize=17)

    print("Saving figure..")
    plt.savefig(os.path.join(save_path, "full_spec.png"), bbox_inches="tight")  

# %%
def single_audio_detection(wav_path, neg_prompts=None, pos_prompts=None, theta=0.5):
    """
    Performs detection on a single audio file and returns an annotated spectrogram with the confidence scores.
    Args:
        wav_path (str): Path to the wav file.
        theta (float): Confidence threshold for detection.
    Returns:
        annotated_img (np.array): Annotated mel spectrogram with bounding box instances and confidence scores.
    """
    # Load the wav file.
    print("Loading data..")
    print(wav_path)
    wav, sr = torchaudio.load(wav_path)
    wav = wav.reshape(-1)

    # Split the wav into windows of 6 seconds.
    inf_dset = Inference_DS(wav, sr, seg_size=6)
    # Add the segments to the DataLoader.
    inf_dl = DataLoader(
                inf_dset, batch_size=10, shuffle=False, 
                pin_memory=True, num_workers=4, drop_last=False
        )

    neg_prompts = neg_prompts.rstrip(';').split(';')
    pos_prompts = pos_prompts.rstrip(';').split(';')
    neg_n = len(neg_prompts)
    pos_n = len(pos_prompts)
    classes = neg_prompts + pos_prompts
    print(classes)
    prompt = "this is a sound of "
    class_prompts = [prompt + x for x in classes]

    print("Computing similarities..")

    total_scores, preds = compute_similarity(clap, inf_dl, class_prompts,
                                             neg_n=neg_n, pos_n=pos_n, theta=theta)

    print("Outputing figures..")

    save_path = "./temp"
    generate_prediction_results(inf_dset.seg_size, wav, sr, preds, total_scores,
                                save_path=save_path)

    print('Done.')

    return [gr.Text("There are {} possible calls.".format(np.sum(preds)), label="Number of detected events:"),
            gr.Image(os.path.join(save_path, "full_spec.png"), label="Detection Predictions:"),
            gr.Column(visible=True),
            gr.Text("Please click the Detect button for possible call detections.", label="Instruction:", visible=True)]

# %%
def generate_validation_segs(wav_path, file_ann, 
                             nperseg=1024, noverlap=512, db_range=90,
                             cmap='gray', khz_lims=[0, 10], save_path="./temp"):
    """
    Generates and saves validation segments as images and audio files.

    This function loads an audio waveform from a specified path, and using provided annotation data,
    it generates and saves spectrogram images and corresponding audio segments for validation. It 
    highlights the segments specified in the annotation data on the spectrogram.

    Args:
    - wav_path (str): The file path of the audio waveform.
    - file_ann (pandas.DataFrame): A dataframe containing the annotation data with columns 'start_time(s)',
      'end_time(s)', and 'detection_conf'.
    - save_path (str, optional): The base path where the spectrogram and segments are saved. Defaults to "./temp".

    The function creates a directory for segments, generates a full spectrogram of the waveform, 
    and overlays each annotated segment on the spectrogram. It saves each segment as an image in a temp directory, 
    and extracts and saves the corresponding audio segment as a WAV file.
    """
    # Load the wav file.
    wav, sr = torchaudio.load(wav_path)
    wav = wav.reshape(-1)
    #  Create a directory for segments and remove any existing segments.
    seg_folder = os.path.join(save_path, "segs")
    shutil.rmtree(seg_folder, ignore_errors=True) 
    os.makedirs(seg_folder, exist_ok=True)

    matplotlib.use("Agg")

    wav = wav.numpy()
    # Compute the mel spectrogram of the waveform.
    Sxx, _, _, ext = sound.spectrogram(wav, sr, nperseg=nperseg, noverlap=noverlap,
                                       flims=[khz_lims[0] * 1000, khz_lims[1] * 1000])
    ext[-1] = khz_lims[1]
    Sxx_db = util.power2dB(Sxx, db_range, 0)

    fig, ax = plt.subplots(figsize=(30, 5))

    ax.imshow(Sxx_db, extent=ext, origin="lower", cmap=cmap, 
              vmin=None, vmax=None)
    ax.axis("tight")
    fig.tight_layout()

    # Iterate over the annotations and overlay the segments on the spectrogram.
    for i in range(len(file_ann)):

        st = file_ann["start_time(s)"].values[i]
        ed = file_ann["end_time(s)"].values[i]
        conf = file_ann["detection_conf"].values[i]

        ax.add_patch(Rectangle((st, 0), ed - st, khz_lims[1], linewidth=2, facecolor="orange",
                               edgecolor="red", alpha=0.25))
        ax.text(st + 0.3, khz_lims[1] - 0.7, "Seg #{}".format(i), c="red", fontsize=15)
        ax.text(st + 0.3, khz_lims[1] - 1.3, "Conf: {:.2f}".format(conf), c="red", fontsize=15)
        bbox = Bbox([[st, 0],[ed, khz_lims[1]]])
        bbox = bbox.transformed(ax.transData).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(seg_folder, "ValImgSeg_{}.png").format(i),
                    bbox_inches=bbox)
        sf.write(os.path.join(seg_folder, "ValAudSeg_{}.wav").format(i),
                 wav[st*sr : ed*sr], sr)

    ax.set(xlabel=None)
    ax.tick_params(axis='x', which="major", labelsize=17)

    print("Saving figure..")
    plt.savefig(os.path.join(save_path, "val_full_spec.png"), bbox_inches="tight")  

# %%
def tab_init():
    """
    Initializes and returns the state of Gradio columns for tab switching.

    This function is designed to be used as a callback for Gradio interface buttons. It manages
    the visibility of two columns, typically used in a tab-like interface. When invoked, it sets 
    the first column to invisible and the second column to visible, facilitating a switch between 
    tabs in the Gradio interface.

    Returns:
    - List[gr.Column, gr.Column]: A list containing two Gradio Column objects. The first Column 
      object is set to invisible, and the second Column object is set to visible.
    """
    return [gr.Column(visible=False), gr.Column(visible=True)]

# %%
class AnnLogger():
    """
    A logger class for handling the annotation process of bioacoustic data.

    Attributes:
    - counter: Counter to keep track of the current file being annotated.
    - tgt_files: List of target files available for annotation.
    - current_file: Currently selected file for annotation.
    - seg_counter: Counter for the current segment being annotated.
    - seg_path: Path where annotated segments are stored.
    - ann_path: Path where annotation results are saved.
    - img_segs: List of image segment paths.
    - aud_segs: List of audio segment paths.
    - ann_file: File path for saving annotations.
    - annotations: File handle for the annotation file.
    """
    def __init__(self, initial_counter=0, seg_path="./temp/segs"):
        """
        Initializes the annotation logger with a counter and segment path.

        Args:
        - initial_counter (int): Initial value of the file counter. Defaults to 0.
        - seg_path (str): Path to the directory where segments are stored. Defaults to "./temp/segs".
        """
        self.counter = initial_counter
        self.tgt_files = None
        self.current_file = None 

        self.seg_counter = 0
        self.seg_path = seg_path
        self.ann_path = None 
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
        self.current_file = self.tgt_files[self.counter]
        # Display the available files and annotation path.
        return [gr.Text("\n".join(tgt_files), lines=5, label="Available files:"),
                gr.Column(visible=True),
                gr.Text("{}/annotations".format(data_root_path), label="Annotation file will be saved to:", info="Please change the directory here if necessary!")]

    def register_ann_file(self, ann_path, name):
        """
        Registers an annotation file to save the annotations.

        Args:
        - ann_path (str): Path where the annotation file will be saved.
        - name (str): Name of the annotator.

        Returns:
        - List containing Gradio components to indicate successful registration.
        """
        # Set the annotation path and create the directory if it doesn't exist.
        self.ann_path = ann_path
        os.makedirs(self.ann_path, exist_ok=True)
        self.ann_file = os.path.join(self.ann_path, "ann_{}.csv".format(name))
        self.annotations = open(self.ann_file, "w")
        self.annotations.write("filename,start_time(s),end_time(s),species,detection_conf\n")
        return [gr.Text("Name Registered as {}.".format(name), label="Registration Successful!"),
                gr.Button("Start annotations!", visible=True)]

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
    def __init__(self, temp_path="./temp"):
        self.ann_path = None
        self.temp_path = temp_path
        self.seg_path = os.path.join(self.temp_path, "segs")
        self.ann_files = []
        self.file_ann = None
        self.file_cat = None
        self.aud_counter = 0
        self.seg_counter = 0
        self.aud_files = []
        self.current_file = None
        self.ann_df = None
        self.validator_name = None
        self.val_file_path = None

    def load_data(self, data_root_path):
        """
        Loads the data from the specified root path for validation.

        Args:
        - data_root_path (str): The root path where the data (audio files) is located.

        Returns:
        - A list of Gradio components to display available audio files and annotation path.
        """
        # Load the files from the directory
        tgt_files = glob(os.path.join(data_root_path, "**/*.wav"), recursive=True)
        return [gr.Text("\n".join(tgt_files), lines=5, label="Available files:"),
                gr.Column(visible=True),
                gr.Text("{}/annotations".format(data_root_path), label="Annotation file will be saved to:", info="Please change the directory here if necessary!")]

    def register_ann_path(self, ann_path):
        """
        Registers the path of annotation files for validation.

        Args:
        - ann_path (str): Path where the annotation files are located.

        Returns:
        - A list of Gradio components to allow the selection of an annotation file for validation.
        """
        self.ann_path = ann_path
        self.ann_files = glob(os.path.join(self.ann_path, "**/*.csv"), recursive=True)

        return [gr.Dropdown(choices=self.ann_files, label="Please select an annotation file to validate:"),
                gr.Column(visible=True)]

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
        self.val_file_path = ann_file.replace(".csv", "_{}_val.csv".format(self.validator_name))
        # Load the annotation file and prepare for validation.
        self.ann_df = pd.read_csv(ann_file)
        self.aud_files = self.ann_df["filename"].unique()

        self.current_file = self.aud_files[self.aud_counter]
        self.file_ann = self.ann_df.loc[self.ann_df["filename"] == self.current_file]
        self.file_cat = self.file_ann["species"].values
        # Generate the validation segments for the current file.
        generate_validation_segs(self.current_file, self.file_ann)

        return [gr.Accordion("Configurations", open=False),
                gr.Column(visible=True),
                gr.Text(self.current_file),
                gr.Image(os.path.join(self.temp_path, "val_full_spec.png"),
                         label="Detection Predictions:"),
                gr.Audio(self.current_file, label=self.current_file),
                gr.Text("There are {} annotated segments.".format(len(self.file_ann)),
                        label="Number of annotated segments:")]

    def end_validation(self):
        """
        Ends the validation process.

        Returns:
        - A list of Gradio components indicating the completion of validation.
        """
        return [gr.Column(visible=False),
                gr.Column(visible=False),
                gr.Text(self.val_file_path, 
                        label="Annotation saved to the following path, please close the annotation app.")]

    def start_segment_validation(self):
        """
        Starts the validation for the first segment of the current audio file.

        Returns:
        - A list of Gradio components for validating the first segment.
        """
        # Set the segment counter to 0 and load the first segment for validation.
        self.seg_counter = 0
        self.img_segs = sorted(glob(os.path.join(self.seg_path, "*.png")))
        self.aud_segs = sorted(glob(os.path.join(self.seg_path, "*.wav")))
        img_seg = self.img_segs[self.seg_counter]
        aud_seg = self.aud_segs[self.seg_counter]
        ann_cat = self.file_cat[self.seg_counter]

        return [gr.Image(img_seg, label="Spectrogram Seg:"),
                gr.Audio(aud_seg, label="Audio Seg:"),
                gr.Dropdown(SPECIES_LIST, label="Select a different species if the current is wrong:", value=ann_cat),
                gr.Text(visible=True),
                gr.Column(visible=True)]

    def next_audio(self):
        """
        Proceeds to the next audio file for validation.

        Returns:
        - A list of Gradio components for the next audio file or a completion message.
        """
        # Proceed to the next audio file for validation.
        try:
            self.aud_counter += 1
            self.current_file = self.aud_files[self.aud_counter]
            self.file_ann = self.ann_df.loc[self.ann_df["filename"] == self.current_file]
            self.file_cat = self.file_ann["species"].values

            generate_validation_segs(self.current_file, self.file_ann)

            return [gr.Column(visible=True),
                    gr.Text(visible=False),
                    gr.Text(self.current_file),
                    gr.Image(os.path.join(self.temp_path, "val_full_spec.png"),
                             label="Detection Predictions:"),
                    gr.Audio(self.current_file, label=self.current_file),
                    gr.Text("There are {} annotated segments.".format(len(self.file_ann)),
                            label="Number of annotated segments:"),
                    gr.Button("Next Audio", visible=True),
                    gr.Button(visible=False)]
        except:
            self.ann_df.to_csv(self.val_file_path, index=False)
            return [gr.Column(visible=False),
                    gr.Text("No more annotations to validate!", visible=True, label="IMPORTANT INFO:"),
                    gr.Text(),
                    gr.Image(),
                    gr.Audio(),
                    gr.Text(),
                    gr.Button(visible=False),
                    gr.Button("Submit", visible=True)]

    def next_segment(self, category):
        """
        Proceeds to the next segment for validation within the current audio file.

        Args:
        - category (str): The selected category for the current segment.

        Returns:
        - A list of Gradio components for the next segment or a completion message.
        """
        # Proceed to the next segment for validation.
        try:
            ann_cat = self.file_cat[self.seg_counter]
            if category != ann_cat:
                self.file_cat[self.seg_counter] = category
            self.seg_counter += 1
            img_seg = self.img_segs[self.seg_counter]
            aud_seg = self.aud_segs[self.seg_counter]
            ann_cat = self.file_cat[self.seg_counter]
            return [gr.Image(img_seg, label="Spectrogram Seg:"),
                    gr.Audio(aud_seg, label="Audio Seg:"),
                    gr.Dropdown(SPECIES_LIST, label="Select a species:", value=ann_cat),
                    gr.Column(visible=True),
                    gr.Text("Please select a category in the dropdown menu.", label="Instruction:")]
        except:
            self.ann_df.loc[self.ann_df["filename"] == self.current_file, "species"] = self.file_cat
            return [gr.Image(),
                    gr.Audio(),
                    gr.Dropdown(),
                    gr.Column(visible=False),
                    gr.Text("NO MORE DETECTED SEGMENTS. PLEASE MOVE ON TO THE NEXT AUDIO.", label="INPORTANT INFO:")]