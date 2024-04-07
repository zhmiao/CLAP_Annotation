
# %%
import os
import shutil
import time
import numpy as np
import soundfile as sf
from maad import sound, util
from tqdm import tqdm

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

import torch
from torch.utils.data import DataLoader
import torchaudio

import gradio as gr

from CLAP.CLAPWrapper import CLAPWrapper
from CLAP.dataset.datasets import Inference_DS, Batch_Inference_DS

from app.species_list import SPECIES_LIST

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
    # clap = None
    print("CLAP model loaded!")
    return weights_path

# %%
def compute_similarity(model, data_loader, class_prompts, neg_n=1, pos_n=1, theta=0.5, progress="tqdm"):
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

    
    - total_scores (numpy.ndarray): An array of similarity scores between audio and text embeddings.
    - preds (numpy.ndarray): An array of binary predictions derived from the similarity scores and the threshold theta.
    """
    # Get the text embeddings using the model embeddings function. 
    text_embeddings = model.get_text_embeddings(class_prompts)

    total_scores = []
    # Iterate over the data loader to get audio embeddings and compute similarity.
    # for x in data_loader:

    if progress == "tqdm":
        progress = tqdm
    elif progress == "gr":
        gr_progress = gr.Progress()
        gr_progress(0, desc="Starting..")
        progress = gr_progress.tqdm

    data_iter = iter(data_loader)

    time.sleep(2)

    # for i in progress.tqdm(range(len(data_loader))):
    for i in progress(range(len(data_loader)), desc="Detection in process..."):

        x = next(data_iter) 
        
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
def generate_spectrogram_results(wav, sr, seg_size=None, preds=None, total_scores=None,
                                 nperseg=1024, noverlap=512, db_range=90,
                                 cmap='gray', khz_lims=[0, 10], prefix="full", save_path="./temp",
                                 output_segs=True):
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
    if output_segs:
        seg_folder = os.path.join(save_path, "segs")
        shutil.rmtree(seg_folder, ignore_errors=True) 
        os.makedirs(seg_folder, exist_ok=True)
    
    os.makedirs(save_path, exist_ok=True)

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

    if preds is not None:
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
                
                if output_segs:
                    fig.savefig(os.path.join(seg_folder, "ImgSeg_{}_{}_{}_{:.2f}.jpg").format(seg_counter,
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

    print("Saving figure {}..".format(prefix))
    plt.savefig(os.path.join(save_path, "{}_spec.jpg".format(prefix)), bbox_inches="tight")  


# %%
def batch_audio_detection(wav_list, neg_prompts=None, pos_prompts=None, theta=0.5,
                          output_spec=True, output_det=False, save_path='./temp',
                          det_file="det.csv", progress="tqdm"):

    if isinstance(wav_list, str):
        wav_list = wav_list.split("\n")

    # Split the wav into windows of 6 seconds.
    inf_dset = Batch_Inference_DS(wav_list)

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
                                             neg_n=neg_n, pos_n=pos_n, theta=theta,
                                             progress=progress)

    if output_spec:
        print("Outputing figures..")
        shutil.rmtree(save_path, ignore_errors=True) 

        for f in np.unique(inf_dset.data):
            wav, sr = torchaudio.load(f)
            wav = wav.reshape(-1)
            preds_f = preds[np.array(inf_dset.data) == f]
            scores_f = total_scores[np.array(inf_dset.data) == f]
            generate_spectrogram_results(wav, sr, inf_dset.seg_size, preds_f, scores_f,
                                         save_path=save_path, prefix=f.split('/')[-1].replace(".wav", ''),
                                         output_segs=False)
    if output_det:
        print("Outputing detection results..")
        with open(os.path.join(save_path, det_file), 'w') as det_preds:
            det_preds.write("filename,start_time(s),end_time(s),species,detection_conf\n")
            for f, st, p, s in zip(inf_dset.data, inf_dset.sts, preds, total_scores):
                if p == 1:
                    det_preds.write("{},{},{},{},{}\n".format(f, st, st+inf_dset.seg_size, "None", s[1]))

    print('Done.')

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
    generate_spectrogram_results(wav, sr, inf_dset.seg_size, preds, total_scores,
                                 save_path=save_path)

    print('Done.')

    # return [gr.Text("There are {} possible calls.".format(np.sum(preds)), label="Number of detected events:"),
    #         gr.Image(os.path.join(save_path, "full_spec.jpg"), label="Detection Predictions:"),
    #         gr.Column(visible=True),
    #         gr.Text("Please click the Detect button for possible call detections.", label="Instruction:", visible=True)]

# %%
def generate_segs(wav_path, file_ann, 
                  nperseg=1024, noverlap=512, db_range=90,
                  cmap='gray', khz_lims=[0, 10], save_path="./temp", prefix="Ann",
                  clean_seg_folder=True, save_full=True):
    """
    Generates and saves segments as images and audio files directly from detection or annotations.

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
    if clean_seg_folder:
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
        fig.savefig(os.path.join(seg_folder, "{}_ImgSeg_{}_{}.jpg".format(prefix, st, i)),
                    bbox_inches=bbox)
        sf.write(os.path.join(seg_folder, "{}_AudSeg_{}_{}.wav".format(prefix, st, i)),
                 wav[st*sr : ed*sr], sr)

    ax.set(xlabel=None)
    ax.tick_params(axis='x', which="major", labelsize=17)

    if save_full:
        print("Saving figure..")
        plt.savefig(os.path.join(save_path, "{}_full_spec.jpg".format(prefix)), bbox_inches="tight")  

# # %%
# def tab_init():
#     """
#     Initializes and returns the state of Gradio columns for tab switching.

#     This function is designed to be used as a callback for Gradio interface buttons. It manages
#     the visibility of two columns, typically used in a tab-like interface. When invoked, it sets 
#     the first column to invisible and the second column to visible, facilitating a switch between 
#     tabs in the Gradio interface.

#     Returns:
#     - List[gr.Column, gr.Column]: A list containing two Gradio Column objects. The first Column 
#       object is set to invisible, and the second Column object is set to visible.
#     """
#     return [gr.Column(visible=False), gr.Column(visible=True)]
