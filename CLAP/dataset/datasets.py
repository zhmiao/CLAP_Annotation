# %%
import os

# %%
import numpy as np
from tqdm import tqdm

# %%
from torch.utils.data import Dataset
import torchaudio
import torch
import torch.nn.functional as F

__all__ = ["Inference_DS", "Batch_Inference_DS"]

# %%
def softmax(x, T=1):
    """
    Computes the softmax of the input array.

    Args:
    - x (np.ndarray): Input array.
    - T (float, optional): Temperature parameter for softmax. Default is 1.

    Returns:
    - np.ndarray: Softmax of the input array.
    """
    e_x = np.exp(x / T)
    return e_x / e_x.sum(axis=1).reshape(-1, 1)

class Inference_DS(Dataset):
    """
    A custom PyTorch Dataset for handling audio inference.

    This dataset is designed to segment an audio waveform into smaller chunks 
    for processing in an inference pipeline.

    Attributes:
    - wav (Tensor): Waveform of the audio.
    - sr (int): Original sampling rate of the audio.
    - seg_size (int): Size of each segment in seconds.
    - target_sr (int): Target sampling rate for resampling.
    - wav_segs (list): List of segmented waveforms.
    - wav_length (int): Length of the audio in seconds.
    - num_segs (int): Number of segments in the audio.
    - step_size (int): Step size for segmentation in sample points.

    Args:
    - wav (Tensor): Waveform of the audio.
    - sr (int): Original sampling rate of the audio.
    - seg_size (int): Size of each segment in seconds.
    - target_sr (int): Target sampling rate for resampling.
    """
    def __init__(self, wav, sr, seg_size=5, target_sr=44100):

        self.wav = wav
        self.sr = sr
        self.seg_size = seg_size
        self.target_sr = target_sr

        self.wav_segs = []
        self.wav_length = int(len(self.wav) / self.sr)
        self.num_segs = int(np.ceil(self.wav_length / self.seg_size))
        self.step_size = self.seg_size * self.sr

        for i in range(self.num_segs):
            st = self.step_size * i
            if (st + self.step_size) <= (len(wav) - 1):
                self.wav_segs.append(wav[st : st + self.step_size])
            else:
                seg = wav[st:]
                seg_mean = seg.mean()
                seg = torch.cat((seg, torch.tensor([seg_mean for _ in range((st + self.step_size) - len(wav))])))
                self.wav_segs.append(seg)
                # self.wav_segs.append(wav[- self.step_size :])

    def __len__(self):
        return self.num_segs

    def __getitem__(self, idx):
        x = self.wav_segs[idx]
        if self.sr != self.target_sr:
            transform = torchaudio.transforms.Resample(self.sr, self.target_sr)
            x = transform(x)

        return torch.FloatTensor(x).reshape((1, -1))


class Batch_Inference_DS(Dataset):

    def __init__(self, wav_list, seg_size=6, target_sr=44100):

        self.data = []
        self.sts = []
        self.wav_lens = []

        self.wav_list = wav_list

        self.seg_size = seg_size
        self.target_sr = target_sr
        
        print("Loading data...")
        for f in tqdm(self.wav_list):

            wav, sr = torchaudio.load(f)
            wav = wav.reshape(-1)

            wav_length = int(len(wav) / sr)
            num_segs = int(np.ceil(wav_length / seg_size))

            for i in range(num_segs):
                self.data.append(f)
                self.sts.append(self.seg_size * i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        wav, sr = torchaudio.load(self.data[idx])
        wav = wav.reshape(-1)
        st = self.sts[idx] * sr

        step_size = self.seg_size * sr
        
        if (st + step_size) <= (len(wav) - 1):
            x = wav[st : st + step_size]
        else:
            x = wav[st:]
            x_mean = x.mean()
            x = torch.cat((x, torch.tensor([x_mean for _ in range((st + step_size) - len(wav))])))

        if sr != self.target_sr:
            transform = torchaudio.transforms.Resample(sr, self.target_sr)
            x = transform(x)

        return torch.FloatTensor(x).reshape((1, -1))