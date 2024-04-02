# %%
import os

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
import torchaudio
import torch
import torch.nn.functional as F

# %%
from glob import glob

# %%
from utils import *

# %%
weights_path = 'weights/{}.pth'.format("CLAP_Jan23")
print("Loading CLAP model..")
clap = CLAPWrapper(weights_path, use_cuda=True)
# Load the weights
clap.load_clap()

# %%

# %%
wav_list = glob("./demo_data/**/*.wav", recursive=True)
wav_list = "\n".join(wav_list)

# %%

# %%
pos_prompts = "birds chirping;"
neg_prompts = "noise;"
batch_audio_detection(wav_list=wav_list, pos_prompts=pos_prompts, neg_prompts=neg_prompts,
                      output_spec=True, output_det=True, annotator=None, save_path="./temp/sample")

# %%
