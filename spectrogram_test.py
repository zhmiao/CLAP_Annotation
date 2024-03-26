# %%
import matplotlib.pyplot as plt
from maad import sound, util

# variables
nperseg = 1024  # window size for STFT
noverlap = 512  # window overlap for STFT
db_range = 90   # dB range, clip background noise
cmap = 'gray'   # colormap

# %%
import torchaudio
# wav, sr = torchaudio.load('demo_data/Recording_1_Segment_02.004.wav')
wav, sr = torchaudio.load('./SWcorner_encounter_9.wav')

# %%
wav = wav[:, sr*15:sr*19]

# %%
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
seg_size = 6
max_khz = 10.0
min_khz = 0.0

# %%
Sxx, _, _, ext = sound.spectrogram(wav[0], sr, nperseg=nperseg, noverlap=noverlap, flims=[min_khz * 1000, max_khz * 1000])
# Sxx, _, _, ext = sound.spectrogram(wav[0], sr, nperseg=1024, noverlap=512, flims=[min_khz * 1000, max_khz * 1000])
ext[-1] = max_khz
Sxx_db = util.power2dB(Sxx, db_range, 0)
# Whale
# Sxx_db = util.power2dB(Sxx, 94, 13)

# # %%
# fig, ax = plt.subplots(figsize=(15, 5))
fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(Sxx_db, extent=ext, origin="lower", cmap=cmap, 
          vmin=None, vmax=None)
ax.axis("tight")
fig.tight_layout()

plt.locator_params(axis='y', nbins=1)
plt.locator_params(axis='x', nbins=10)
