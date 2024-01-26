# %%
import numpy as np
from transformers import AutoModel

import torch
import torch.nn.functional as F
from torch import nn

from .htsat import HTSATWrapper

# %%
def get_audio_encoder(name: str):
    """
    Retrieves the audio encoder class based on the specified name.

    Args:
    - name (str): Name of the desired audio encoder.

    Returns:
    - A class corresponding to the specified audio encoder.

    Raises:
    - Exception: If the audio encoder name is incorrect or not supported.
    """
    # Currently only HTSAT is supported
    if name == "HTSAT":
        return HTSATWrapper
    else:
        raise Exception('The audio encoder name {} is incorrect or not supported'.format(name))

# %%
class Projection(nn.Module):
    """
    A projection layer module.

    Attributes:
    - linear1 (nn.Linear): First linear layer.
    - linear2 (nn.Linear): Second linear layer.
    - layer_norm (nn.LayerNorm): Layer normalization.
    - drop (nn.Dropout): Dropout layer.

    Args:
    - d_in (int): Dimension of the input.
    - d_out (int): Dimension of the output.
    - p (float): Dropout probability.
    """
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name:str, d_in: int, d_out: int, sample_rate: int, window_size: int,
            hop_size: int, mel_bins: int, fmin: int, fmax: int, classes_num: int) -> None:
        super().__init__()

        audio_encoder = get_audio_encoder(audioenc_name)

        self.base = audio_encoder(
            sample_rate, window_size,
            hop_size, mel_bins, fmin, fmax,
            classes_num, d_in)

        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output

class TextEncoder(nn.Module):
    def __init__(self, d_out: int, text_model: str, transformer_embed_dim: int) -> None:
        super().__init__()
        self.text_model = text_model
        self.base = AutoModel.from_pretrained(text_model)

        if 'clip' in text_model:
            self.clip_text_projection = self.base.text_projection
            self.base = self.base.text_model
            if 'base' in text_model:
                transformer_embed_dim = 512
        
        self.projection = Projection(transformer_embed_dim, d_out)

    def forward(self, x):
        if 'clip' in self.text_model:
            pooled_output = self.base(**x)[1] # get pooled output
            out = self.clip_text_projection(pooled_output)  # get CLS token output
        elif 'gpt' in self.text_model:
            batch_size = x['input_ids'].shape[0]
            hidden_states = self.base(**x)[0] # (batch_size=4, seq_len, 768)

            sequence_lengths = torch.ne(x['input_ids'], 0).sum(-1) - 1 # tensor([13, 14, 18, 17])
            out = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths] # [batch_size, 768] = [4, 768]
        else:
            out = self.base(**x)[0]
            out = out[:, 0, :]  # get CLS token output
        
        projected_vec = self.projection(out)

        return projected_vec

class CLAP(nn.Module):
    """
    CLAP (Contrastive Language-Audio Pretraining) Model.

    A PyTorch module for cross-modal audio-text representation learning.

    Args:
    - audioenc_name (str): Name of the audio encoder.
    - sample_rate (int): Sample rate of audio.
    - window_size (int): Window size for the audio encoder.
    - hop_size (int): Hop size for the audio encoder.
    - mel_bins (int): Number of Mel bins.
    - fmin (int): Minimum frequency.
    - fmax (int): Maximum frequency.
    - classes_num (int): Number of classes for classification.
    - out_emb (int): Dimension of the output embeddings.
    - text_model (str): Name or path of the pretrained text model.
    - transformer_embed_dim (int): Embedding dimension of the transformer.
    - d_proj (int): Dimension of the projection.
    """
    def __init__(self,
                # audio
                audioenc_name: str = "HTSAT",
                sample_rate: int = 44100, 
                window_size: int = 1024, 
                hop_size: int = 320, 
                mel_bins: int = 64, 
                fmin: int = 50, 
                fmax: int = 8000, 
                classes_num: int = 527, 
                out_emb: int = 768,
                # text
                text_model: str = 'openai/clip-vit-base-patch16',
                transformer_embed_dim: int = 512,
                # common
                d_proj: int = 1024,
                ):
        super().__init__()

        # Define audio encoder
        self.audio_encoder = AudioEncoder(
            audioenc_name, out_emb, d_proj,
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
        # Define text encoder
        self.caption_encoder = TextEncoder(
            d_proj, text_model, transformer_embed_dim
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, audio, text):
        audio_embed, _ = self.audio_encoder(audio)
        caption_embed = self.caption_encoder(text)

        return caption_embed, audio_embed, self.logit_scale.exp()

# %%
