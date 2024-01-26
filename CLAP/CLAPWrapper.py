import warnings
warnings.filterwarnings("ignore")

import os
import random
import math
import collections
import re
import numpy as np

from transformers import AutoTokenizer, logging

import torch
import torchaudio
import torchaudio.transforms as T

from .clap import CLAP

logging.set_verbosity_error()


class CLAPWrapper():
    """
    A wrapper class for the CLAP (Contrastive Language-Audio Pretraining) model,
    facilitating audio and text embedding generation and similarity computation.
    """  

    def __init__(self, model_fp, use_cuda=False):
        """
        Initializes the CLAPWrapper class.

        Args:
        - model_fp (str): File path to the pre-trained CLAP model.
        - use_cuda (bool): Flag to enable CUDA for model computations.
        """
        self.np_str_obj_array_pattern = re.compile(r'[SaUO]')
        self.file_path = os.path.realpath(__file__)
        self.default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")
        self.text_model = 'openai/clip-vit-base-patch16'
        self.token_keys = ['input_ids', 'attention_mask']
        self.model_fp = model_fp
        self.use_cuda = use_cuda
        self.clap, self.tokenizer = self.load_clap()
        self.text_len = 77
        self.sampling_rate = 44100
        self.duration = 7

    def load_clap(self):
        """
        Loads the CLAP model and tokenizer. Currently the model loads the Jan23 version only.

        Returns:
        - The CLAP model and the tokenizer.
        """
        
        clap = CLAP()
        # Load pretrained weights for model
        model_state_dict = torch.load(self.model_fp, map_location=torch.device('cpu'))['model']
        clap.load_state_dict(model_state_dict, strict=False)
        clap.eval()  # set clap in eval mode
        tokenizer = AutoTokenizer.from_pretrained(self.text_model)
        if self.use_cuda and torch.cuda.is_available():
            clap = clap.cuda()

        return clap, tokenizer

    def default_collate(self, batch):
        """
        Collates a batch of data into tensors, handling various data types.

        Args:
        - batch: A batch of data.

        Returns:
        - Collated batch as tensors.
        """
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if self.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        self.default_collate_err_msg_format.format(elem.dtype))

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, (str, bytes)):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    'each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))

    def load_audio_into_tensor(self, audio_path, audio_duration, resample=False):
        """
        Loads an audio file into a tensor, with optional resampling.

        Args:
        - audio_path (str): Path to the audio file.
        - audio_duration (int): Duration for the audio clip.
        - resample (bool): Flag to apply resampling.

        Returns:
        - A tensor representation of the audio.
        """
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        resample_rate = self.sampling_rate
        if resample:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
        audio_time_series = audio_time_series.reshape(-1)

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if audio_duration*sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                        audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:audio_duration*sample_rate]
        else:
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - audio_duration*sample_rate)
            audio_time_series = audio_time_series[start_index:start_index +
                                                  audio_duration*sample_rate]
        return torch.FloatTensor(audio_time_series)

    def preprocess_audio(self, audio_files, resample):
        """
        Loads a list of audio files and returns raw audio.

        Args:
        - audio_files (list): A list of audio file paths.
        - resample (bool): Flag to apply resampling.

        Returns:
        - A batch of raw audio tensors.
        """
        audio_tensors = []
        for audio_file in audio_files:
            audio_tensor = self.load_audio_into_tensor(
                audio_file, self.duration, resample)
            audio_tensor = audio_tensor.reshape(
                1, -1).cuda() if self.use_cuda and torch.cuda.is_available() else audio_tensor.reshape(1, -1)
            audio_tensors.append(audio_tensor)
        return self.default_collate(audio_tensors)

    def preprocess_text(self, text_queries):
        
        """
        Load list of class labels and return tokenized text.

        Args:
        - text_queries (list): A list of text queries.

        Returns:
        - A batch of tokenized and preprocessed text tensors.
        """
        tokenized_texts = []
        for ttext in text_queries:
            tok = self.tokenizer.encode_plus(
                text=ttext, add_special_tokens=True, max_length=self.text_len, pad_to_max_length=True, return_tensors="pt")
            for key in self.token_keys:
                tok[key] = tok[key].reshape(-1).cuda() if self.use_cuda and torch.cuda.is_available() else tok[key].reshape(-1)
            tokenized_texts.append(tok)
        return self.default_collate(tokenized_texts)

    def get_text_embeddings(self, class_labels):
        """
        Generates text embeddings for a list of class labels.

        Args:
        - class_labels (list): A list of class labels.

        Returns:
        - Text embeddings for the given class labels.
        """
        preprocessed_text = self.preprocess_text(class_labels)
        return self._get_text_embeddings(preprocessed_text)

    def get_audio_embeddings(self, audio_files, resample):
        """
        Generates audio embeddings for a list of audio files.

        Args:
        - audio_files (list): A list of audio file paths.
        - resample (bool): Flag to apply resampling.

        Returns:
        - Audio embeddings for the given audio files.
        """
        preprocessed_audio = self.preprocess_audio(audio_files, resample)
        return self._get_audio_embeddings(preprocessed_audio)

    def _get_text_embeddings(self, preprocessed_text):
        """
        Generates text embeddings from preprocessed text data.

        Args:
        - preprocessed_text: Preprocessed text data.

        Returns:
        - Text embeddings.
        """
        with torch.no_grad():
            return self.clap.caption_encoder(preprocessed_text)

    def _get_audio_embeddings(self, preprocessed_audio):
        """
        Generates audio embeddings from preprocessed audio data.

        Args:
        - preprocessed_audio: Preprocessed audio data.

        Returns:
        - Audio embeddings.
        """
        with torch.no_grad():
            preprocessed_audio = preprocessed_audio.reshape(
                preprocessed_audio.shape[0], preprocessed_audio.shape[2])
            #Append [0] the audio emebdding, [1] has output class probabilities
            return self.clap.audio_encoder(preprocessed_audio)[0]

    def _generic_batch_inference(self, func, *args):
        """
        Processes audio and/or text data in batches.

        Args:
        - func: The function to apply for batch processing.
        - *args: Arguments for the function.

        Returns:
        - Processed batch output from the given function.
        """
        input_tmp = args[0]
        batch_size = args[-1]
        # args[0] has audio_files, args[1] has class_labels
        inputs = [args[0], args[1]] if len(args) == 3 else [args[0]]
        args0_len = len(args[0])
        # compute text_embeddings once for all the audio_files batches
        if len(inputs) == 2:
            text_embeddings = self.get_text_embeddings(args[1])
            inputs = [args[0], args[1], text_embeddings]
        dataset_idx = 0
        for _ in range(math.ceil(args0_len/batch_size)):
            next_batch_idx = dataset_idx + batch_size
            # batch size is bigger than available audio/text items
            if next_batch_idx >= args0_len:
                inputs[0] = input_tmp[dataset_idx:]
                return func(*tuple(inputs))
            else:
                inputs[0] = input_tmp[dataset_idx:next_batch_idx]
                yield func(*tuple(inputs))
            dataset_idx = next_batch_idx

    def get_audio_embeddings_per_batch(self, audio_files, batch_size):
        """
        Generates audio embeddings for audio files in batches.

        Args:
        - audio_files (list): A list of audio file paths.
        - batch_size (int): Size of each batch for processing.

        Returns:
        - Audio embeddings in batches.
        """
        return self._generic_batch_inference(self.get_audio_embeddings, audio_files, batch_size)

    def get_text_embeddings_per_batch(self, class_labels, batch_size):
        """
        Generates text embeddings for class labels in batches.

        Args:
        - class_labels (list): A list of class labels.
        - batch_size (int): Size of each batch for processing.

        Returns:
        - Text embeddings in batches.
        """
        return self._generic_batch_inference(self.get_text_embeddings, class_labels, batch_size)

    def compute_similarity(self, audio_embeddings, text_embeddings):
        """
        Computes similarity scores between audio and text embeddings.

        Args:
        - audio_embeddings: Embeddings of the audio data.
        - text_embeddings: Embeddings of the text data.

        Returns:
        - Similarity scores between each pair of audio and text embeddings.
        """
        audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
        text_embeddings = text_embeddings/torch.norm(text_embeddings, dim=-1, keepdim=True)
    
        logit_scale = self.clap.logit_scale.exp()
        similarity = logit_scale*text_embeddings @ audio_embeddings.T
        return similarity.T

    def classify_audio_files_per_batch(self, audio_files, class_labels, batch_size):
        """
        Classifies audio files using given class labels in batches.

        Args:
        - audio_files (list): A list of audio file paths.
        - class_labels (list): A list of class labels.
        - batch_size (int): Size of each batch for processing.

        Returns:
        - Classification results for each audio recording and class label in batches.
        """
        return self._generic_batch_inference(self.classify_audio_files, audio_files, class_labels, batch_size)