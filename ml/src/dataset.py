# Disable torchcodec before importing torchaudio — torchcodec requires FFmpeg
# which is not available on this machine, so we fall back to soundfile instead
import os
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

from config import (
    TARGET_WORDS,
    SAMPLE_RATE,
    NUM_SAMPLES,
    N_MELS,
    BATCH_SIZE,
    DATA_ROOT,
)


class FilteredSpeechCommands(Dataset):
    def __init__(self, subset):
        # Download the Google Speech Commands dataset if not already present
        # subset can be "training", "validation", or "testing"
        self.dataset = SPEECHCOMMANDS(
            root=DATA_ROOT,
            download=True,
            subset=subset,
        )

        # Map each command word to a number (e.g. "yes" -> 0, "no" -> 1, ...)
        self.label_to_index = {
            label: index for index, label in enumerate(TARGET_WORDS)
        }

        # Collect only the indices of samples whose label is one of our 8 commands
        # We read the label from the file path (label/speaker_utterance.wav)
        # instead of loading audio, which would be very slow across 100k+ files
        self.samples = []
        for i, filepath in enumerate(self.dataset._walker):
            label = os.path.basename(os.path.dirname(filepath))
            if label in TARGET_WORDS:
                self.samples.append(i)

        # Converts raw waveform into a Mel spectrogram (a 2D image of sound)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
        )

        # Converts amplitude values to decibels so the range is more manageable
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        # Returns how many valid samples this split contains
        return len(self.samples)

    def _resample_if_needed(self, waveform, sample_rate):
        # If the audio was recorded at a different rate, convert it to 16,000 Hz
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=SAMPLE_RATE,
            )
            waveform = resampler(waveform)
        return waveform

    def _mix_down_if_needed(self, waveform):
        # If the audio has more than one channel (stereo), average them into mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _cut_or_pad(self, waveform):
        # Trim clips that are too long
        if waveform.shape[1] > NUM_SAMPLES:
            waveform = waveform[:, :NUM_SAMPLES]

        # Zero-pad clips that are too short so every clip is exactly 1 second
        if waveform.shape[1] < NUM_SAMPLES:
            missing_samples = NUM_SAMPLES - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, missing_samples))

        return waveform

    def __getitem__(self, index):
        # Look up the real position of this sample inside the full dataset
        dataset_index = self.samples[index]

        # Load the audio file from disk
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[
            dataset_index
        ]

        # Standardize the audio: correct sample rate, mono, fixed length
        waveform = self._resample_if_needed(waveform, sample_rate)
        waveform = self._mix_down_if_needed(waveform)
        waveform = self._cut_or_pad(waveform)

        # Convert waveform to a Mel spectrogram in decibels
        spectrogram = self.mel_transform(waveform)
        spectrogram = self.db_transform(spectrogram)

        # Convert the text label to its numeric index
        label_index = self.label_to_index[label]

        return spectrogram, label_index


def get_dataloaders():
    # Create separate datasets for training, validation, and testing
    train_dataset = FilteredSpeechCommands(subset="training")
    val_dataset = FilteredSpeechCommands(subset="validation")
    test_dataset = FilteredSpeechCommands(subset="testing")

    # Wrap each dataset in a DataLoader that handles batching and shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,   # Shuffle training data each epoch to avoid learning order
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # No shuffle needed for evaluation
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
