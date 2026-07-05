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
        self.dataset = SPEECHCOMMANDS(
            root=DATA_ROOT,
            download=True,
            subset=subset,
        )

        self.label_to_index = {
            label: index for index, label in enumerate(TARGET_WORDS)
        }

        self.samples = []

        for i in range(len(self.dataset)):
            waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[i]

            if label in TARGET_WORDS:
                self.samples.append(i)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.samples)

    def _resample_if_needed(self, waveform, sample_rate):
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=SAMPLE_RATE,
            )
            waveform = resampler(waveform)
        return waveform

    def _mix_down_if_needed(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _cut_or_pad(self, waveform):
        if waveform.shape[1] > NUM_SAMPLES:
            waveform = waveform[:, :NUM_SAMPLES]

        if waveform.shape[1] < NUM_SAMPLES:
            missing_samples = NUM_SAMPLES - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, missing_samples))

        return waveform

    def __getitem__(self, index):
        dataset_index = self.samples[index]

        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[
            dataset_index
        ]

        waveform = self._resample_if_needed(waveform, sample_rate)
        waveform = self._mix_down_if_needed(waveform)
        waveform = self._cut_or_pad(waveform)

        spectrogram = self.mel_transform(waveform)
        spectrogram = self.db_transform(spectrogram)

        label_index = self.label_to_index[label]

        return spectrogram, label_index


def get_dataloaders():
    train_dataset = FilteredSpeechCommands(subset="training")
    val_dataset = FilteredSpeechCommands(subset="validation")
    test_dataset = FilteredSpeechCommands(subset="testing")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
