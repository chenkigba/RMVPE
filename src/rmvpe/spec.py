import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from librosa.filters import mel
from librosa.util import pad_center
from scipy.signal import get_window
from torch.autograd import Variable
import torch


class STFT(nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length, hop_length, win_length=None, window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window

        self.forward_transform = None
        scale = self.filter_length / self.hop_length

        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        if window is not None:
            assert (filter_length >= self.win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, self.win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            # window the bases
            forward_basis *= torch.FloatTensor(fft_window)

        self.register_buffer('forward_basis', forward_basis)

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(input_data.unsqueeze(1),
                           (int(self.filter_length // 2), int(self.filter_length // 2), 0, 0), mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)

        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)

        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def forward(self, input_data):
        return self.transform(input_data)


class MelSpectrogram(nn.Module):
    def __init__(self, n_mel_channels, sampling_rate, filter_length, hop_length, win_length, mel_fmin, mel_fmax):
        super(MelSpectrogram, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        self.stft = STFT(filter_length, hop_length, win_length)

        mel_basis = mel(sr=self.sampling_rate, n_fft=self.filter_length, n_mels=n_mel_channels,
                        fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = torch.log(torch.clamp(magnitudes, min=1e-5))
        return output

    def forward(self, y):
        assert (torch.min(y) >= -1)
        assert (torch.max(y) <= 1)
        magnitudes, phases = self.stft.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

