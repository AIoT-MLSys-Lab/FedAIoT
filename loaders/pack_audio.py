import random

import numpy as np
import torch


def temporal_sampling(spectrogram, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(0, spectrogram.shape[1] - 1, num_samples).long()
    spectrogram = torch.index_select(spectrogram, 1, index)
    return spectrogram


def get_start_end_idx(audio_size, clip_size, clip_idx, num_clips, start_sample=0):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        audio_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(audio_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = np.linspace(0, delta, num=num_clips)[clip_idx]
    end_idx = start_idx + clip_size - 1
    return start_sample + start_idx, start_sample + end_idx


def pack_audio(audio_dataset, video_record, temporal_sample_index, sampling_rate=24000, clip_secs=1.999, n_ensemble=5):
    samples = audio_dataset[video_record.video_id][()]
    start_idx, end_idx = get_start_end_idx(
        video_record.num_audio_samples,
        int(round(sampling_rate * clip_secs)),
        temporal_sample_index,
        n_ensemble,
        start_sample=video_record.start_audio_sample
    )
    spectrogram = _extract_sound_feature(
        samples,
        video_record,
        int(start_idx),
        int(end_idx),
        clip_secs
    )
    return spectrogram


def _log_specgram(
        audio,
        window_size=10,
        step_size=5,
        eps=1e-6,
        sampling_rate=24000
):
    nperseg = int(round(window_size * sampling_rate / 1e3))
    noverlap = int(round(step_size * sampling_rate / 1e3))
    from librosa import stft, filters

    # Mel-Spectrogram
    spec = stft(
        audio,
        n_fft=2048,
        window='hann',
        hop_length=noverlap,
        win_length=nperseg,
        pad_mode='constant'
    )
    mel_basis = filters.mel(
        sr=sampling_rate,
        n_fft=2048,
        n_mels=128,
        htk=True,
        norm=None
    )
    mel_spec = np.dot(mel_basis, np.abs(spec))

    # Log-Mel-Spectrogram
    log_mel_spec = np.log(mel_spec + eps)
    return log_mel_spec.T


def _extract_sound_feature(samples, video_record, start_idx, end_idx, clip_duration, sampling_rate=24000):
    if video_record.num_audio_samples < int(round(sampling_rate * clip_duration)):
        samples = samples[video_record.start_audio_sample:video_record.end_audio_sample]
    else:
        samples = samples[start_idx:end_idx]
    spectrogram = _log_specgram(samples,
                                window_size=10,
                                step_size=5
                                )
    if spectrogram.shape[0] < 400:
        num_timesteps_to_pad = 400 - spectrogram.shape[0]
        spectrogram = np.pad(spectrogram, ((0, num_timesteps_to_pad), (0, 0)), 'edge')
    return torch.tensor(spectrogram).unsqueeze(0)
