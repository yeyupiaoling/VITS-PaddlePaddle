import paddle
from paddle.nn import functional as F
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_paddle(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return paddle.log(x=paddle.clip(x=x, min=clip_val) * C)


def dynamic_range_decompression_paddle(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return paddle.exp(x) / C


def spectral_normalize_paddle(magnitudes):
    output = dynamic_range_compression_paddle(magnitudes)
    return output


def spectral_de_normalize_paddle(magnitudes):
    output = dynamic_range_decompression_paddle(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_paddle(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    # if paddle.min(y) < -1.:
    #     print('min value is ', paddle.min(y))
    # if paddle.max(y) > 1.:
    #     print('max value is ', paddle.max(y))

    global hann_window
    wnsize_dtype = str(win_size)
    if wnsize_dtype not in hann_window:
        hann_window[wnsize_dtype] = paddle.audio.functional.get_window(window='hann', win_length=win_size,
                                                                       dtype=y.dtype)
    y = F.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
              mode='reflect', data_format="NCL")
    y = y.squeeze(1)

    spec = paddle.signal.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                              window=hann_window[wnsize_dtype],
                              center=center, pad_mode='reflect', normalized=False, onesided=True)
    spec = paddle.as_real(spec)
    spec = paddle.norm(spec, p=2, axis=-1)
    return spec


def spec_to_mel_paddle(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = paddle.to_tensor(mel, dtype=spec.dtype)
    spec = paddle.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_paddle(spec)
    return spec


def mel_spectrogram_paddle(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # if paddle.min(y) < -1.:
    #     print('min value is ', paddle.min(y))
    # if paddle.max(y) > 1.:
    #     print('max value is ', paddle.max(y))

    global mel_basis, hann_window
    fmax_dtype = str(fmax) + '_' + str(y.dtype)
    wnsize_dtype = str(win_size) + '_' + str(y.dtype)
    if fmax_dtype not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype] = paddle.to_tensor(mel, dtype=y.dtype)
    if wnsize_dtype not in hann_window:
        hann_window[wnsize_dtype] = paddle.audio.functional.get_window(window='hann', win_length=win_size,
                                                                       dtype=y.dtype)

    y = F.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect',
              data_format='NCL')
    y = y.squeeze(1)

    spec = paddle.signal.stft(y.astype(paddle.float32), n_fft, hop_length=hop_size, win_length=win_size,
                              window=hann_window[wnsize_dtype], center=center, pad_mode='reflect',
                              normalized=False, onesided=True)
    spec = paddle.as_real(spec)
    spec = paddle.norm(spec, p=2, axis=-1)

    spec = paddle.matmul(mel_basis[fmax_dtype], spec)
    spec = spectral_normalize_paddle(spec)

    return spec
