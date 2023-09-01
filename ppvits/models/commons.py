import math
from typing import Union

import paddle
from paddle.nn import functional as F


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (paddle.exp(2. * logs_p) + ((m_p - m_q) ** 2)) * paddle.exp(-2. * logs_q)
    return kl


def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = paddle.rand(shape) * 0.99998 + 0.00001
    return -paddle.log(-paddle.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.shape).astype(x.dtype)
    return g


def slice_segments(x, ids_str, segment_size=4):
    ret = paddle.zeros_like(x[:, :, :segment_size])
    for i in range(x.shape[0]):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        try:
            ret[i] = x[i, :, idx_str:idx_end]
        except RuntimeError:
            print("?")
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.shape
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (paddle.rand([b]) * ids_str_max).astype(paddle.int64)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(
        length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = paddle.arange(length, dtype=paddle.float32)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (num_timescales - 1))
    inv_timescales = min_timescale * paddle.exp(
        paddle.arange(num_timescales, dtype=paddle.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = paddle.concat([paddle.sin(scaled_time), paddle.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, channels % 2])
    signal = signal.reshape([1, channels, length])
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.shape
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.astype(x.dtype)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.shape
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return paddle.concat([x, signal.astype(x.dtype)], axis)


def subsequent_mask(length):
    mask = paddle.tril(paddle.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@paddle.jit.to_static()
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = paddle.tanh(in_act[:, :n_channels_int, :])
    s_act = paddle.nn.functional.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = paddle.arange(max_length, dtype=length.dtype)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = paddle.cumsum(duration, -1)

    cum_duration_flat = cum_duration.reshape([b * t_x])
    path = sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)
    path = path.reshape([b, t_x, t_y])
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose([0, 1, 3, 2]) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            # TODO
            p.grad.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def broadcast_shape(shp1, shp2):
    result = []
    for a, b in zip(shp1[::-1], shp2[::-1]):
        result.append(max(a, b))
    return result[::-1]


def masked_fill(xs: paddle.Tensor,
                mask: paddle.Tensor,
                value: Union[float, int]):
    # will be nan when value is `inf`.
    # mask = mask.astype(xs.dtype)
    # return xs * (1.0 - mask) + mask * value

    bshape = broadcast_shape(xs.shape, mask.shape)
    mask.stop_gradient = True
    # tmp = paddle.ones(shape=[len(bshape)], dtype='int32')
    # for index in range(len(bshape)):
    #     tmp[index] = bshape[index]
    mask = mask.broadcast_to(bshape)
    trues = paddle.full_like(xs, fill_value=value)
    xs = paddle.where(mask, trues, xs)
    return xs
