import math
from typing import Tuple, Union, Optional

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn.utils import weight_norm, remove_weight_norm

from ppvits.models.commons import get_padding, fused_add_tanh_sigmoid_multiply
from ppvits.models.transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1


class LayerNorm(nn.Layer):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = paddle.create_parameter(shape=[channels], dtype=paddle.float32,
                                             default_initializer=paddle.nn.initializer.Assign(
                                                 paddle.ones(shape=[channels])))
        self.beta = paddle.create_parameter(shape=[channels], dtype=paddle.float32,
                                            default_initializer=paddle.nn.initializer.Assign(
                                                paddle.zeros(shape=[channels])))

    def forward(self, x):
        x = x.transpose([0, 2, 1])
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose([0, 2, 1])


class ConvReluNorm(nn.Layer):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.LayerList()
        self.norm_layers = nn.LayerList()
        self.conv_layers.append(nn.Conv1D(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1D(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1D(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(nn.Layer):
    """
    Dilated and Depth-Separable Convolution
    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.LayerList()
        self.convs_1x1 = nn.LayerList()
        self.norms_1 = nn.LayerList()
        self.norms_2 = nn.LayerList()
        for i in range(n_layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(nn.Conv1D(channels, channels, kernel_size,
                                            groups=channels, dilation=dilation, padding=padding))
            self.convs_1x1.append(nn.Conv1D(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class WN(paddle.nn.Layer):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert (kernel_size % 2 == 1)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = paddle.nn.LayerList()
        self.res_skip_layers = paddle.nn.LayerList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = paddle.nn.Conv1D(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = paddle.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = paddle.nn.Conv1D(hidden_channels, 2 * hidden_channels, kernel_size,
                                        dilation=dilation, padding=padding)
            in_layer = paddle.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = paddle.nn.Conv1D(hidden_channels, res_skip_channels, 1)
            res_skip_layer = paddle.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = paddle.zeros_like(x)
        n_channels_tensor = paddle.to_tensor(data=[self.hidden_channels], dtype=paddle.int32)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = paddle.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            paddle.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            paddle.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            paddle.nn.utils.remove_weight_norm(l)


class ResBlock1(nn.Layer):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.LayerList([
            weight_norm(nn.Conv1D(channels, channels, kernel_size, 1, dilation=dilation[0],
                                  padding=get_padding(kernel_size, dilation[0]),
                                  weight_attr=paddle.framework.ParamAttr(
                                      initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01)))),
            weight_norm(nn.Conv1D(channels, channels, kernel_size, 1, dilation=dilation[1],
                                  padding=get_padding(kernel_size, dilation[1]),
                                  weight_attr=paddle.framework.ParamAttr(
                                      initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01)))),
            weight_norm(nn.Conv1D(channels, channels, kernel_size, 1, dilation=dilation[2],
                                  padding=get_padding(kernel_size, dilation[2]),
                                  weight_attr=paddle.framework.ParamAttr(
                                      initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01))))
        ])

        self.convs2 = nn.LayerList([
            weight_norm(nn.Conv1D(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1),
                                  weight_attr=paddle.framework.ParamAttr(
                                      initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01)))),
            weight_norm(nn.Conv1D(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1),
                                  weight_attr=paddle.framework.ParamAttr(
                                      initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01)))),
            weight_norm(nn.Conv1D(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1),
                                  weight_attr=paddle.framework.ParamAttr(
                                      initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01))))
        ])

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(paddle.nn.Layer):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.LayerList([
            weight_norm(nn.Conv1D(channels, channels, kernel_size, 1, dilation=dilation[0],
                                  padding=get_padding(kernel_size, dilation[0]),
                                  weight_attr=paddle.framework.ParamAttr(
                                      initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01)))),
            weight_norm(nn.Conv1D(channels, channels, kernel_size, 1, dilation=dilation[1],
                                  padding=get_padding(kernel_size, dilation[1]),
                                  weight_attr=paddle.framework.ParamAttr(
                                      initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01))))
        ])

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Log(nn.Layer):
    """Log flow module."""

    def forward(self,
                x: paddle.Tensor,
                x_mask: paddle.Tensor,
                reverse: bool = False,
                eps: float = 1e-5,
                **kwargs
                ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
        """Calculate forward propagation.
        Args:
            x (Tensor):
                Input tensor (B, channels, T).
            x_mask (Tensor):
                Mask tensor (B, 1, T).
            reverse (bool):
                Whether to inverse the flow.
            eps (float):
                Epsilon for log.
        Returns:
            Tensor:
                Output tensor (B, channels, T).
            Tensor:
                Log-determinant tensor for NLL (B,) if not inverse.
        """
        if not reverse:
            y = paddle.log(paddle.clip(x, min=eps)) * x_mask
            logdet = paddle.sum(-y, [1, 2])
            return y, logdet
        else:
            x = paddle.exp(x) * x_mask
            return x


class Flip(nn.Layer):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = paddle.flip(x, [1])
        if not reverse:
            logdet = paddle.zeros([x.shape[0]]).astype(x.dtype)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Layer):
    """Elementwise affine flow module."""

    def __init__(self, channels: int):
        """Initialize ElementwiseAffineFlow module.
        Args:
            channels (int):
                Number of channels.
        """
        super().__init__()
        m = paddle.zeros([channels, 1])
        self.m = paddle.create_parameter(shape=m.shape,
                                         dtype=paddle.float32,
                                         default_initializer=paddle.nn.initializer.Assign(m))
        logs = paddle.zeros([channels, 1])
        self.logs = paddle.create_parameter(shape=logs.shape,
                                            dtype=paddle.float32,
                                            default_initializer=paddle.nn.initializer.Assign(logs))

    def forward(self,
                x: paddle.Tensor,
                x_mask: paddle.Tensor,
                reverse: bool = False,
                **kwargs
                ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
        """Calculate forward propagation.
        Args:
            x (Tensor):
                Input tensor (B, channels, T).
            x_mask (Tensor):
                Mask tensor (B, 1, T).
            reverse (bool):
                Whether to inverse the flow.
        Returns:
            Tensor:
                Output tensor (B, channels, T).
            Tensor:
                Log-determinant tensor for NLL (B,) if not inverse.
        """
        if not reverse:
            y = self.m + paddle.exp(self.logs) * x
            y = y * x_mask
            logdet = paddle.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * paddle.exp(-self.logs) * x_mask
            return x


class ResidualCouplingLayer(nn.Layer):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 p_dropout=0,
                 gin_channels=0,
                 mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1D(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout,
                      gin_channels=gin_channels)
        self.post = nn.Conv1D(hidden_channels, self.half_channels * (2 - mean_only), 1)
        # TODO
        # self.post.weight.zero_()
        # self.post.bias.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = paddle.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = paddle.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = paddle.zeros_like(m)

        if not reverse:
            x1 = m + x1 * paddle.exp(logs) * x_mask
            x = paddle.concat([x0, x1], 1)
            logdet = paddle.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * paddle.exp(-logs) * x_mask
            x = paddle.concat([x0, x1], 1)
            return x


class ConvFlow(nn.Layer):
    """Convolutional flow module."""

    def __init__(self, in_channels: int, filter_channels: int, kernel_size: int, n_layers: int, num_bins: int = 10,
                 tail_bound: float = 5.0, ):
        """Initialize ConvFlow module.
        Args:
            in_channels (int):
                Number of input channels.
            filter_channels (int):
                Number of hidden channels.
            kernel_size (int):
                Kernel size.
            n_layers (int):
                Number of layers.
            num_bins (int):
                Number of bins.
            tail_bound (float):
                Tail bound value.
        """
        super().__init__()
        self.half_channels = in_channels // 2
        self.hidden_channels = filter_channels
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        self.input_conv = nn.Conv1D(self.half_channels, filter_channels, 1, )
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.)
        self.proj = nn.Conv1D(filter_channels, self.half_channels * (num_bins * 3 - 1), 1, )

        weight = paddle.zeros(paddle.shape(self.proj.weight))
        self.proj.weight = paddle.create_parameter(shape=weight.shape,
                                                   dtype=paddle.float32,
                                                   default_initializer=paddle.nn.initializer.Assign(weight))

        bias = paddle.zeros(paddle.shape(self.proj.bias))
        self.proj.bias = paddle.create_parameter(shape=bias.shape,
                                                 dtype=paddle.float32,
                                                 default_initializer=paddle.nn.initializer.Assign(bias))

    def forward(
            self,
            x: paddle.Tensor,
            x_mask: paddle.Tensor,
            g: Optional[paddle.Tensor] = None,
            reverse: bool = False,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
        """Calculate forward propagation.
        Args:
            x (Tensor):
                Input tensor (B, channels, T).
            x_mask (Tensor):
                Mask tensor (B, 1, T).
            g (Optional[Tensor]):
                Global conditioning tensor (B, channels, 1).
            reverse (bool):
                Whether to inverse the flow.
        Returns:
            Tensor:
                Output tensor (B, channels, T).
            Tensor:
                Log-determinant tensor for NLL (B,) if not inverse.
        """
        x0, x1 = x.split(2, 1)
        h = self.input_conv(x0)
        h = self.convs(h, x_mask, g=g)
        # (B, half_channels * (bins * 3 - 1), T)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        # (B, half_channels, bins * 3 - 1, T) -> (B, half_channels, T, bins * 3 - 1)
        h = h.reshape([b, c, -1, t]).transpose([0, 1, 3, 2])

        denom = math.sqrt(self.hidden_channels)
        unnorm_widths = h[..., :self.num_bins] / denom
        unnorm_heights = h[..., self.num_bins:2 * self.num_bins] / denom
        unnorm_derivatives = h[..., 2 * self.num_bins:]

        x1, logdet_abs = piecewise_rational_quadratic_transform(inputs=x1,
                                                                unnormalized_widths=unnorm_widths,
                                                                unnormalized_heights=unnorm_heights,
                                                                unnormalized_derivatives=unnorm_derivatives,
                                                                inverse=reverse,
                                                                tails="linear",
                                                                tail_bound=self.tail_bound, )
        x = paddle.concat([x0, x1], 1) * x_mask
        logdet = paddle.sum(logdet_abs * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x
