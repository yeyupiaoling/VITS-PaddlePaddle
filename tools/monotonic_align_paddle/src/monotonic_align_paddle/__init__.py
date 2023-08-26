import numpy as np
import paddle

from .core import maximum_path_c


def maximum_path(neg_cent, mask):
    """Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    neg_cent = neg_cent.numpy().astype(np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.int32)

    t_t_max = mask.sum(1)[:, 0].numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].numpy().astype(np.int32)
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return paddle.to_tensor(path, dtype=neg_cent.dtype)
