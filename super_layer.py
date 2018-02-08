from layer import *
from utils import *


def linear_relu_forward(x, w, b):
    if len(x.shape) == 4:
       x = flatten(x)
    h, linear_cache = linear_forward(x, w, b)
    out, relu_cache = relu_forward(h)
    cache = (linear_cache, relu_cache)
    return out, cache


def linear_relu_backward(derivative_input, cache):
    linear_cache, relu_cache = cache
    d_relu = relu_backward(derivative_input, relu_cache)
    dx, dw, db = linear_backward(d_relu, linear_cache)
    return dx, dw, db


def conv_maxpool_forward(x, w, b, conv_stride=1, conv_padding='same', pool_kernel=[2, 2], pool_stride=2):

    if len(x.shape) == 2:
        batch_size = x.shape[0]
        dim_x = x.shapex[1]
        side_len = int(np.sqrt(dim_x))
        x = x.reshape((batch_size, 1, side_len, side_len))

    h, conv_cache = conv_forward(x, w, b, conv_stride, conv_padding)
    out, maxpool_cache = max_pool_forward(h, pool_kernel, pool_stride)
    cache = (conv_cache, maxpool_cache)
    return out, cache


def conv_maxpool_backward(derivative_input, cache):
    conv_cache, maxpool_cache = cache
    if len(derivative_input.shape) != 4:
        _, _, _, out = maxpool_cache
        derivative_input=derivative_input.reshape(out.shape)

    d_maxpool = max_pool_backward(derivative_input, maxpool_cache)
    dx, dw, db = conv_backward(d_maxpool, conv_cache)
    return dx, dw, db
