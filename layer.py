import numpy as np


def linear_forward(x, w, b):
    out = x.dot(w) + b
    cache = (x, w, b)
    return out, cache


def linear_backward(derivative_input, cache):
    ### input:
    ### derivative_input: N*M (batch * output (forward) so in this case, it's batch_shape * incoming_derivative_dimension)
    ### cache: x,w,b
    ### M: output dimension (forward, or incoming_derivative_dimension)
    ### D:(shape of each data) in this case (28*28)
    ### N:mini_batch_size
    ### w:D*M (weight)
    ### b:M (bias)
    ### x:N*D

    ### output:
    ### d_in/d_x: N*D
    ### d_in/d_w: D*M
    ### d_in/d_b: M

    x, w, b = cache

    dx = derivative_input.dot(w.T)
    dw = x.T.dot(derivative_input)
    db = np.sum(derivative_input, axis=0)

    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(derivative_input, cache):
    dx = derivative_input
    dx[cache < 0] = 0
    return dx


def conv_forward(x, w, b, stride=1, padding='same'):
    ### input
    ### x=-1*channel*28*28 batch_size*channel*d1*d2
    ### w=channel_out*channel_in*3*3
    ### b= channel_out

    batch_size = x.shape[0]
    dim_1 = x.shape[2]
    dim_2 = x.shape[3]
    channel_out = w.shape[0]
    filter_dim1 = w.shape[2]
    filter_dim2 = w.shape[3]

    assert padding == 'same'

    pad = int((filter_dim1 - 1) / 2)
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    dim1_out = int((dim_1 - filter_dim1 + 2 * pad) / stride + 1)
    dim2_out = int((dim_2 - filter_dim2 + 2 * pad) / stride + 1)

    out = np.zeros((batch_size, channel_out, dim1_out, dim2_out))

    for i in range(dim1_out):
        for j in range(dim2_out):
            x_local = x_pad[:, :, i * stride:i * stride + filter_dim1, j * stride:j * stride + filter_dim2]
            w_expand = np.expand_dims(w, axis=0)
            x_local_expand = np.expand_dims(x_local, axis=1)
            out[:, :, i, j] = np.sum(x_local_expand * w_expand, axis=(2, 3, 4)) + b

    cache = (x, w, b, stride, padding, pad)
    return out, cache


def conv_backward(derivative_input, cache):
    x, w, b, stride, padding, pad = cache
    dx = np.zeros_like(x)
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)

    db = np.sum(derivative_input, axis=(0, 2, 3))

    assert padding == 'same'

    dim_1 = x.shape[2]
    dim_2 = x.shape[3]
    filter_dim1 = w.shape[2]
    filter_dim2 = w.shape[3]
    dim1_out = int((dim_1 - filter_dim1 + 2 * pad) / stride + 1)
    dim2_out = int((dim_2 - filter_dim2 + 2 * pad) / stride + 1)

    for i in range(dim1_out):
        for j in range(dim2_out):
            x_local = x_pad[:, :, i * stride:i * stride + filter_dim1, j * stride:j * stride + filter_dim2]

            derivative_input_expand = np.expand_dims(derivative_input, axis=2)
            x_local_expand = np.expand_dims(x_local, axis=1)
            dw += np.sum(derivative_input_expand[:, :, :, i, j][:, :, :, None, None] * x_local_expand, axis=0)

            w_expand = np.expand_dims(w, axis=0)
            derivative_input_expand = np.expand_dims(derivative_input, axis=2)
            dx_pad[:, :, i * stride:i * stride + filter_dim1, j * stride:j * stride + filter_dim2] += np.sum(
                w_expand * derivative_input_expand[:, :, :, i, j][:, :, :, None, None], axis=1)

    dx = dx_pad[:, :, pad:-pad, pad:-pad]

    return dx, dw, db


def max_pool_forward(x, pool_kernel=[2, 2], stride=2):
    batch_size = x.shape[0]
    channel_in = x.shape[1]
    dim_1 = x.shape[2]
    dim_2 = x.shape[3]

    k_dim1 = pool_kernel[0]
    k_dim2 = pool_kernel[1]

    out_dim1 = int((dim_1 - k_dim1) / stride + 1)
    out_dim2 = int((dim_2 - k_dim2) / stride + 1)

    out = np.zeros((batch_size, channel_in, out_dim1, out_dim2))

    for i in range(out_dim1):
        for j in range(out_dim2):
            x_local = x[:, :, i * stride:i * stride + k_dim1, j * stride:j * stride + k_dim2]
            out[:, :, i, j] = np.max(x_local, axis=(2, 3))

    cache = (x, pool_kernel, stride, out)
    return out, cache


def max_pool_backward(derivative_input, cache):
    x, pool_kernel, stride, out = cache

    dim_1 = x.shape[2]
    dim_2 = x.shape[3]

    k_dim1 = pool_kernel[0]
    k_dim2 = pool_kernel[1]

    out_dim1 = int((dim_1 - k_dim1) / stride + 1)
    out_dim2 = int((dim_2 - k_dim2) / stride + 1)

    dx = np.zeros_like(x)

    for i in range(out_dim1):
        for j in range(out_dim2):
            x_local = x[:, :, i * stride:i * stride + k_dim1, j * stride:j * stride + k_dim2]
            max_location = ((out[:, :, i, j][:, :, None, None]) == x_local)
            # print('location:',np.shape(max_location))
            dx[:, :, i * stride:i * stride + k_dim1, j * stride:j * stride + k_dim2] += max_location * (
            derivative_input[:, :, i, j][:, :, None, None])

    return dx


def softmax_cross_entropy(x, y):
    ### input
    ### x:N*K
    ### y:K
    batch_size = x.shape[0]

    ### forwards
    eps = 1e-9
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(np.log(np.sum(probs * y, axis=1)+eps))

    ###  backwards
    dx = probs.copy()
    dx = dx - y

    return loss, dx