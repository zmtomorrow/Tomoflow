import numpy as np

def conv_xavier(filter_size, input_depth):
    fan_in = filter_size[0] * filter_size[1] * input_depth
    fan_out = filter_size[0] * filter_size[1] * filter_size[2]
    std = np.sqrt(2 / (fan_in + fan_out))
    initial = std * np.random.randn(filter_size[2], input_depth, filter_size[0], filter_size[1])
    return initial
#
#
# def xavier_init(input_dim, output_dim):
#     std = np.sqrt(2 / (input_dim + output_dim))
#     initial = std * np.random.randn(input_dim, output_dim)
#     # initial = np.random.normal(scale=std, size=(input_dim, output_dim))
#     return initial


def xavier(fan_in, fan_out):
    std = np.sqrt(2 / (fan_in + fan_out))
    initial = std * np.random.randn(fan_in, fan_out)
    return initial


