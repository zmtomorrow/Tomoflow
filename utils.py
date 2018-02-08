def flatten(x):
    batch_size = x.shape[0]
    x_flatten = x.reshape(batch_size, -1)
    return x_flatten

