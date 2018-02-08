import numpy as np
from layer import *
from super_layer import *
from optimizer import *
from initialize import *
from utils import *

all_layers = {'linear_forward': linear_forward,
              'linear_backward': linear_backward,
              'linear_relu_forward': linear_relu_forward,
              'linear_relu_backward': linear_relu_backward,
              'conv_maxpool_forward': conv_maxpool_forward,
              'conv_maxpool_backward': conv_maxpool_backward
              }

loss_method = {'softmax_cross_entropy': softmax_cross_entropy}
init_method = {'xavier': xavier, 'conv_xavier': conv_xavier}


class NN(object):
    def __init__(self, structure,
                 loss_function='softmax_cross_entropy', optimizer='sgd', initializer='xavier',
                 input_dim=784, input_channel=1, output_dim=10, hidden_dim=32, conv_filter=[3, 3, 8],
                 padding='same', stride=1, max_pool=[2, 2], max_pool_stride=2):

        self.structure = structure
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.initializer = initializer
        self.len = len(structure)
        self.conv_filter = conv_filter
        self.padding = padding
        self.stride = stride
        self.max_pool = max_pool
        self.max_pool_stride = 1
        self.input_dim = input_dim
        self.input_channel = input_channel
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.params = {}
        self.grads = {}
        self.layer_cache = {}
        self.initial_parameters()



    def initial_parameters(self):

        if self.len == 1:
            self.params['w0'] = init_method[self.initializer](self.input_dim, self.output_dim)
            self.params['b0'] = np.zeros(self.output_dim)
        else:
            tmp_rec_dim = self.input_dim
            tmp_channel = self.input_channel
            tmp_dim = tmp_rec_dim * tmp_channel
            for layer_index, layer_name in enumerate(self.structure[:-1]):
                if layer_name == 'conv_maxpool':
                    self.params['w%d' % layer_index] = init_method['conv_' + self.initializer](self.conv_filter,
                                                                                               tmp_channel)
                    self.params['b%d' % layer_index] = np.zeros(self.conv_filter[2])
                    tmp_channel = self.conv_filter[2]
                    tmp_rec_dim = int(tmp_rec_dim / 4)
                    tmp_dim = tmp_rec_dim * tmp_channel
                else:
                    self.params['w%d' % layer_index] = init_method[self.initializer](tmp_dim, self.hidden_dim)
                    self.params['b%d' % layer_index] = np.zeros(self.hidden_dim)
                    tmp_dim = self.hidden_dim

            self.params['w%d' % (self.len - 1)] = init_method[self.initializer](self.hidden_dim, self.output_dim)
            self.params['b%d' % (self.len - 1)] = np.zeros(self.output_dim)


    def forward_backward(self, x, y):
        tem_hidden_forw = x
        for layer_index, layer_name in enumerate(self.structure):
            # print(self.structure)
            layer_type = layer_name + '_forward'

            tem_hidden_forw, tem_hidden_cache = all_layers[layer_type](tem_hidden_forw, \
                                                                       self.params['w%d' % layer_index], \
                                                                       self.params['b%d' % layer_index])
            self.layer_cache[layer_index] = tem_hidden_cache

        loss, loss_grad_last_layer = loss_method[self.loss_function](tem_hidden_forw, y)

        tem_hidden_grad = loss_grad_last_layer
        for layer_index, layer_name in reversed(list(enumerate(self.structure))):
            layer_type = layer_name + '_backward'
            tem_hidden_grad, self.grads['w%d' % layer_index], self.grads['b%d' % layer_index] \
                = all_layers[layer_type](tem_hidden_grad, self.layer_cache[layer_index])

        return loss

    def update(self, learning_rate):
        for key, val in self.params.items():
            self.params[key] = sgd(self.params[key], self.grads[key], learning_rate)

    def train(self, x, y, learning_rate):
        loss = self.forward_backward(x, y)
        self.update(learning_rate)
        return loss

    def predict(self, x):
        tem_hidden_forw = x
        for layer_index, layer_name in enumerate(self.structure):
            layer_type = layer_name + '_forward'

            tem_hidden_forw, tem_hidden_cache = all_layers[layer_type](tem_hidden_forw, \
                                                                       self.params['w%d' % layer_index], \
                                                                       self.params['b%d' % layer_index])
        y_pred = np.argmax(tem_hidden_forw, axis=1)
        return y_pred




