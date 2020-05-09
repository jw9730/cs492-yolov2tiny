import os
import sys
import math
import networkx as nx
import numpy as np
import time

"""
The class DNNInferenceEngine will take a graph of DNN nodes to produce its computation result.
DNNGraphBuilder consists of several create functions that create and append a DNN node.
We treat ve types of DNN nodes; Conv2d, BiasAdd, MaxPool2D, BatchNorm, LeakyReLU.
Your job is to implement actual computations in the DNN nodes.
Every intermediate DNN node must contain in node and result.
The in node is a previous node that feeds a tensor to the current node through result.
"""


class DnnInferenceEngine(object):
    def __init__(self, graph):
        self.g = graph

    def run(self, tin):
        self.g.in_node.set_input(tin)
        out = {}
        currents = [self.g.in_node]
        done = set()
        while (len(currents) != 0):
            nexts = []
            for current in currents:
                skip_current = False
                predecessors = self.g.G.predecessors(current)
                for predecessor in predecessors:
                    if predecessor not in done:
                        nexts.append(predecessor)
                        skip_current = True
                if skip_current:
                    continue
                current.run()
                if self.g.is_out_node(current):
                    out = current.result
                done.add(current)
                for successor in self.g.G.successors(current):
                    nexts.append(successor)
            currents = nexts
        return out


class DnnGraphBuilder(object):
    def __init__(self):
        self.G = nx.DiGraph()
        self.name_num = {"conv2d": 0,
                         "bias_add": 0,
                         "max_pool2d": 0,
                         "batch_norm": 0,
                         "leaky_relu": 0,
                         "input": 0}
        self.in_node = None
        self.out_node = None

    def set_in_node(self, node):
        self.in_node = node

    def set_out_node(self, node):
        self.out_node = node

    def is_out_node(self, node):
        return self.out_node is node

    def get_name(self, layer_name):
        name = layer_name + "_" + str(self.name_num[layer_name])
        self.name_num[layer_name] += 1
        return name

    def create_conv2d(self, in_node, kernel, strides, padding):
        out_node = Conv2D(self.get_name("conv2d"), in_node, kernel, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_bias_add(self, in_node, biases):
        out_node = BiasAdd(self.get_name("bias_add"), in_node, biases)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_max_pool2d(self, in_node, ksize, strides, padding):
        out_node = MaxPool2D(self.get_name("max_pool2d"), in_node, ksize, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_batch_norm(self, in_node, mean, variance, gamma, epsilon):
        out_node = BatchNorm(self.get_name("batch_norm"), in_node, mean, variance, gamma, epsilon)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_leaky_relu(self, in_node):
        out_node = LeakyReLU(self.get_name("leaky_relu"), in_node)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_input(self, in_shape):
        out_node = Input(self.get_name("input"), in_shape)
        self.G.add_node(out_node)
        self.set_in_node(out_node)  # Assume there's only one input
        return out_node


class DnnNode(object):
    def __init__(self):
        pass

    def run(self):
        self.result = None

    #


# Complete below classes.
#
"""
<Computation>
        
Implement run method for the nodes. Five points per each node implementation.
Take the result from previous node, do proper computation with it class variables, and
save the value at self.result for further computations.

In the run, there must be nested loops which take really long time.
Use multiprocessing library to improve the performance using, for example, batching technique.
"""


class Conv2D(DnnNode):
    def __init__(self, name, in_node, kernel, strides, padding):
        # Given an input tensor of shape [batch, in_height, iout_width, iout_channels]
        # and a filter / kernel tensor of shape [filter_height, filter_width, iout_channels, out_channels]
        self.name = name
        self.in_node = in_node
        self.kernel = kernel
        self.strides = strides
        self.padding = padding

        # check types
        assert isinstance(self.in_node, DnnNode)
        assert (type(self.kernel) is np.memmap) or (type(self.kernel) is np.ndarray)
        # check strides (an int or list of ints with length 1, 2 or 4)
        assert (type(self.strides) is int) or ((type(self.strides) is list) and (len(self.strides) in [1, 2, 4]))
        # check kernel dimension (h, w, iout_channels, out_channels)
        assert (len(self.kernel.shape) == 4)
        # check padding
        assert self.padding in ['SAME', 'VALID']

        # get input shape
        prev_out_shape = self.in_node.in_shape  # (B, H, W, C)
        assert len(prev_out_shape) == 4
        b, h, w, c = prev_out_shape

        # parse ksize
        k_h = kernel.shape[0]
        k_w = kernel.shape[1]
        k_in = kernel.shape[2]
        k_out = kernel.shape[3]
        assert k_in == c
        self.parsed_ksize = [k_h, k_w, k_in, k_out]

        # parse strides
        s_b = s_h = s_w = s_c = 1
        if type(self.strides) is int:
            s_h = s_w = self.strides
        elif len(self.strides) == 2:
            s_h, s_w = self.strides
        elif len(self.strides) == 4:
            s_b, s_h, s_w, s_c = self.strides
        else:
            raise AttributeError
        self.parsed_strides = [s_b, s_h, s_w, s_c]

        # compute padding
        p_h, p_w = 0, 0
        if padding == 'SAME':
            p_h = h * (s_h - 1) + k_h - s_h
            p_w = w * (s_w - 1) + k_w - s_w
        self.parsed_padding = [p_h, p_w]

        # compute output shape
        out_b = (b - 1) // s_b + 1
        out_h = (h + p_h - k_h) // s_h + 1
        out_w = (w + p_w - k_w) // s_w + 1
        out_c = k_out

        if padding == 'SAME': assert out_h == h and out_w == w

        # set output shape
        self.in_shape = [out_b, out_h, out_w, out_c]

        print(self.name)
        print("__init__: input shape " + str(prev_out_shape) + ", output shape" + str(self.in_shape))

    def run(self):
        # strides along each dimension
        s_b, s_h, s_w, s_c = self.strides
        # padding along each dimension
        p_h, p_w = self.parsed_padding
        # caution: tensorflow implementation pads more on rightmost
        # https://intellipaat.com/community/558/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-maxpool-of-tensorflow
        p_h_left = p_h // 2
        p_h_right = (p_h + 1) // 2
        p_w_left = p_w // 2
        p_w_right = (p_w + 1) // 2

        # input dimension
        in_b, in_h, in_w, in_c = self.in_node.in_shape
        # kernel size
        k_h, k_w, k_in, k_out = self.parsed_ksize
        # output dimension
        out_b, out_h, out_w, out_c = self.in_shape

        assert p_h == p_h_left + p_h_right and p_w == p_w_left + p_w_right
        assert k_in == in_c and k_out == out_c

        # zero-pad feature map
        padded_input = np.zeros((in_b, in_h + p_h, in_w + p_w, in_c), dtype=np.float32)
        padded_input[:, p_h_left:(in_h + p_h) - p_h_right, p_w_left:(in_w + p_w) - p_w_right, :] = self.in_node.result

        print("Conv2D: input (B, H, W, C) = (%d, %d, %d, %d)" % (in_b, in_h, in_w, in_c))
        print("Conv2D: kernel (H, W, C_in, C_out) = (%d, %d, %d, %d)" % (k_h, k_w, k_in, k_out))
        print("Conv2D: output (B, H, W, C) = (%d, %d, %d, %d)" % (out_b, out_h, out_w, out_c))
        print("Conv2D: padded input (B, H, W, C) = (%d, %d, %d, %d)" % padded_input.shape)

        # Initialise the array
        self.result = np.zeros((out_b, out_h, out_w, out_c), dtype=np.float32)

        mark = time.time()
        # Loop over output pixels
        for n in range(out_b):
            for m in range(out_c):
                for y in range(out_h):
                    for x in range(out_w):
                        # test for boundary
                        # print("input pool range: w [%d:%d], h [%d:%d]" % (x * s_w, x * s_w + k_w, y * s_h, y * s_h + k_h))
                        assert (y < out_h - 1) or (y == out_h - 1 and y * s_h + k_h == padded_input.shape[1])
                        assert (x < out_w - 1) or (x == out_w - 1 and x * s_w + k_w == padded_input.shape[2])

                        # (n, y, x, m) for batch, row, column and channel of output feature map
                        # convolve with kernel using 1d dot product
                        kernel_1d = self.kernel.flatten()
                        input_1d = padded_input[n*s_b, y*s_h:y * s_h + k_h, x*s_w:x*s_w + k_w, m*s_c].flatten()
                        dot_product = np.dot(kernel_1d, input_1d)

                        # Loop over kernel pixels
                        for c in range(k_in):
                            for i in range(k_h):
                                for j in range(k_w):
                                    # (i, j, c, m) for row, column, in_channel and out_channel of kernel
                                    # corresponding pos on input: (n * s_b, y * s_h, x * s_w, c * s_c) + (0, i, j, 0)
                                    self.result[n, y, x, m] += self.kernel[i, j, c, m] * \
                                                               padded_input[n * s_b, y * s_h + i, x * s_w + j, c * s_c]
                        assert dot_product == self.result[n, y, x, m]
                        raise NotImplementedError
        print("Conv2D: elapsed time %.2fsec" % (time.time() - mark))
        return self.result


class BiasAdd(DnnNode):
    def __init__(self, name, in_node, biases):
        self.name = name
        self.in_node = in_node
        self.biases = biases

        # check types
        assert isinstance(self.in_node, DnnNode)
        assert (type(self.biases) is np.memmap) or (type(self.biases) is np.ndarray)

        # get input shape
        prev_out_shape = self.in_node.in_shape  # (B, ...)

        # check biases shape
        assert list(biases.shape) == list(prev_out_shape[-1:])

        # set output shape
        self.in_shape = prev_out_shape

        print(self.name)
        print("__init__: input shape " + str(prev_out_shape) + ", output shape" + str(self.in_shape))

    def run(self):
        # biases should be broadcasted for b, w and h dimensions
        # e.g. input (1, 416, 416, 256), biases dimension (256,)
        # Initialise the array
        bias = np.zeros(self.in_shape, dtype=np.float32)
        # iterate over the shape of the vector / number of channels
        for i in range(self.in_shape[3]):
            # Fill the entire feature map/channel with the corresponding bias
            bias[:, :, :, i].fill(self.biases[i])
        # Add the bias
        self.result = self.in_node.result + bias
        return self.result


class MaxPool2D(DnnNode):
    def __init__(self, name, in_node, ksize, strides, padding):
        self.name = name
        self.in_node = in_node
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

        # check types
        assert isinstance(self.in_node, DnnNode)
        # check ksize (an int or list of ints with length 1, 2 or 4)
        assert (type(self.ksize) is int) or ((type(self.ksize) is list) and (len(self.ksize) in [1, 2, 4]))
        # check strides (an int or list of ints with length 1, 2 or 4)
        assert (type(self.strides) is int) or ((type(self.strides) is list) and (len(self.strides) in [1, 2, 4]))
        # check padding
        assert self.padding in ['SAME', 'VALID']

        # get input shape
        prev_out_shape = self.in_node.in_shape  # (B, H, W, C)
        assert len(prev_out_shape) == 4
        b, h, w, c = prev_out_shape

        # parse ksize
        k_b = k_h = k_w = k_c = 1
        if type(self.ksize) is int:
            k_h = k_w = self.ksize
        elif len(self.ksize) == 2:
            k_h, k_w = self.ksize
        elif len(self.ksize) == 4:
            k_b, k_h, k_w, k_c = self.ksize
        else:
            raise AttributeError
        self.parsed_ksize = [k_b, k_h, k_w, k_c]

        # parse strides
        s_b = s_h = s_w = s_c = 1
        if type(self.strides) is int:
            s_h = s_w = self.strides
        elif len(self.strides) == 2:
            s_h, s_w = self.strides
        elif len(self.strides) == 4:
            s_b, s_h, s_w, s_c = self.strides
        else:
            raise AttributeError
        self.parsed_strides = [s_b, s_h, s_w, s_c]

        # compute padding
        p_h, p_w = 0, 0
        if padding == 'SAME':
            p_h = h * (s_h - 1) + k_h - s_h
            p_w = w * (s_w - 1) + k_w - s_w
        self.parsed_padding = [p_h, p_w]

        # compute output shape
        out_b = (b - k_b) // s_b + 1
        out_h = (h + p_h - k_h) // s_h + 1
        out_w = (w + p_w - k_w) // s_w + 1
        out_c = (c - k_c) // s_c + 1
        self.in_shape = [out_b, out_h, out_w, out_c]

        print(self.name)
        print("__init__: input shape " + str(prev_out_shape) + ", output shape" + str(self.in_shape))

    def run(self):
        # pooling size along each dimension
        k_b, k_h, k_w, k_c = self.parsed_ksize
        # strides along each dimension
        s_b, s_h, s_w, s_c = self.parsed_strides

        # padding along each dimension
        p_h, p_w = self.parsed_padding
        # caution: tensorflow implementation pads more on rightmost
        p_h_left = p_h // 2
        p_h_right = (p_h + 1) // 2
        p_w_left = p_w // 2
        p_w_right = (p_w + 1) // 2

        # input dimension
        in_b, in_h, in_w, in_c = self.in_node.in_shape
        # output dimension
        out_b, out_h, out_w, out_c = self.in_shape

        assert p_h == p_h_left + p_h_right and p_w == p_w_left + p_w_right

        # zero-pad feature map
        padded_input = np.zeros((in_b, in_h + p_h, in_w + p_w, in_c), dtype=np.float32)
        padded_input[:, p_h_left:in_h + p_h - p_h_right, p_w_left:in_w + p_w - p_w_right, :] = self.in_node.result

        print("MaxPool2D: input (B, H, W, C) = (%d, %d, %d, %d)" % (in_b, in_h, in_w, in_c))
        print("MaxPool2D: output (B, H, W, C) = (%d, %d, %d, %d)" % (out_b, out_h, out_w, out_c))
        print("MaxPool2D: padded input (B, H, W, C) = (%d, %d, %d, %d)" % padded_input.shape)

        # Initialise the array
        self.result = np.zeros((out_b, out_h, out_w, out_c), dtype=np.float32)

        mark = time.time()
        # loop over output pixels
        for n in range(out_b):
            for m in range(out_c):
                for y in range(out_h):
                    for x in range(out_w):
                        # test for boundary
                        # print("input pool range: w [%d:%d], h [%d:%d]" % (x * s_w, x * s_w + k_w, y * s_h, y * s_h + k_h))
                        assert (y < out_h - 1) or (y == out_h - 1 and y * s_h + k_h == padded_input.shape[1])
                        assert (x < out_w - 1) or (x == out_w - 1 and x * s_w + k_w == padded_input.shape[2])

                        self.result[n, y, x, m] = np.amax(padded_input[(n * s_b):(n * s_b + k_b),
                                                          (y * s_h):(y * s_h + k_h),
                                                          (x * s_w):(x * s_w + k_w),
                                                          (m * s_c):(m * s_c + k_c)])

        print("MaxPool2D: elapsed time %.2f" % (time.time() - mark))
        return self.result


class BatchNorm(DnnNode):
    def __init__(self, name, in_node, mean, variance, gamma, epsilon):
        self.name = name
        self.in_node = in_node
        self.mean = mean
        self.variance = variance
        self.gamma = gamma
        self.epsilon = epsilon

        # check types
        assert isinstance(self.in_node, DnnNode)
        assert (type(self.mean) is np.memmap) or (type(self.mean) is np.ndarray)
        assert (type(self.variance) is np.memmap) or (type(self.variance) is np.ndarray)
        assert (type(self.gamma) is np.memmap) or (type(self.gamma) is np.ndarray)
        assert type(self.epsilon) is float

        # get input shape
        prev_out_shape = self.in_node.in_shape  # (B, ...)

        # check mean, variance, gamma shape
        assert list(mean.shape) == list(variance.shape) == list(gamma.shape) == list(prev_out_shape[-1:])

        # set output shape
        self.in_shape = prev_out_shape

        print(self.name)
        print("__init__: input shape " + str(prev_out_shape) + ", output shape" + str(self.in_shape))

    def run(self):
        # mean, variance, and gamma should be broadcasted for b, w and h dimensions
        # e.g. input (1, 416, 416, 256), mean dimension (256,)
        # intitialise the tensor
        self.result = np.zeros(self.in_shape, dtype=np.float32)
        # iterate over the shape of the vector / number of channels
        for i in range(self.in_shape[3]):
            # normalize with mean and variance
            x_hat = (self.in_node.result[:, :, :, i] - self.mean[i]) / math.sqrt((self.variance[i] + self.epsilon))
            # multiply with gamma
            self.result[:, :, :, i] = self.gamma[i] * x_hat
        return self.result


class LeakyReLU(DnnNode):
    def __init__(self, name, in_node):
        self.name = name
        self.in_node = in_node

        # check types
        assert isinstance(self.in_node, DnnNode)

        # get input shape
        prev_out_shape = self.in_node.in_shape  # (B, ...)
        # set output shape
        self.in_shape = prev_out_shape

        print(self.name)
        print("__init__: input shape " + str(prev_out_shape) + ", output shape" + str(self.in_shape))

    def run(self):
        self.result = np.maximum(0.1 * self.in_node.result, self.in_node.result)
        return self.result


# Do not modify below
class Input(DnnNode):
    def __init__(self, name, in_shape):
        self.name = name
        self.in_shape = in_shape
        self.result = np.ndarray(self.in_shape)

    def set_input(self, tensor):
        assert tuple(self.in_shape) == tuple(tensor.shape)
        self.result = tensor

    def run(self):
        pass
