import os
import sys
import math
import networkx as nx
import numpy as np

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
        # Given an input tensor of shape [batch, in_height, in_width, in_channels]
        # and a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
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
        # check kernel dimension (h, w, in_channels, out_channels)
        assert (len(self.kernel.shape) == 4)
        # check padding
        assert self.padding in ['SAME', 'VALID']

        # get input shape
        prev_out_shape = self.in_node.in_shape  # (B, H, W, C)
        assert len(prev_out_shape) == 4
        b, h, w, c = prev_out_shape

        # parse ksize
        print(kernel.shape)
        k_h = kernel.shape[0]
        k_w = kernel.shape[1]
        k_in = kernel.shape[2]
        k_out = kernel.shape[3]
        assert k_in == c

        # parse strides
        s_b = s_h = s_w = s_c = 1
        if type(self.strides) is int:
            s_h = s_w = self.strides
        elif len(self.strides) == 2:
            s_h = self.strides[0]
            s_w = self.strides[1]
        elif len(self.strides) == 4:
            s_b = self.strides[0]
            s_h = self.strides[1]
            s_w = self.strides[2]
            s_c = self.strides[3]
        else: raise AttributeError

        # compute padding
        p_h, p_w = 0, 0
        if padding == 'SAME':
            p_h = h * (s_h - 1) + k_h - s_h
            p_w = w * (s_w - 1) + k_w - s_w

        # compute output shape
        n_b = (b - 1) // s_b + 1
        n_h = (h + p_h - k_h) // s_h + 1
        n_w = (w + p_w - k_w) // s_w + 1
        n_c = k_out

        if padding == 'SAME': assert n_h == h and n_w == w

        # set output shape
        self.in_shape = [n_b, n_h, n_w, n_c]

        print(self.name, self.in_shape)

    def run(self):
        pass

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
        print(list(biases.shape))
        print(list(prev_out_shape[1:]))
        assert list(biases.shape) == list(prev_out_shape[1:])

        # set output shape
        self.in_shape = prev_out_shape

        print(self.name, self.in_shape)

    def run(self):
        pass

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
            k_h = self.ksize[0]
            k_w = self.ksize[1]
        elif len(self.ksize) == 4:
            k_b = self.ksize[0]
            k_h = self.ksize[1]
            k_w = self.ksize[2]
            k_c = self.ksize[3]
        else: raise AttributeError

        # parse strides
        s_b = s_h = s_w = s_c = 1
        if type(self.strides) is int:
            s_h = s_w = self.strides
        elif len(self.strides) == 2:
            s_h = self.strides[0]
            s_w = self.strides[1]
        elif len(self.strides) == 4:
            s_b = self.strides[0]
            s_h = self.strides[1]
            s_w = self.strides[2]
            s_c = self.strides[3]
        else: raise AttributeError

        # compute padding
        p_h, p_w = 0, 0
        if padding == 'SAME':
            p_h = h * (s_h - 1) + k_h - s_h
            p_w = w * (s_w - 1) + k_w - s_w

        # compute output shape
        n_b = (b - k_b) // s_b + 1
        n_h = (h + p_h - k_h) // s_h + 1
        n_w = (w + p_w - k_w) // s_w + 1
        n_c = (c - k_c) // s_c + 1
        self.in_shape = [n_b, n_h, n_w, n_c]

        print(self.name, self.in_shape)
        
    def run(self):
        pass

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
        print(mean.shape)
        print(variance.shape)
        print(gamma.shape)
        print(np.array(prev_out_shape[1:]))
        assert 0 not in (mean.shape == variance.shape == gamma.shape == np.array(prev_out_shape[1:]))

        # set output shape
        self.in_shape = prev_out_shape

        print(self.name, self.in_shape)

    def run(self):
        pass

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

        print(self.name, self.in_shape)

    def run(self):
        pass


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

