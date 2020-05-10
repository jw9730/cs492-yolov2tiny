import os
import sys
import math
import networkx as nx
import numpy as np
import time
import multiprocessing as mp

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
            p_h = int(h * (s_h - 1) + k_h - s_h)
            p_w = int(w * (s_w - 1) + k_w - s_w)
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
        # print("__init__: input shape " + str(prev_out_shape) + ", output shape" + str(self.in_shape))

    def run(self):
        assert tuple(self.in_node.in_shape) == tuple(self.in_node.result.shape)

        # padding along each dimension
        p_h, p_w = self.parsed_padding
        # caution: tensorflow implementation pads more on rightmost
        p_h_left = p_h // 2
        p_h_right = (p_h + 1) // 2
        p_w_left = p_w // 2
        p_w_right = (p_w + 1) // 2
        assert p_h == p_h_left + p_h_right and p_w == p_w_left + p_w_right

        # input dimension
        in_b, in_h, in_w, in_c = self.in_node.in_shape
        # kernel size
        k_h, k_w, k_in, k_out = self.parsed_ksize
        # strides along each dimension
        s_b, s_h, s_w, s_c = self.strides
        # output dimension
        out_b, out_h, out_w, out_c = self.in_shape
        assert k_in == in_c and k_out == out_c
        assert in_b == out_b

        # zero-pad feature map
        padded_input = np.zeros((in_b, in_h + p_h, in_w + p_w, in_c), dtype=np.float32)
        padded_input[:, p_h_left:(in_h + p_h) - p_h_right, p_w_left:(in_w + p_w) - p_w_right, :] = self.in_node.result

        # print("Conv2D: input (B, H, W, C) = (%d, %d, %d, %d)" % (in_b, in_h, in_w, in_c))
        # print("Conv2D: kernel (H, W, C_in, C_out) = (%d, %d, %d, %d)" % (k_h, k_w, k_in, k_out))
        # print("Conv2D: output (B, H, W, C) = (%d, %d, %d, %d)" % (out_b, out_h, out_w, out_c))
        # print("Conv2D: padded input (B, H, W, C) = (%d, %d, %d, %d)" % padded_input.shape)

        ################################################################################################################
        def run_split(queue, padded_input, kernel, start, end, idx):
            mark = time.time()
            result = np.zeros((out_b, out_h, out_w, out_c), dtype=np.float32)
            # loop over output pixels
            for n in range(out_b):
                for m in range(start, end):
                    for y in range(out_h):
                        for x in range(out_w):
                            for c in range(k_in):
                                for i in range(k_h):
                                    for j in range(k_w):
                                        result[n, y, x, m] += kernel[i, j, c, m] * \
                                                              padded_input[n * s_b, y * s_h + i, x * s_w + j, c * s_c]
            queue.put(result)
            print("Conv2D mp: [%d] elapsed time %.2fsec" % (idx, time.time() - mark))

        # parallelization should be done across batches, output pixels and channels
        q = mp.Queue()
        p_list = list()

        c_per_split = 3
        num_splits = math.ceil(out_c / c_per_split)
        for split_idx in range(num_splits):
            start = split_idx * c_per_split
            end = min(start + c_per_split, out_c)
            print("[%d] %d:%d" % (split_idx, start, end))

            p = mp.Process(target=run_split, args=(q, padded_input, self.kernel, start, end, split_idx))
            p_list.append(p)
            p.start()

        cnt = 0
        mp_result = np.zeros((out_b, out_h, out_w, out_c), dtype=np.float32)
        while cnt < num_splits:
            mp_result += q.get()
            cnt += 1
        for p in p_list:
            p.join()
        ################################################################################################################

        kernel_2d = self.kernel.reshape((-1, out_c))  # (h * w * in_c, out_c)
        vectorized_result = np.zeros((out_b, out_h, out_w, out_c), dtype=np.float32)
        mark = time.time()
        for y in range(out_h):
            for x in range(out_w):
                # vectorized convolution
                input_rf = padded_input[0::s_b, (y * s_h):(y * s_h + k_h), (x * s_w):(x * s_w + k_w), 0::s_c]
                vectorized_result[:, y, x, :] = np.matmul(input_rf.reshape((out_b, -1)), kernel_2d)
        assert (mp_result-vectorized_result).mean() < 1e-5
        print("Conv2D: elapsed time %.2fsec" % (time.time() - mark))

        raise NotImplementedError

        # initialization
        self.result = np.zeros((out_b, out_h, out_w, out_c), dtype=np.float32)
        mark = time.time()
        # loop over output pixels
        for n in range(out_b):
            for m in range(out_c):
                for y in range(out_h):
                    for x in range(out_w):
                        for c in range(k_in):
                            for i in range(k_h):
                                for j in range(k_w):
                                    self.result[n, y, x, m] += self.kernel[i, j, c, m] * \
                                                                padded_input[n * s_b, y * s_h + i, x * s_w + j, c * s_c]
        print("Conv2D long: elapsed time %.2fsec" % (time.time() - mark))
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
        # print("__init__: input shape " + str(prev_out_shape) + ", output shape" + str(self.in_shape))

    def run(self):
        assert tuple(self.in_node.in_shape) == tuple(self.in_node.result.shape)

        # biases should be broadcasted for b, w and h dimensions
        # e.g. input (1, 416, 416, 256), biases dimension (256,)
        self.result = self.in_node.result + self.biases.reshape((1, 1, 1, -1))

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
        target_out_h = target_out_w = 0
        if padding == 'SAME':
            target_out_h = np.ceil(float(h) / float(s_h))
            target_out_w = np.ceil(float(w) / float(s_w))
        elif padding == 'VALID':
            target_out_h = np.ceil(float(h - k_h + 1) / float(s_h))
            target_out_w = np.ceil(float(w - k_w + 1) / float(s_w))
        p_h = int(np.ceil((target_out_h - 1) * s_h + k_h - h))
        p_w = int(np.ceil((target_out_w - 1) * s_w + k_w - w))
        self.parsed_padding = [p_h, p_w]

        # compute output shape
        out_b = (b - k_b) // s_b + 1
        out_h = (h + p_h - k_h) // s_h + 1
        out_w = (w + p_w - k_w) // s_w + 1
        out_c = (c - k_c) // s_c + 1
        self.in_shape = [out_b, out_h, out_w, out_c]

        print(self.name)
        # print("__init__: input shape " + str(prev_out_shape) + ", output shape" + str(self.in_shape))

    def run(self):
        assert tuple(self.in_node.in_shape) == tuple(self.in_node.result.shape)

        # padding along each dimension
        p_h, p_w = self.parsed_padding
        # caution: tensorflow implementation pads more on rightmost
        p_h_left = p_h // 2
        p_h_right = (p_h + 1) // 2
        p_w_left = p_w // 2
        p_w_right = (p_w + 1) // 2
        assert p_h == p_h_left + p_h_right and p_w == p_w_left + p_w_right

        # input dimension
        in_b, in_h, in_w, in_c = self.in_node.in_shape
        # pooling size along each dimension
        k_b, k_h, k_w, k_c = self.parsed_ksize
        # strides along each dimension
        s_b, s_h, s_w, s_c = self.parsed_strides
        # output dimension
        out_b, out_h, out_w, out_c = self.in_shape

        # zero-pad feature map
        padded_input = - (np.ones((in_b, in_h + p_h, in_w + p_w, in_c), dtype=np.float32) * np.inf)
        padded_input[:, p_h_left:in_h + p_h - p_h_right, p_w_left:in_w + p_w - p_w_right, :] = self.in_node.result

        # print("MaxPool2D: input (B, H, W, C) = (%d, %d, %d, %d)" % (in_b, in_h, in_w, in_c))
        # print("MaxPool2D: output (B, H, W, C) = (%d, %d, %d, %d)" % (out_b, out_h, out_w, out_c))
        # print("MaxPool2D: padded input (B, H, W, C) = (%d, %d, %d, %d)" % padded_input.shape)

        # initialise
        self.result = np.zeros((out_b, out_h, out_w, out_c), dtype=np.float32)

        # mark = time.time()
        # loop over output pixels
        for y in range(out_h):
            for x in range(out_w):
                # vectorized max
                # problem: this implementation assumes k_b == 1 and k_c == 1
                # todo: allow pooling across batches or channels, that is, k_b != 1 or k_c != 1 conditions
                # for that, you can add additional loop over batches and channels and
                # extend input_rf over batch and channel dimension
                input_rf = padded_input[0::s_b, (y * s_h):(y * s_h + k_h), (x * s_w):(x * s_w + k_w), 0::s_c]
                self.result[:, y, x, :] = np.amax(input_rf.reshape((out_b, k_h * k_w, out_c)), axis=1)

        # print("MaxPool2D: elapsed time %.2fsec" % (time.time() - mark))
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
        # print("__init__: input shape " + str(prev_out_shape) + ", output shape" + str(self.in_shape))

    def run(self):
        assert tuple(self.in_node.in_shape) == tuple(self.in_node.result.shape)

        # mean, variance, and gamma should be broadcasted for b, w and h dimensions
        # e.g. input (1, 416, 416, 256), mean dimension (256,)
        self.result = self.gamma.reshape((1, 1, 1, -1)) * \
                      (self.in_node.result - self.mean.reshape((1, 1, 1, -1))) / \
                      (np.sqrt(self.variance).reshape((1, 1, 1, -1)) + self.epsilon)

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
        # print("__init__: input shape " + str(prev_out_shape) + ", output shape" + str(self.in_shape))

    def run(self):
        assert tuple(self.in_node.in_shape) == tuple(self.in_node.result.shape)

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
