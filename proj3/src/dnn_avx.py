import os
import sys
import math
import networkx as nx
import numpy as np
from itertools import product
from multiprocessing import Process, sharedctypes

import time
from ctypes import *
mylib = cdll.LoadLibrary('./avx.so')

parallelism = 8

class DnnInferenceEngine(object):
    def __init__(self, graph, debug):
        self.g = graph
        self.debug = debug

    def run(self, tin):
        self.g.in_node.set_input(tin)
        out = {}
        currents = [self.g.in_node]
        done = set()
        counter = 0
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
                current.run(counter)

                # debug mode
                if self.debug:
                    # using a counter is heuristic! once a layer gets input from >1 predecessors, race condition happens
                    # still, no such case in yolov2tiny
                    np.save(file='intermediate/layer_{}'.format(counter), arr=current.result)
                    print("run: execution result of {} saved at ./intermediate/layer_{}.npy".format(current.name, counter))

                if not isinstance(current, Input):
                    counter += 1
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
        if self.out_node is node:
            return True
        else:
            return False

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

    def run(self, counter):
        self.result = None 

class Conv2D(DnnNode):
    def __init__(self, name, in_node, kernel, strides, padding):
        self.name = name
        
        # input node
        self.in_node = in_node

        # weights
        self.weights = kernel
        assert len(self.weights.shape) == 4
        if len(self.in_node.result.shape) < 3:
            input_channels = 1
        else:
            input_channels = self.in_node.result.shape[-1]

        # strides
        if strides is None:
            strides = (1,1,1,1)
        assert len(strides) == len(self.in_node.result.shape)
        self.strides = strides

        # padding
        if padding == 'SAME':
            self.pad = (
                    (0,0),
                    (self.weights.shape[0]//2, self.weights.shape[0]//2),
                    (self.weights.shape[1]//2, self.weights.shape[1]//2),
                    (0,0)
                    )
        elif padding == 'VALID':
            self.pad = ((0,0), (0,0), (0,0), (0,0))
        else:
            assert len(padding) == 2
            self.pad = padding
            
        ptin = np.pad(self.in_node.result, self.pad, mode='constant')
        self.PW = ptin.shape[1]
        self.PH = ptin.shape[2]
        self.KW = self.weights.shape[0]
        self.KH = self.weights.shape[1]
        self.IC = self.weights.shape[2]
        self.OC = self.weights.shape[3]
        self.SW = self.strides[1]
        self.SH = self.strides[2]
        self.OW = int((self.PW - self.KW) / self.SW + 1)
        self.OH = int((self.PH - self.KH) / self.SH + 1)

        self.result = np.zeros((1, self.OW, self.OH, self.OC))
        tmp_result = np.ctypeslib.as_ctypes(self.result)
        self.shm_result = sharedctypes.RawArray(tmp_result._type_, tmp_result)

    def run(self, counter):
        # 1. Toeplitz matrix multiplication
        kernel = self.weights.reshape((self.KW * self.KH * self.IC, self.OC)).astype(np.float32)
        pin = np.pad(self.in_node.result, self.pad, mode='constant')
        toeplitz_in = np.zeros((self.OW * self.OH, self.KW * self.KH * self.IC), dtype=np.float32)
        for ow in range(0, self.OW):
            for oh in range(0, self.OH):
                w0 = self.SW * ow
                h0 = self.SH * oh
                toeplitz_in[ow * self.OH + oh, :] = pin[0, w0:w0+self.KW, h0:h0+self.KH, :].flatten()

        """
        tic = time.time()
        ref_result = np.matmul(toeplitz_in, kernel).reshape((1, self.OW, self.OH, self.OC))# correctness check
        toc = time.time()
        print("Conv2D: TOEPLITZ-NUMPY elapsed time {:1.5f}s".format(toc - tic))
        """

        tic = time.time()
        c_float_p = POINTER(c_float)
        in_p = np.ascontiguousarray(toeplitz_in).ctypes.data_as(c_float_p)
        k_p = np.asfortranarray(kernel).ctypes.data_as(c_float_p)
        out_p = np.zeros((1, self.OW, self.OH, self.OC), dtype=np.float32, order='c').ctypes.data_as(c_float_p)
        mylib.matmul.argtypes = c_float_p, c_float_p, c_float_p, c_int, c_int, c_int
        mylib.matmul(in_p, k_p, out_p, c_int(self.OW * self.OH), c_int(self.KW * self.KH * self.IC), c_int(self.OC))
        avx_result = np.ctypeslib.as_array(out_p, (1, self.OW, self.OH, self.OC))
        toc = time.time()
        print("Conv2D: TOEPLITZ-OFFLOAD elapsed time {:1.5f}s".format(toc - tic))

        self.result = avx_result
        #assert abs(avx_result - ref_result).mean() < 1e-5, "Conv2D: correctness check failed with mean err {}".format((avx_result - ref_result).mean())


class BiasAdd(DnnNode):
    def __init__(self, name, in_node, biases):
        self.name = name

        self.in_node = in_node 

        tin = self.in_node.result
        self.OW = tin.shape[1]
        self.OH = tin.shape[2]
        self.OC = tin.shape[3]

        self.biases = biases 
        assert self.biases.shape[-1] == self.OC 

        self.result = self.in_node.result 

    def run(self, counter):
        """
        tic = time.time()
        ref_result = (self.in_node.result + self.biases.reshape((1, 1, 1, -1))).astype(np.float32)
        toc = time.time()
        print("BiasAdd: NUMPY elapsed time {:1.5f}s".format(toc - tic))
        """

        tic = time.time()
        c_float_p = POINTER(c_float)
        in_p = np.ascontiguousarray(self.in_node.result.reshape((-1, self.OC)).transpose()).astype(np.float32).ctypes.data_as(c_float_p)
        b_p = np.ascontiguousarray(self.biases).astype(np.float32).ctypes.data_as(c_float_p)
        out_p = np.zeros((self.OC, self.OW * self.OH), dtype=np.float32, order='c').ctypes.data_as(c_float_p)
        mylib.bias_add.argtypes = c_float_p, c_float_p, c_float_p, c_int, c_int
        mylib.bias_add(in_p, b_p, out_p, c_int(self.OW * self.OH), c_int(self.OC))
        avx_result = np.ctypeslib.as_array(out_p, (self.OC, self.OW * self.OH)).transpose().reshape((1, self.OW, self.OH, self.OC))
        toc = time.time()
        print("BiasAdd: OFFLOAD elapsed time {:1.5f}s".format(toc - tic))

        self.result = avx_result
        #assert abs(avx_result - ref_result).mean() < 1e-5, "BiasAdd: correctness check failed with mean err {}".format(abs(avx_result - ref_result).mean())
        #assert np.count_nonzero(np.isnan(self.result)) == 0, "{} nans found in output".format(np.count_nonzero(np.isnan(self.result)))

class MaxPool2D(DnnNode):
    def __init__(self, name, in_node, ksize, strides, padding):
        self.name = name

        # input node 
        self.in_node = in_node

        tin = self.in_node.result
        IW = tin.shape[1]
        IH = tin.shape[2]
        self.OC = tin.shape[3]

        # pooling kernel size
        assert len(ksize) == len(self.in_node.result.shape)
        self.ksize = ksize

        # stride
        self.stride = strides

        if padding == 'VALID':
            self.pad = (
                    (0,0),
                    (0,0),
                    (0,0),
                    (0,0))
        elif padding == 'SAME':
            w = self.in_node.result.shape[1]
            h = self.in_node.result.shape[2]

            out_w = math.ceil(float(w) / float(self.stride[1]))
            out_h = math.ceil(float(h) / float(self.stride[2]))
            pad_along_w = max(int((w - self.ksize[1]) / self.stride[1]) + 1 - w, 0)
            pad_along_h = max(int((h - self.ksize[2]) / self.stride[2]) + 1 - h, 0)
            pad_left = pad_along_w // 2
            pad_right = pad_along_w - pad_left
            pad_top = pad_along_h // 2
            pad_bottom = pad_along_h - pad_top
            self.pad = (
                    (0,0),
                    (pad_left,pad_right),
                    (pad_top,pad_bottom),
                    (0,0))
        else:
            raise Exception("Unexpected padding mode: {}".format(padding))	

        ptin= np.pad(self.in_node.result, self.pad, mode='constant')
        self.PW = ptin.shape[1]
        self.PH = ptin.shape[2]
        self.result = np.zeros((1, int(self.PW / self.stride[1]), int(self.PH / self.stride[2]), self.OC))

    def run(self, counter):
        # Toeplitz matrix + max filter
        _, OW, OH, OC = self.result.shape
        pin = np.pad(self.in_node.result, self.pad, mode='constant')
        rpin = np.zeros((OW * OH, self.ksize[1], self.ksize[2], self.OC), dtype=np.float32)
        for ow in range(0, OW):
            for oh in range(0, OH):
                w0 = self.stride[1] * ow
                h0 = self.stride[2] * oh
                rpin[ow * OH + oh, :, :, :] = pin[0, w0:w0+self.ksize[1], h0:h0+self.ksize[2], :]
        toeplitz_in = rpin.transpose((0, 3, 1, 2)).reshape((OW * OH * self.OC, self.ksize[1] * self.ksize[2]))

        """
        tic = time.time()
        ref_result = np.max(toeplitz_in, axis=1).reshape((1, OW, OH, self.OC))  # correctness check
        toc = time.time()
        print("MaxPool2D: TOEPLITZ-NUMPY elapsed time {:1.5f}s".format(toc - tic))
        """

        tic = time.time()
        c_float_p = POINTER(c_float)
        in_p = np.ascontiguousarray(toeplitz_in).ctypes.data_as(c_float_p)
        out_p = np.zeros((OW * OH * self.OC,), dtype=np.float32, order='c').ctypes.data_as(c_float_p)
        mylib.max_pool.argtypes = c_float_p, c_float_p, c_int, c_int
        mylib.max_pool(in_p, out_p, c_int(OW * OH * self.OC), c_int(self.ksize[1] * self.ksize[2]))
        avx_result = np.ctypeslib.as_array(out_p, (1, OW, OH, self.OC))
        toc = time.time()
        print("MaxPool2D: TOEPLITZ-OFFLOAD elapsed time {:1.5f}s".format(toc - tic))

        self.result = avx_result
        #assert abs(avx_result - ref_result).mean() < 1e-5, "MaxPool2D: correctness check failed with mean err {}".format((avx_result - ref_result).mean())


class BatchNorm(DnnNode):
    def __init__(self, name, in_node, mean, variance, gamma, epsilon):
        self.name = name

        self.in_node = in_node

        tin = self.in_node.result
        self.OW = tin.shape[1]
        self.OH = tin.shape[2]
        self.OC = tin.shape[3]

        self.mean = mean 
        assert self.mean.shape[0] == self.OC
        self.variance = variance
        assert self.variance.shape[0] == self.OC 
        self.gamma = gamma
        assert self.gamma.shape[0] == self.OC
        self.epsilon = epsilon
        
        self.result = self.in_node.result

    def run(self, counter):
        """
        tic = time.time()
        ref_result = self.gamma.reshape((1, 1, 1, -1)) * \
                    (self.in_node.result - self.mean.reshape((1, 1, 1, -1))) / \
                    (np.sqrt(self.variance).reshape((1, 1, 1, -1)) + self.epsilon).astype(np.float32)
        toc = time.time()
        print("BatchNorm: NUMPY elapsed time {:1.5f}s".format(toc - tic))
        """

        tic = time.time()
        c_float_p = POINTER(c_float)
        in_p = np.ascontiguousarray(self.in_node.result.reshape((-1, self.OC)).transpose()).astype(np.float32).ctypes.data_as(c_float_p)
        mu_p = np.ascontiguousarray(self.mean.astype(np.float32)).ctypes.data_as(c_float_p)
        gamma_p = np.ascontiguousarray(self.gamma.astype(np.float32)).ctypes.data_as(c_float_p)
        var_p = np.ascontiguousarray(self.variance.astype(np.float32)).ctypes.data_as(c_float_p)
        out_p = np.zeros((self.OC, self.OW * self.OH), dtype=np.float32, order='c').ctypes.data_as(c_float_p)
        mylib.batch_norm.argtypes = c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_float, c_int, c_int
        mylib.batch_norm(in_p, mu_p, gamma_p, var_p, out_p, c_float(self.epsilon), c_int(self.OW * self.OH), c_int(self.OC))
        avx_result = np.ctypeslib.as_array(out_p, (self.OC, self.OW * self.OH)).transpose().reshape((1, self.OW, self.OH, self.OC))
        toc = time.time()
        print("BatchNorm: OFFLOAD elapsed time {:1.5f}s".format(toc - tic))

        self.result = avx_result
        # correctness check
        #assert abs(avx_result - ref_result).mean() < 1e-5, "BatchNorm: correctness check failed with mean err {}".format(abs(avx_result - ref_result).mean())
        #assert np.count_nonzero(np.isnan(self.result)) == 0, "{} nans found in output".format(np.count_nonzero(np.isnan(self.result)))

class LeakyReLU(DnnNode):
    def __init__(self, name, in_node):
        self.name = name

        self.in_node = in_node

        tin = self.in_node.result
        self.OW = tin.shape[1]
        self.OH = tin.shape[2]
        self.OC = tin.shape[3]

        self.result = self.in_node.result

    def run(self, counter):
        """
        tic = time.time()
        ref_result = np.maximum(0.1 * self.in_node.result, self.in_node.result)
        toc = time.time()
        print("LeakyReLU: NUMPY elapsed time {:1.5f}s".format(toc - tic))
        """

        tic = time.time()
        c_float_p = POINTER(c_float)
        in_p = np.ascontiguousarray(self.in_node.result).astype(np.float32).ctypes.data_as(c_float_p)
        out_p = np.zeros((self.OW * self.OH * self.OC), dtype=np.float32, order='c').ctypes.data_as(c_float_p)
        mylib.leaky_relu.argtypes = c_float_p, c_float_p, c_int
        mylib.leaky_relu(in_p, out_p, c_int(self.OW * self.OH * self.OC))
        avx_result = np.ctypeslib.as_array(out_p, (1, self.OW, self.OH, self.OC))
        toc = time.time()
        print("LeakyReLU: OFFLOAD elapsed time {:1.5f}s".format(toc - tic))

        self.result = avx_result
        #assert abs(avx_result - ref_result).mean() < 1e-5, "LeakyReLU: correctness check failed with mean err {}".format(abs(avx_result - ref_result).mean())
        #assert np.count_nonzero(np.isnan(self.result)) == 0, "{} nans found in output".format(np.count_nonzero(np.isnan(self.result)))

class Input(DnnNode):
   def __init__(self, name, in_shape):
       self.name = name
       self.in_shape = in_shape
       self.result = np.ndarray(self.in_shape)

   def set_input(self, tensor):
       assert tuple(self.in_shape) == tuple(tensor.shape)
       self.result = tensor

   def run(self, counter):
       pass


