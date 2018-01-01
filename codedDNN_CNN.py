import numpy as np
import scipy.sparse
import random
import abc
import time
from datetime import datetime
from scipy import signal
from numpy.polynomial import polynomial
import sys
from mpi4py import MPI

import mnistLoad
import cifar10Load

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MNIST_PATH = './datasets/mnist/'
OUTPUT_PATH = './output/'

np.random.seed(seed=2017 + rank + 30)

class Graph(object):
    def __init__(self, p_num):
        self._p_num = p_num
        self._p_count = 0
        self._node_list = []
        self._node_count = 0
        self._current_node = None
        self._current_node_idx = None
        self._start_time = None
        self._store_init_time = None
    
    def addNode(self, previous, next, N, pre_N, config):
        self._node_count += 1
        type = config.get("type")
        if type == "Sigmoid":
            Sigmoid_layer = "Sigmoid"
            if rank == 0:
                Sigmoid_layer = Sigmoid()
            self._node_list.append(Sigmoid_layer)
        elif type == "Relu":
            Relu_layer = "Relu"
            if rank == 0:
                Relu_layer = Relu()
            self._node_list.append(Relu_layer)
        elif type == "Bias":
            alpha = config.get("alpha")
            Bias_layer = "Bias"
            if rank == 0:
                Bias_layer = Bias(alpha, [], previous, next, N, pre_N)
            self._node_list.append(Bias_layer)
        elif type == "ReshapeTensor":
            ReshapeTensor_layer = "ReshapeTensor"
            if rank == 0:
                ReshapeTensor_layer = ReshapeTensor(N, pre_N)
            self._node_list.append(ReshapeTensor_layer)
        elif type == "DecentralMdsFc":
            Pr = config.get("Pr")
            qr = config.get("qr")
            Pc = config.get("Pc")
            qc = config.get("qc")
            P = config.get("P")
            alpha = config.get("alpha")
            beta1 = config.get("beta1")
            beta2 = config.get("beta2")
            M = config.get("M")
            error_rate = config.get("error_rate")
            mini_batch = config.get("mini_batch")
            p_used_num = P
            processors_id = range(p_used_num)
            node_group = comm.Get_group().Incl(processors_id)
            node_comm = comm.Create(node_group)
            DecentralMdsFc_layer = "DecentralMdsFc"
            if node_comm != MPI.COMM_NULL:
                DecentralMdsFc_layer = DecentralMdsFc(Pr, qr, Pc, qc, P, alpha, beta1, beta2, M, error_rate, node_comm, processors_id, previous, next, N, pre_N)
            self._node_list.append(DecentralMdsFc_layer)
            return DecentralMdsFc_layer
        elif type == "DecentralReplicaFc":
            Pr = config.get("Pr")
            Pc = config.get("Pc")
            P = config.get("P")
            alpha = config.get("alpha")
            beta1 = config.get("beta1")
            beta2 = config.get("beta2")
            M = config.get("M")
            error_rate = config.get("error_rate")
            mini_batch = config.get("mini_batch")
            p_used_num = P
            processors_id = range(p_used_num)
            node_group = comm.Get_group().Incl(processors_id)
            node_comm = comm.Create(node_group)
            DecentralReplicaFc_layer = "DecentralReplicaFc"
            if node_comm != MPI.COMM_NULL:
                DecentralReplicaFc_layer = DecentralReplicaFc(Pr, Pc, P, alpha, beta1, beta2, M, error_rate, node_comm, processors_id, previous, next, N, pre_N)
            self._node_list.append(DecentralReplicaFc_layer)
            return DecentralReplicaFc_layer
        elif type == "DecentralUncodedFc":
            Pr = config.get("Pr")
            Pc = config.get("Pc")
            P = config.get("P")
            alpha = config.get("alpha")
            beta1 = config.get("beta1")
            beta2 = config.get("beta2")
            M = config.get("M")
            error_rate = config.get("error_rate")
            mini_batch = config.get("mini_batch")
            p_used_num = P
            processors_id = range(p_used_num)
            node_group = comm.Get_group().Incl(processors_id)
            node_comm = comm.Create(node_group)
            DecentralUncodedFc_layer = "DecentralUncodedFc"
            if node_comm != MPI.COMM_NULL:
                DecentralUncodedFc_layer = DecentralUncodedFc(Pr, Pc, P, alpha, beta1, beta2, M, error_rate, node_comm, processors_id, previous, next, N, pre_N)
            self._node_list.append(DecentralUncodedFc_layer)
            return DecentralUncodedFc_layer
        elif type == "DecentralRelu":
            P = config.get("P")
            Relu_layer = "DecentralRelu"
            if rank < P:
                Relu_layer = Relu()
            self._node_list.append(Relu_layer)
        elif type == "DecentralBias":
            P = config.get("P")
            alpha = config.get("alpha")
            beta1 = config.get("beta1")
            beta2 = config.get("beta2")
            mini_batch = config.get("mini_batch")
            Bias_layer = "DecentralBias"
            if rank < P:
                Bias_layer = Bias(alpha, beta1, beta2, [], previous, next, N, pre_N)
            self._node_list.append(Bias_layer)
        else:
            print("Error: Invalid Node type.")
            return None
    
    def init(self):
        if rank == 0:
            print("Initialize nodes...")
        self._store_init_time = None
        self._start_time = datetime.now()
        for i, node in enumerate(self._node_list):
            if node is not None and type(node) is not str:
                node.init(i)
        #synchronize all processors after initialization
        if rank == 0:
            for i in range(size-1):
                msg = comm.recv(source=i+1)
            print("Initialize finish.")
        else:
            comm.send('y', dest=0)
    
    def saveW(self):
        for i, node in enumerate(self._node_list):
            if node is not None and type(node) is not str:
                node.saveW(rank, i)

    def loadW(self):
        for i, node in enumerate(self._node_list):
            if node is not None and type(node) is not str:
                node.loadW(rank, i)
    
    def fitSeq(self, X, Y, data_size, iter=100, checkpoint_iter=100, replication=False, init_check=True):
        if rank == 0:
            print("fitting...")
            loss_list = [None] * (iter * data_size)
            time_list = [None] * (iter * data_size)
        else:
            loss_list = None
            time_list = None
        check_point_t = [-1, -1] #[0]: last t. [1]: last j.
        fail = False
        roll_back_count = 0
        t = 0
        j = 0
        while t < iter:
            j = 0
            while j < data_size:
                if rank == 0:
                    time = datetime.now() - self._start_time
                    if self._store_init_time is not None:
                        time -= self._store_init_time
                    print(time.total_seconds())
                    time_list[t * data_size + j] = time
                generate_error = True
                if t * data_size + j <= checkpoint_iter and replication:
                    generate_error = False
                if fail:
                    #roll back
                    roll_back_count += 1
                    fail = False
                    self.loadW()
                    t = check_point_t[0]
                    j = check_point_t[1]
                    if rank == 0:
                        print("roll back")
                if (t * data_size + j)%checkpoint_iter == 0 and (t != check_point_t[0] or j != check_point_t[1]):
                    #do a dot product with (1,1,...) vector to make sure no errors before check-pointing
                    if init_check or t != 0 or j != 0:
                        layer_num = len(self._node_list)
                        x = None
                        y = None
                        if X is not None:
                            x = np.ones(X[j].shape)
                        for i, node in enumerate(self._node_list):
                            if type(node) is DecentralMdsFc or type(node) is DecentralReplicaFc:
                                y, fail = node.forward(x, t * data_size * layer_num + j * layer_num + i + 2 * iter * data_size * layer_num, False, False)
                            if y is not None:
                                x = np.ones(y.shape)
                            if fail:
                                break
                        if fail:
                            continue
                        dy = x
                        dx = None
                        for i, node in enumerate(self._node_list[::-1]):
                            if type(node) is DecentralMdsFc:
                                dx, fail = node.backprop(dy, t * data_size * layer_num + j * layer_num + i + 3 * iter * data_size * layer_num, False, False)
                            if dx is not None:
                                dy = np.ones(dx.shape)
                            if fail:
                                break
                        if fail:
                            continue
                    #check-pointing
                    self.saveW()
                    check_point_t[0] = t
                    check_point_t[1] = j
                
                #forward of one data instance
                x = None
                if X is not None:
                    x = X[j]
                y = None
                layer_num = len(self._node_list)
                for i, node in enumerate(self._node_list):
                    if node is not None and type(node) is not str:
                        if type(node) is DecentralMdsFc or type(node) is DecentralReplicaFc or type(node) is DecentralUncodedFc:
                            y, fail = node.forward(x, t * data_size * layer_num + j * layer_num + i, generate_error, False)
                        else:
                            y = node.forward(x, t * data_size * layer_num + j * layer_num + i)
                        x = y
                    if fail:
                        break
                if fail:
                    continue
                
                #calculate gradient of the last layer
                dy = None
                if y is not None:
                    #regression loss
                    """
                    dy = -2 * (y - Y[j])
                    if (t * data_size + j) % 500 == 0:
                        print(np.sum((y - Y[j]) * (y - Y[j])))
                    #print(dy)
                    """
                    #softmax loss
                    y -= np.amax(y)
                    dy = np.empty(10, dtype='d')
                    for k in range(10):
                        dy[k] = -np.exp(y[k]) / np.sum(np.exp(y))
                        if k == Y[j]:
                            dy[k] += 1
                    if (t * data_size + j) % 1 == 0 and rank == 0:
                        loss = -y[Y[j]] + np.log(np.sum(np.exp(y)))
                        print("epoch " + str(t) + " iter " + str(j) + " roll_back_count " + str(roll_back_count) + " loss " + str(loss))
                        loss_list[t * data_size + j] = loss
                        #print("gradient:", dy)
                        #print("predict score:", y)
                    """
                    if (t * data_size + j == 499 or t * data_size + j == 999 or t * data_size + j == 1499) and rank == 0:
                        loss_file = open("loss" + str(t * data_size + j) + ".txt", 'w')
                        time_file = open("time" + str(t * data_size + j) + ".txt", 'w')
                        for i in range(t * data_size + j + 1):
                            loss_file.write(str(loss_list[i]) + "\n")
                            time_file.write(str(time_list[i].total_seconds()) + "\n")
                        loss_file.close()
                        time_file.close()
                    """
                #backprop of one data instance
                for i, node in enumerate(self._node_list[::-1]):
                    if node is not None and type(node) is not str:
                        if type(node) is DecentralMdsFc or type(node) is DecentralReplicaFc or type(node) is DecentralUncodedFc:
                            dx, fail = node.backprop(dy, t * data_size * layer_num + j * layer_num + i + iter * data_size * layer_num, generate_error, True)
                        else:
                            dx = node.backprop(dy, t * data_size * layer_num + j * layer_num + i + iter * data_size * layer_num)
                        dy = dx
                    if fail:
                        break
                if fail:
                    continue
                
                if t == iter - 1 and j == data_size - 1:
                    #make sure no errors before ending training
                    layer_num = len(self._node_list)
                    x = None
                    y = None
                    if X is not None:
                        x = np.ones(X[j].shape)
                    for i, node in enumerate(self._node_list):
                        if type(node) is DecentralMdsFc or type(node) is DecentralReplicaFc:
                            y, fail = node.forward(x, t * data_size * layer_num + j * layer_num + i + 2 * iter * data_size * layer_num, False, False)
                        if y is not None:
                            x = np.ones(y.shape)
                        if fail:
                            break
                    if fail:
                        continue
                    dy = x
                    dx = None
                    for i, node in enumerate(self._node_list[::-1]):
                        if type(node) is DecentralMdsFc:
                            dx, fail = node.backprop(dy, t * data_size * layer_num + j * layer_num + i + 3 * iter * data_size * layer_num, False, False)
                        if dx is not None:
                            dy = np.ones(dx.shape)
                        if fail:
                            break
                    if fail:
                        continue
                j += 1
            t += 1
        if rank == 0:
            time = datetime.now() - self._start_time
            if self._store_init_time is not None:
                time -= self._store_init_time
            print(time.total_seconds())
            time_list[iter * data_size - 1] = time
        return loss_list, time_list
    
    def predict(self, X, data_size, checkpoint_iter=100):
        if rank == 0:
            print("predicting...")
            Y = [None] * data_size
        else:
            Y = [None] * data_size
        check_point_t = -1 #last j.
        fail = False
        j = 0
        while j < data_size:
            if fail:
                #roll back
                fail = False
                self.loadW()
                j = check_point_t
                if rank == 0:
                    print("roll back")
            if j%checkpoint_iter == 0 and j != check_point_t:
                #check-pointing
                #self.saveW() currently no errors are introduced in prediction, thus no need to checkpoint
                check_point_t = j
            #forward of one data instance
            x = None
            if X is not None:
                x = X[j]
            y = None
            layer_num = len(self._node_list)
            for i, node in enumerate(self._node_list):
                if node is not None and type(node) is not str:
                    req = None
                    if type(node) is DecentralMdsFc or type(node) is DecentralReplicaFc or type(node) is DecentralUncodedFc:
                        y, fail = node.forward(x, j * layer_num + i, False, True)
                    else:
                        y = node.forward(x, j * layer_num + i)
                    x = y
                if fail:
                    break
            if fail:
                continue
            
            #predict class
            if y is not None:
                y -= np.amax(y)
                p = np.exp(y) / np.sum(np.exp(y))
                Y[j] = np.argmax(p)
            
            j += 1
        
        if y is not None:
            return np.array(Y)
        return None
    
    
    def debug(self):
        print(rank, self._p_num, self._p_count, self._node_list, self._used_by, self._current_node)
        for node in self._node_list:
            if node is not None:
                node.debug()


class Node(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, processors_id, previous, next, N, pre_N):
        self._processors_id = processors_id
        self._previous = previous
        self._next = next
        self._N = N
        self._pre_N = pre_N
    
    
    @abc.abstractmethod
    def init(self, layer_index):
        pass
    
    @abc.abstractmethod
    #return y
    def forward(self, x, tag):
        pass
    
    @abc.abstractmethod
    #return dx
    def backprop(self, dy, tag):
        pass
    
    @abc.abstractmethod
    def saveW(self, processor_id, node_index):
        pass
    
    @abc.abstractmethod
    def loadW(self, processor_id, node_index):
        pass


class Sigmoid(Node):
    def __init__(self):
        self._x = None
    
    def init(self, layer_index):
        pass
    
    def forward(self, x, tag):
        self._x = x
        return 1.0 / (1.0 + np.exp(-1.0 * x))
    
    def backprop(self, dy, tag):
        s = 1.0 / (1.0 + np.exp(-1.0 * self._x))
        return s * (1 - s) * dy
    
    def saveW(self, processor_id, node_index):
        pass
    
    def loadW(self, processor_id, node_index):
        pass
    
    def outputTime(self, data_size, iter, id):
        pass


class Relu(Node):
    def __init__(self):
        self._x = None
    
    def init(self, layer_index):
        pass
    
    def forward(self, x, tag):
        self._x = x
        return np.maximum(np.zeros(x.shape), x)
    
    def backprop(self, dy, tag):
        ori_shape = dy.shape
        length = (np.multiply.reduce(dy.shape),)
        dx = dy.reshape(length)
        for i, e in enumerate(self._x.reshape(length)):
            if e < 0.0:
                dx[i] = 0.0
        return dx.reshape(ori_shape)
    
    def saveW(self, processor_id, node_index):
        pass
    
    def loadW(self, processor_id, node_index):
        pass
    
    def outputTime(self, data_size, iter, id):
        pass


class Bias(Node):
    def __init__(self, alpha, beta1, beta2, processors_id, previous, next, N, pre_N):
        super(Bias, self).__init__(processors_id, previous, next, N, pre_N)
        self._b = None
        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        self._iter = None
        self._m = None
        self._v = None
    
    def init(self, layer_index):
        self._b = np.zeros(self._N, dtype='d')
        self._iter = 0
        self._m = np.zeros(self._N, dtype='d')
        self._v = np.zeros(self._N, dtype='d')
    
    def forward(self, x, tag):
        return x + self._b
    
    def backprop(self, dy, tag):
        self._iter += 1
        rate = 1.0
        """
        if 500 < self._iter <= 1000:
            rate = 1.0
        elif 1000 < self._iter <= 1500:
            rate = 1.0
        elif 1500 < self._iter <= 2000:
            rate = 0.1
        elif 2000 < self._iter:
            rate = 0.1
        """
        self._b += rate * self._alpha * dy
        """
        self._m = self._beta1 * self._m + self._alpha * dy
        self._b += self._m
        """
        """
        self._iter += 1
        self._m = self._beta1 * self._m + (1 - self._beta1) * grad
        self._v = self._beta2 * self._v + (1 - self._beta2) * (grad * grad)
        tmp_m = self._m / (1 - self._beta1 ** self._iter)
        tmp_v = self._v / (1 - self._beta2 ** self._iter)
        self._b += self._alpha * tmp_m / (np.sqrt(tmp_v) + 1e-8)
        """
        return dy
    
    def saveW(self, processor_id, node_index):
        np.savez("./checkpointing/" + str(processor_id) + "_" + str(node_index) + ".npz", b=self._b, iter=np.array([self._iter]))
    
    def loadW(self, processor_id, node_index):
        with np.load("./checkpointing/" + str(processor_id) + "_" + str(node_index) + ".npz") as data:
            self._b = data['b']
            self._iter = data['iter'][0]
    
    def outputTime(self, data_size, iter, id):
        pass


class ReshapeTensor(Node):
    def __init__(self, N, pre_N):
        self._N = N
        self._pre_N = pre_N
    
    def init(self, layer_index):
        pass
    
    def forward(self, x, tag):
        return x.reshape(self._N)
    
    def backprop(self, dy, tag):
        return dy.reshape(self._pre_N)
    
    def saveW(self, processor_id, node_index):
        pass
    
    def loadW(self, processor_id, node_index):
        pass
    
    def outputTime(self, data_size, iter, id):
        pass


class DecentralMds(Node):
    __metaclass__ = abc.ABCMeta

    def __init__(self, Pr, qr, Pc, qc, P, alpha, beta1, beta2, M, error_rate, node_comm, processors_id, previous, next, N, pre_N):
        super(DecentralMds, self).__init__(processors_id, previous, next, N, pre_N)
        self._M = M
        self._total_N = np.multiply.reduce(N)
        self._total_pre_N = np.multiply.reduce(pre_N)
        self._total_M = np.multiply.reduce(M)
        self._N_seg = (N[0]//qr,) + N[1:]
        self._pre_N_seg = (pre_N[0]//qc,) + pre_N[1:]
        self._M_seg = (M[0]//qr, M[1]//qc) + M[2:]
        self._total_N_seg = self._total_N // qr
        self._total_pre_N_seg = self._total_pre_N // qc
        self._total_M_seg = self._total_M // (qr*qc)
        self._fan_in = None
        self._fan_out = None
        self._W = None
        self._coded_W = None
        self._coded_x = None
        self._coded_dy = None
        self._x = None
        self._Pr = Pr
        self._qr = qr
        self._Pc = Pc
        self._qc = qc
        self._P = P
        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        self._iter = None
        self._m = None
        self._v = None
        self._error_rate = error_rate
        self._node_comm = node_comm
        self._Gr = np.empty(Pr * qr, dtype='d').reshape((Pr, qr))
        self._Gc = np.empty(Pc * qc, dtype='d').reshape((Pc, qc))
        if self._node_comm.rank == 0:
            self._Gr = np.concatenate((np.identity(qr), np.random.randn(Pr - qr, qr)), axis=0)
            self._Gc = np.concatenate((np.identity(qc), np.random.randn(Pc - qc, qc)), axis=0)
        self._node_comm.Bcast([self._Gr, MPI.DOUBLE], root=0)
        self._node_comm.Bcast([self._Gc, MPI.DOUBLE], root=0)
        self._forward_sum_comm = MPI.COMM_NULL
        for i in range(Pr):
            include = [self.getProcessorId((i, j)) for j in range(qc)]
            new_group = self._node_comm.Get_group().Incl(include)
            new_comm = self._node_comm.Create(new_group)
            if new_comm != MPI.COMM_NULL:
                self._forward_sum_comm = new_comm
        self._forward_recovery_comm = MPI.COMM_NULL
        for i in range(qc):
            include = [self.getProcessorId((j, i)) for j in range(Pr)]
            new_group = self._node_comm.Get_group().Incl(include)
            new_comm = self._node_comm.Create(new_group)
            if new_comm != MPI.COMM_NULL:
                self._forward_recovery_comm = new_comm
        self._backprop_sum_comm = MPI.COMM_NULL
        for i in range(Pc):
            include = [self.getProcessorId((j, i)) for j in range(qr)]
            new_group = self._node_comm.Get_group().Incl(include)
            new_comm = self._node_comm.Create(new_group)
            if new_comm != MPI.COMM_NULL:
                self._backprop_sum_comm = new_comm
        self._backprop_recovery_comm = MPI.COMM_NULL
        for i in range(qr):
            include = [self.getProcessorId((i, j)) for j in range(Pc)]
            new_group = self._node_comm.Get_group().Incl(include)
            new_comm = self._node_comm.Create(new_group)
            if new_comm != MPI.COMM_NULL:
                self._backprop_recovery_comm = new_comm
        self._forward_total_time = []
        self._forward_dot_time = []
        self._forward_sum_time = []
        self._forward_decode_time = []
        self._backprop_total_time = []
        self._backprop_dot_time = []
        self._backprop_sum_time = []
        self._backprop_decode_time = []
        self._update_time = []
    
    def getGridId(self, node_rank):
        if node_rank >= self._qr * self._Pc + (self._Pr - self._qr) * self._qc:
            return (-1, -1) #current processor is not used in this layer
        if node_rank < self._qr * self._Pc:
            return (node_rank // self._Pc, node_rank % self._Pc)
        return (self._qr + (node_rank - self._qr * self._Pc) // self._qc, \
                (node_rank - self._qr * self._Pc) % self._qc)
    
    def getProcessorId(self, grid):
        r, c = grid
        if r == -1 and c == -1: #current processor is not used in this layer
            return self._qr * self._Pc + (self._Pr - self._qr) * self._qc
        if r < self._qr:
            return r * self._Pc + c
        return (r - self._qr) * self._qc + c + self._qr * self._Pc
    
    def encodeW(self, layer_index):
        r, c = self.getGridId(self._node_comm.rank)
        if -1 < r and r < self._qr and -1 < c and c < self._qc:
            #self._W = np.random.random_sample(self._M_seg) * 0.1 - 0.05
            stddev = np.sqrt(2.0 / (self._fan_in + self._fan_out))
            self._W = np.random.normal(loc=0.0, scale=stddev, size=self._M_seg)
            send_req = []
            for i in range(self._qc, self._Pc):
                send_req.append(self._node_comm.Isend([self._W,  MPI.DOUBLE], dest=self.getProcessorId((r, i))))
            for i in range(self._qr, self._Pr):
                send_req.append(self._node_comm.Isend([self._W,  MPI.DOUBLE], dest=self.getProcessorId((i, c))))
            MPI.Request.Waitall(send_req)
        elif r >= self._qr:
            self._coded_W = np.zeros(self._total_M_seg, dtype='d').reshape(self._M_seg)
            tmp = np.empty(self._total_M_seg, dtype='d').reshape(self._M_seg)
            for i in range(self._qr):
                self._node_comm.Recv([tmp, MPI.DOUBLE], source=self.getProcessorId((i, c)))
                self._coded_W += self._Gr[r, i] * tmp
            self._W = self._coded_W
        elif c >= self._qc:
            self._coded_W = np.zeros(self._total_M_seg, dtype='d').reshape(self._M_seg)
            tmp = np.empty(self._total_M_seg, dtype='d').reshape(self._M_seg)
            for i in range(self._qc):
                self._node_comm.Recv([tmp, MPI.DOUBLE], source=self.getProcessorId((r, i)))
                self._coded_W += self._Gc[c, i] * tmp
            self._W = self._coded_W
        """
        counts = (self._total_M_seg,) * self._qc + (0,) * (self._Pc - self._qc)
        counts *= self._qr
        counts += (0,) * (self._P - (self._qr * self._Pc))
        dspls = ()
        for i in range(self._qr):
            for j in range(self._qc):
                dspls += ((i * self._qc + j) * self._total_M_seg,)
            dspls += ((i+1) * self._qc * self._total_M_seg,) * (self._Pc - self._qc)
        dspls += (self._total_M,) * (self._P - (self._qr * self._Pc))
        if -1 < r and r < self._qr and -1 < c and c < self._qc:
            senddata = self._W
        else:
            senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        W_seg = np.empty(self._total_M, dtype='d').reshape((self._qr, self._qc) + self._M_seg)
        self._node_comm.Gatherv([senddata, counts[self._node_comm.rank]], \
                                   [W_seg, counts, dspls, MPI.DOUBLE], root=0)
        if self._node_comm.rank == 0:
            W = np.empty(self._total_M, dtype='d').reshape(self._M)
            for i in range(self._qr):
                for j in range(self._qc):
                    W[i * self._M_seg[0] : (i+1) * self._M_seg[0], j * self._M_seg[1] : (j+1) * self._M_seg[1]] = W_seg[i, j]
            np.save("./init/W" + str(layer_index) + ".npy", W)
        """
        """
        start = datetime.now()
        """
        if -1 < r and r < self._qr and -1 < c and c < self._qc:
            np.save("./init/W_" + str(r) + "_" + str(c) + "_" + str(layer_index) + ".npy", self._W)
        """
        if self._store_init_time is None:
            self._store_init_time = datetime.now() - start
        else:
            self._store_init_time += datetime.now() - start
        """
 
    def init(self, layer_index):
        self.encodeW(layer_index)
        self._iter = 0
        self._m = np.zeros(self._total_M_seg, dtype='d').reshape(self._M_seg)
        self._v = np.zeros(self._total_M_seg, dtype='d').reshape(self._M_seg)
        self._forward_total_time = []
        self._forward_dot_time = []
        self._forward_sum_time = []
        self._forward_decode_time = []
        self._backprop_total_time = []
        self._backprop_dot_time = []
        self._backprop_sum_time = []
        self._backprop_decode_time = []
        self._update_time = []
    
    @abc.abstractmethod
    def forwardCompute(self, x, generate_error):
        pass
    
    @abc.abstractmethod
    def backpropCompute(self, dy, generate_error):
        pass
    
    @abc.abstractmethod
    def update(self, x, dy, generate_error):
        pass
    
    def regenerate(self, error_index, recovery_comm, P, q, G, tag):
        if recovery_comm != MPI.COMM_NULL:
            if recovery_comm.rank == error_index:
                print(error_index)
                count = 0
                i = 0
                c = None
                rows = np.empty(q, dtype='i')
                while count < q and i < P:
                    if i == error_index:
                        i += 1
                        continue
                    recovery_comm.send('r', dest=i, tag=tag)
                    c_seg = np.empty(self._total_M_seg, dtype='d').reshape(self._M_seg)
                    recovery_comm.Recv([c_seg, MPI.DOUBLE], source=i, tag=tag)
                    if c is None:
                        c = c_seg.reshape((1,) + self._M_seg)
                    else:
                        c = np.vstack((c, c_seg.reshape((1,) + self._M_seg)))
                    rows[count] = i
                    count += 1
                    i += 1
                print("send f")
                while i < P:
                    if i == error_index:
                        i += 1
                        continue
                    recovery_comm.send('f', dest=i, tag=tag)
                    i += 1
                print("recover")
                decode_mat = np.linalg.inv(G[rows, :])
                recovery_vec = np.dot(G[error_index, :], decode_mat)
                self._W = np.tensordot(recovery_vec, c, (0, 0)).reshape(self._M_seg)
                print("recover finish")
            else:
                status = MPI.Status()
                i = 0
                while i < 1:
                    flag = recovery_comm.recv(source=MPI.ANY_SOURCE, tag=tag, status=status)
                    if flag == 'f':
                        i += 1
                        continue
                    recovery_comm.Send([self._W, MPI.DOUBLE], dest=status.Get_source(), tag=tag)
                    i += 1
    
    def decode(self, coded_res, total_seg, recovery_comm, P, q, G, tag):
        error_threshold = 1e-9
        decode_mat = None
        decode_subset = None
        error_index = None
        has_error = False
        #detect error
        for extra_index in range(P):
            subset_index = range(P)
            subset_index.remove(extra_index)
            subset = coded_res[subset_index]
            G_sub = G[subset_index[0:q]]
            G_sub_inv = np.linalg.inv(G_sub)
            vec = G[subset_index[-1]]
            linear_comb = np.dot(vec, G_sub_inv)
            computed_last_piece = np.tensordot(linear_comb, subset[0:q], (0,0)).reshape(coded_res.shape[1:])
            diff = computed_last_piece - subset[-1]
            if np.amax(diff) <= error_threshold and np.amin(diff) >= -error_threshold:
                decode_mat = G_sub_inv
                decode_subset = subset[0:q]
                error_index = extra_index
            else:
                has_error = True
        #has more than 1 error
        if has_error and decode_mat is None:
            return None, True
        #has 1 error
        if has_error and decode_mat is not None:
            self.regenerate(error_index, recovery_comm, P, q, G, tag)
        #has 1 error or no error
        res = np.tensordot(decode_mat, decode_subset, (1,0)).reshape((coded_res.shape[1] * q,) + coded_res.shape[2:])
        return res, False
    
    def forward(self, x, tag, generate_error, test):
        forward_start = datetime.now()
        self._x = x
        r, c = self.getGridId(self._node_comm.rank)
        if -1 < c and c < self._qc:
            #introduce error to W, compute unsum_coded_y_seg
            forward_cal_start = datetime.now()
            unsum_coded_y_seg = self.forwardCompute(x, generate_error, test)
            self._forward_dot_time.append(datetime.now() - forward_cal_start)
            
            #reduction(summation)
            forward_sum_start = datetime.now()
            coded_y_seg = np.empty(self._total_N_seg, dtype='d').reshape(self._N_seg)
            self._forward_sum_comm.Reduce(unsum_coded_y_seg, coded_y_seg, op=MPI.SUM, root=0)
            self._forward_sum_time.append(datetime.now() - forward_sum_start)
            
        #get coded_y and send it to all processors
        counts = ((self._total_N_seg,) + (0,) * (self._Pc - 1)) * self._qr
        counts += ((self._total_N_seg,) + (0,) * (self._qc - 1)) * (self._Pr - self._qr)
        counts += (0,) * (self._P - (self._qr * self._Pc + (self._Pr - self._qr) * self._qc))
        dspls = ()
        for i in range(self._qr):
            dspls += (i * self._total_N_seg,) + ((i + 1) * self._total_N_seg,) * (self._Pc - 1)
        for i in range(self._qr, self._Pr):
            dspls += (i * self._total_N_seg,) + ((i + 1) * self._total_N_seg,) * (self._qc - 1)
        dspls += (self._Pr * self._total_N_seg,) * (self._P - (self._qr * self._Pc + (self._Pr - self._qr) * self._qc))
        if self._forward_sum_comm != MPI.COMM_NULL:
            if self._forward_sum_comm.rank == 0:
                senddata = coded_y_seg
            else:
                senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        else:
            senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        coded_y = np.empty(self._total_N_seg * self._Pr, dtype='d').reshape((self._Pr,) + self._N_seg)
        self._node_comm.Allgatherv([senddata, counts[self._node_comm.rank]], \
                                   [coded_y, counts, dspls, MPI.DOUBLE])
        
        #detect error, regenerate W, decode to get y
        forward_decode_start = datetime.now()
        y, fail = self.decode(coded_y, self._total_N_seg, self._forward_recovery_comm, self._Pr, self._qr, self._Gr, tag)
        self._forward_decode_time.append(datetime.now() - forward_decode_start)
        self._forward_total_time.append(datetime.now() - forward_start)
        return y, fail
    
    def backprop(self, dy, tag, generate_error, update):
        backprop_start = datetime.now()
        self._dy = dy
        r, c = self.getGridId(self._node_comm.rank)
        if -1 < r and r < self._qr:
            #introduce error to W, compute unsum_coded_dx_seg
            backprop_cal_start = datetime.now()
            unsum_coded_dx_seg = self.backpropCompute(dy, generate_error)
            self._backprop_dot_time.append(datetime.now() - backprop_cal_start)
            
            #reduction(summation)
            backprop_sum_start = datetime.now()
            coded_dx_seg = np.empty(self._total_pre_N_seg, dtype='d').reshape(self._pre_N_seg)
            self._backprop_sum_comm.Reduce(unsum_coded_dx_seg, coded_dx_seg, op=MPI.SUM, root=0)
            self._backprop_sum_time.append(datetime.now() - backprop_sum_start)
            
        #get coded_dx and send it to all processors
        counts = (self._total_pre_N_seg,) * self._Pc
        counts += (0,) * (self._P - self._Pc)
        dspls = ()
        for i in range(self._Pc):
            dspls += (i * self._total_pre_N_seg,)
        dspls += (self._Pc * self._total_pre_N_seg,) * (self._P - self._Pc)
        if self._backprop_sum_comm != MPI.COMM_NULL:
            if self._backprop_sum_comm.rank == 0:
                senddata = coded_dx_seg
            else:
                senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        else:
            senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        coded_dx = np.empty(self._total_pre_N_seg * self._Pc, dtype='d').reshape((self._Pc,) + self._pre_N_seg)
        self._node_comm.Allgatherv([senddata, counts[self._node_comm.rank]], \
                                   [coded_dx, counts, dspls, MPI.DOUBLE])
        
        #detect error, regenerate W, decode to get dx
        backprop_decode_start = datetime.now()
        dx, fail = self.decode(coded_dx, self._total_pre_N_seg, self._backprop_recovery_comm, self._Pc, self._qc, self._Gc, tag)
        self._backprop_decode_time.append(datetime.now() - backprop_decode_start)
        
        #update with error
        if update:
            update_start = datetime.now()
            self.update(self._x, self._dy, generate_error)
            self._update_time.append(datetime.now() - update_start)
        self._backprop_total_time.append(datetime.now() - backprop_start)
        return dx, fail
    
    def saveW(self, processor_id, node_index):
        np.savez("./checkpointing/" + str(processor_id) + "_" + str(node_index) + ".npz", W=self._W, iter=np.array([self._iter]))#, m=self._m)
    
    def loadW(self, processor_id, node_index):
        with np.load("./checkpointing/" + str(processor_id) + "_" + str(node_index) + ".npz") as data:
            self._W = data['W']
            self._iter = data['iter'][0]
            #self._m = data['m']
    
    def outputTime(self, data_size, iter=100, id=0):
        file = open(str(id) + str(rank) + '_' + str(self.getGridId(self._node_comm.rank)) + '_coded_time.csv', 'w')
        file.write("_forward_total_time,_forward_dot_time,_forward_sum_time,_forward_decode_time,_backprop_total_time,_backprop_dot_time,_backprop_sum_time,_backprop_decode_time,_update_time\n")
        #print(len(self._forward_total_time))
        
        for i in range(data_size * iter):
            if self._forward_total_time:
                file.write(str(self._forward_total_time[i].total_seconds()))
            file.write(',')
            if self._forward_dot_time:
                file.write(str(self._forward_dot_time[i].total_seconds()))
            file.write(',')
            if self._forward_sum_time:
                file.write(str(self._forward_sum_time[i].total_seconds()))
            file.write(',')
            if self._forward_decode_time:
                file.write(str(self._forward_decode_time[i].total_seconds()))
            file.write(',')
            if self._backprop_total_time:
                file.write(str(self._backprop_total_time[i].total_seconds()))
            file.write(',')
            if self._backprop_dot_time:
                file.write(str(self._backprop_dot_time[i].total_seconds()))
            file.write(',')
            if self._backprop_sum_time:
                file.write(str(self._backprop_sum_time[i].total_seconds()))
            file.write(',')
            if self._backprop_decode_time:
                file.write(str(self._backprop_decode_time[i].total_seconds()))
            file.write(',')
            if self._update_time:
                file.write(str(self._update_time[i].total_seconds()))
            file.write('\n')
    
    def debug(self):
        for j in range(15):
            if rank == j:
                print(rank, self._processors_id, self._previous, self._next, self._N, self._pre_N, \
                      self._Pr, self._qr, self._Pc, self._qc, self._node_comm, self._Gr, self._Gc)
            time.sleep(1)
    
    def encodeTest(self):
        #self.encodeW()
        for j in range(15):
            if rank == j:
                if self._W is not None:
                    print(self.getGridId(self._node_comm.rank), self._W)
                else:
                    print(self.getGridId(self._node_comm.rank), self._coded_W)
            time.sleep(0.2)


class DecentralMdsFc(DecentralMds):
    def __init__(self, Pr, qr, Pc, qc, P, alpha, beta1, beta2, M, error_rate, node_comm, processors_id, previous, next, N, pre_N):
        super(DecentralMdsFc, self).__init__(Pr, qr, Pc, qc, P, alpha, beta1, beta2, M, error_rate, node_comm, processors_id, previous, next, N, pre_N)
        self._fan_in = self._total_pre_N
        self._fan_out = self._total_N
    
    def forwardCompute(self, x, generate_error, test):
        r, c = self.getGridId(self._node_comm.rank)
        if not test:
            random_number = np.random.random_sample()
            if random_number < self._error_rate and generate_error:
                error = scipy.sparse.rand(self._M_seg[0], self._M_seg[1], density=0.005)
                error = 2.0 * error - error.ceil()
                self._W += error.toarray() * 5.0
                print(np.amax(error.toarray()), np.amin(error.toarray()))
                print("forward error")
        unsum_coded_y_seg = np.dot(self._W, x[c * self._pre_N_seg[0] : (c+1) * self._pre_N_seg[0]])
        return unsum_coded_y_seg
    
    def backpropCompute(self, dy, generate_error):
        r, c = self.getGridId(self._node_comm.rank)
        random_number = np.random.random_sample()
        if random_number < self._error_rate and generate_error:
            error = scipy.sparse.rand(self._M_seg[0], self._M_seg[1], density=0.005)
            error = 2.0 * error - error.ceil()
            self._W += error.toarray() * 5.0
            print(np.amax(error.toarray()), np.amin(error.toarray()))
            print("backprop error")
        unsum_coded_dx_seg = np.dot(dy[r * self._N_seg[0] : (r+1) * self._N_seg[0]], self._W)
        return unsum_coded_dx_seg
    
    def update(self, x, dy, generate_error):
        r, c = self.getGridId(self._node_comm.rank)
        if r == -1 and c == -1:
            return
        x_seg = x.reshape((self._qc,) + self._pre_N_seg)
        coded_x_seg = np.dot(self._Gc[c, :], x_seg)
        dy_seg = dy.reshape((self._qr,) + self._N_seg)
        coded_dy_seg = np.dot(self._Gr[r, :], dy_seg)
        self._iter += 1
        rate = 1.0
        """
        if self._iter > 1500:
            rate = 0.1
        """
        
        self._W += rate * self._alpha * np.outer(coded_dy_seg, coded_x_seg)
        """
        self._m = self._beta1 * self._m + self._alpha * np.outer(coded_dy_seg, coded_x_seg)
        self._W += self._m
        """
        """
        self._iter += 1
        self._m = self._beta1 * self._m + (1 - self._beta1) * grad
        self._v = self._beta2 * self._v + (1 - self._beta2) * (grad * grad)
        tmp_m = self._m / (1 - self._beta1 ** self._iter)
        tmp_v = self._v / (1 - self._beta2 ** self._iter)
        self._W += self._alpha * tmp_m / (np.sqrt(tmp_v) + 1e-8)
        """
        random_number = np.random.random_sample()
        if random_number < self._error_rate and generate_error:
            error = scipy.sparse.rand(self._M_seg[0], self._M_seg[1], density=0.005)
            error = 2.0 * error - error.ceil()
            self._W += error.toarray() * 5.0
            print(np.amax(error.toarray()), np.amin(error.toarray()))
            print("update error")


class DecentralReplica(Node):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, Pr, Pc, P, alpha, beta1, beta2, M, error_rate, node_comm, processors_id, previous, next, N, pre_N):
        super(DecentralReplica, self).__init__(processors_id, previous, next, N, pre_N)
        self._M = M
        self._total_N = np.multiply.reduce(N)
        self._total_pre_N = np.multiply.reduce(pre_N)
        self._total_M = np.multiply.reduce(M)
        self._N_seg = (N[0]//Pr,) + N[1:]
        self._pre_N_seg = (pre_N[0]//Pc,) + pre_N[1:]
        self._M_seg = (M[0]//Pr, M[1]//Pc) + M[2:]
        self._total_N_seg = self._total_N // Pr
        self._total_pre_N_seg = self._total_pre_N // Pc
        self._total_M_seg = self._total_M // (Pr*Pc)
        self._W = None
        self._x = None
        self._Pr = Pr
        self._Pc = Pc
        self._P = P
        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        self._iter = None
        self._m = None
        self._v = None
        self._error_rate = error_rate
        self._node_comm = node_comm
        self._forward_sum_comm = MPI.COMM_NULL
        for sys in [0, 1]:
            for i in range(Pr):
                include = [self.getProcessorId((i, j, sys)) for j in range(Pc)]
                new_group = self._node_comm.Get_group().Incl(include)
                new_comm = self._node_comm.Create(new_group)
                if new_comm != MPI.COMM_NULL:
                    self._forward_sum_comm = new_comm
        self._backprop_sum_comm = MPI.COMM_NULL
        for sys in [0, 1]:
            for i in range(Pc):
                include = [self.getProcessorId((j, i, sys)) for j in range(Pr)]
                new_group = self._node_comm.Get_group().Incl(include)
                new_comm = self._node_comm.Create(new_group)
                if new_comm != MPI.COMM_NULL:
                    self._backprop_sum_comm = new_comm
        self._forward_total_time = []
        self._forward_tosink_time = []
        self._forward_dot_time = []
        self._forward_sum_time = []
        self._backprop_total_time = []
        self._backprop_tosink_time = []
        self._backprop_dot_time = []
        self._backprop_sum_time = []
        self._update_time = []

    def getGridId(self, node_rank):
        if node_rank >= self._Pr * self._Pc * 2:
            return (-1, -1, -1) #current processor is not used in this layer
        if 0 <= node_rank and node_rank < self._Pr * self._Pc:
            return (node_rank // self._Pc, node_rank % self._Pc, 0)
        return ((node_rank-self._Pr*self._Pc) // self._Pc, (node_rank-self._Pr*self._Pc) % self._Pc, 1)
    
    def getProcessorId(self, grid):
        r, c, sys = grid
        if r == -1 and c == -1 and sys == -1:
            return self._Pr * self._Pc * 2 #current processor is not used in this layer
        return r * self._Pc + c + sys * self._Pr * self._Pc
    
    def init(self, layer_index):
        r, c, sys = self.getGridId(self._node_comm.rank)
        """
        if sys == 0:
            self._W = np.random.random_sample(self._M_seg) * 0.01
            self._node_comm.Send([self._W, MPI.DOUBLE], dest=self.getProcessorId((r, c, 1)))
        elif sys == 1:
            self._W = np.empty(self._total_M_seg, dtype='d').reshape(self._M_seg)
            self._node_comm.Recv([self._W, MPI.DOUBLE], source=self.getProcessorId((r, c, 0)))
        """
        if r > -1 and c > -1:
            self._W = np.load("./init/W_" + str(r) + "_" + str(c) + "_" + str(layer_index) + ".npy")
        self._iter = 0
        self._m = np.zeros(self._total_M_seg, dtype='d').reshape(self._M_seg)
        self._v = np.zeros(self._total_M_seg, dtype='d').reshape(self._M_seg)
        self._forward_total_time = []
        self._forward_tosink_time = []
        self._forward_dot_time = []
        self._forward_sum_time = []
        self._backprop_total_time = []
        self._backprop_tosink_time = []
        self._backprop_dot_time = []
        self._backprop_sum_time = []
        self._update_time = []
    
    @abc.abstractmethod
    def forwardCompute(self, x, generate_error):
        pass
    
    @abc.abstractmethod
    def backpropCompute(self, dy, generate_error):
        pass
    
    @abc.abstractmethod
    def update(self, x, dy, generate_error):
        pass
    
    def forward(self, x, tag, generate_error, test):
        forward_start = datetime.now()
        self._x = x
        r, c, sys = self.getGridId(self._node_comm.rank)
        if -1 < c and c < self._Pc:
            #introduce error to W, compute unsum_y_seg
            forward_cal_start = datetime.now()
            unsum_y_seg = self.forwardCompute(x, generate_error, test)
            self._forward_dot_time.append(datetime.now() - forward_cal_start)
            
            #reduction(summation)
            forward_sum_start = datetime.now()
            y_seg = np.empty(self._total_N_seg, dtype='d').reshape(self._N_seg)
            self._forward_sum_comm.Reduce(unsum_y_seg, y_seg, op=MPI.SUM, root=0)
            self._forward_sum_time.append(datetime.now() - forward_sum_start)
            
        #get coded_y and send it to all processors
        counts = ((self._total_N_seg,) + (0,) * (self._Pc - 1)) * (self._Pr * 2)
        counts += (0,) * (self._P - (self._Pr * self._Pc * 2))
        dspls = ()
        for i in range(self._Pr * 2):
            dspls += (i * self._total_N_seg,) + ((i + 1) * self._total_N_seg,) * (self._Pc - 1)
        dspls += (self._Pr * 2 * self._total_N_seg,) * (self._P - (self._Pr * self._Pc * 2))
        if self._forward_sum_comm != MPI.COMM_NULL:
            if self._forward_sum_comm.rank == 0:
                senddata = y_seg
            else:
                senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        else:
            senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        y_segs = np.empty(self._total_N_seg * self._Pr * 2, dtype='d').reshape((self._Pr * 2,) + self._N_seg)
        self._node_comm.Allgatherv([senddata, counts[self._node_comm.rank]], \
                                   [y_segs, counts, dspls, MPI.DOUBLE])
        
        #detect error, get y
        y1 = y_segs[0 : self._Pr].reshape(self._N)
        y2 = y_segs[self._Pr : self._Pr * 2].reshape(self._N)
        diff = y1 - y2
        error_threshold = 1e-9
        if np.amax(diff) <= error_threshold and np.amin(diff) >= -error_threshold:
            self._forward_total_time.append(datetime.now() - forward_start)
            return y1, False
        else:
            self._forward_total_time.append(datetime.now() - forward_start)
            return None, True
    
    def backprop(self, dy, tag, generate_error, update):
        backprop_start = datetime.now()
        self._dy = dy
        r, c, sys = self.getGridId(self._node_comm.rank)
        if -1 < r and r < self._Pr:
            #introduce error to W, compute unsum_dx_seg
            backprop_cal_start = datetime.now()
            unsum_dx_seg = self.backpropCompute(dy, generate_error)
            self._backprop_dot_time.append(datetime.now() - backprop_cal_start)
            
            #reduction(summation)
            backprop_sum_start = datetime.now()
            dx_seg = np.empty(self._total_pre_N_seg, dtype='d').reshape(self._pre_N_seg)
            self._backprop_sum_comm.Reduce(unsum_dx_seg, dx_seg, op=MPI.SUM, root=0)
            self._backprop_sum_time.append(datetime.now() - backprop_sum_start)
            
        #get coded_dx and send it to all processors
        counts = (self._total_pre_N_seg,) * self._Pc
        counts += (0,) * ((self._Pr - 1) * self._Pc)
        counts *= 2
        counts += (0,) * (self._P - (self._Pr * self._Pc * 2))
        dspls = ()
        for i in range(self._Pc):
            dspls += (i * self._total_pre_N_seg,)
        dspls += (self._Pc * self._total_pre_N_seg,) * ((self._Pr - 1) * self._Pc)
        for i in range(self._Pc, self._Pc * 2):
            dspls += (i * self._total_pre_N_seg,)
        dspls += (self._Pc * 2 * self._total_pre_N_seg,) * (self._P - ((self._Pr + 1) * self._Pc))
        if self._backprop_sum_comm != MPI.COMM_NULL:
            if self._backprop_sum_comm.rank == 0:
                senddata = dx_seg
            else:
                senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        else:
            senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        dx_segs = np.empty(self._total_pre_N_seg * self._Pc * 2, dtype='d').reshape((self._Pc * 2,) + self._pre_N_seg)
        self._node_comm.Allgatherv([senddata, counts[self._node_comm.rank]], \
                                   [dx_segs, counts, dspls, MPI.DOUBLE])
        
        #detect error, get dx
        dx1 = dx_segs[0 : self._Pc].reshape(self._pre_N)
        dx2 = dx_segs[self._Pc : self._Pc * 2].reshape(self._pre_N)
        diff = dx1 - dx2
        error_threshold = 1e-9
        dx = None
        fail = None
        if np.amax(diff) <= error_threshold and np.amin(diff) >= -error_threshold:
            dx = dx1
            fail = False
        else:
            if rank == 0:
                print(np.amax(diff), np.amin(diff))
            dx = None
            fail = True

        #update with error
        if update:
            update_start = datetime.now()
            self.update(self._x, self._dy, generate_error)
            self._update_time.append(datetime.now() - update_start)
        self._backprop_total_time.append(datetime.now() - backprop_start)
        return dx, fail

    def saveW(self, processor_id, node_index):
        np.savez("./checkpointing/" + str(processor_id) + "_" + str(node_index) + ".npz", W=self._W, iter=np.array([self._iter]))
    
    def loadW(self, processor_id, node_index):
        with np.load("./checkpointing/" + str(processor_id) + "_" + str(node_index) + ".npz") as data:
            self._W = data['W']
            self._iter = data['iter'][0]
    
    def outputTime(self, data_size, iter=100, id=0):
        file = open(str(id) + str(rank) + '_' + str(self.getGridId(self._node_comm.rank)) + '_uncoded_time.csv', 'w')
        file.write("_forward_total_time, _forward_bcast_time,_forward_tosink_time,_forward_dot_time,_forward_sum_time,_backprop_total_time,_backprop_bcast_time,_backprop_tosink_time,_backprop_dot_time,_backprop_sum_time,_update_time\n")
        #print(len(self._forward_total_time))
        for i in range(data_size * iter):
            if self._forward_total_time:
                file.write(str(self._forward_total_time[i].total_seconds()))
            file.write(',')
            if self._forward_bcast_time:
                file.write(str(self._forward_bcast_time[i].total_seconds()))
            file.write(',')
            if self._forward_tosink_time:
                file.write(str(self._forward_tosink_time[i].total_seconds()))
            file.write(',')
            if self._forward_dot_time:
                file.write(str(self._forward_dot_time[i].total_seconds()))
            file.write(',')
            if self._forward_sum_time:
                file.write(str(self._forward_sum_time[i].total_seconds()))
            file.write(',')
            if self._backprop_total_time:
                file.write(str(self._backprop_total_time[i].total_seconds()))
            file.write(',')
            if self._backprop_bcast_time:
                file.write(str(self._backprop_bcast_time[i].total_seconds()))
            file.write(',')
            if self._backprop_tosink_time:
                file.write(str(self._backprop_tosink_time[i].total_seconds()))
            file.write(',')
            if self._backprop_dot_time:
                file.write(str(self._backprop_dot_time[i].total_seconds()))
            file.write(',')
            if self._backprop_sum_time:
                file.write(str(self._backprop_sum_time[i].total_seconds()))
            file.write(',')
            if self._update_time:
                file.write(str(self._update_time[i].total_seconds()))
            file.write('\n')
    
    def encodeTest(self):
        for j in range(16):
            if rank == j:
                print(self.getGridId(self._node_comm.rank), self._W)
            time.sleep(0.2)


class DecentralReplicaFc(DecentralReplica):
    def forwardCompute(self, x, generate_error, test):
        r, c, sys = self.getGridId(self._node_comm.rank)
        if not test:
            random_number = np.random.random_sample()
            if random_number < self._error_rate and generate_error:
                error = scipy.sparse.rand(self._M_seg[0], self._M_seg[1], density=0.005)
                error = 2.0 * error - error.ceil()
                self._W += error.toarray() * 5.0
                print(np.amax(error.toarray()), np.amin(error.toarray()))
                print(self.getGridId(self._node_comm.rank))
                print("forward error")
        unsum_y_seg = np.dot(self._W, x[c * self._pre_N_seg[0] : (c+1) * self._pre_N_seg[0]])
        return unsum_y_seg
    
    def backpropCompute(self, dy, generate_error):
        r, c, sys = self.getGridId(self._node_comm.rank)
        random_number = np.random.random_sample()
        if random_number < self._error_rate and generate_error:
            error = scipy.sparse.rand(self._M_seg[0], self._M_seg[1], density=0.005)
            error = 2.0 * error - error.ceil()
            self._W += error.toarray() * 5.0
            print(np.amax(error.toarray()), np.amin(error.toarray()))
            print(self.getGridId(self._node_comm.rank))
            print("backprop error")
        unsum_dx_seg = np.dot(dy[r * self._N_seg[0] : (r+1) * self._N_seg[0]], self._W)
        return unsum_dx_seg
    
    def update(self, x, dy, generate_error):
        r, c, sys = self.getGridId(self._node_comm.rank)
        if r == -1 and c == -1 and sys == -1:
            return
        x_seg = x[c * self._pre_N_seg[0] : (c+1) * self._pre_N_seg[0]]
        dy_seg = dy[r * self._N_seg[0] : (r+1) * self._N_seg[0]]
        self._iter += 1
        rate = 1.0
        """
        if self._iter > 1500:
            rate = 0.1
        """
        self._W += rate * self._alpha * np.outer(dy_seg, x_seg)
        """
        self._m = self._beta1 * self._m + self._alpha * np.outer(dy_seg, x_seg)
        self._W += self._m
        """
        random_number = np.random.random_sample()
        if random_number < self._error_rate and generate_error:
            error = scipy.sparse.rand(self._M_seg[0], self._M_seg[1], density=0.005)
            error = 2.0 * error - error.ceil()
            self._W += error.toarray() * 5.0
            print(np.amax(error.toarray()), np.amin(error.toarray()))
            print(self.getGridId(self._node_comm.rank))
            print("update error")


class DecentralUncoded(Node):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, Pr, Pc, P, alpha, beta1, beta2, M, error_rate, node_comm, processors_id, previous, next, N, pre_N):
        super(DecentralUncoded, self).__init__(processors_id, previous, next, N, pre_N)
        self._M = M
        self._total_N = np.multiply.reduce(N)
        self._total_pre_N = np.multiply.reduce(pre_N)
        self._total_M = np.multiply.reduce(M)
        self._N_seg = (N[0]//Pr,) + N[1:]
        self._pre_N_seg = (pre_N[0]//Pc,) + pre_N[1:]
        self._M_seg = (M[0]//Pr, M[1]//Pc) + M[2:]
        self._total_N_seg = self._total_N // Pr
        self._total_pre_N_seg = self._total_pre_N // Pc
        self._total_M_seg = self._total_M // (Pr*Pc)
        self._W = None
        self._x = None
        self._Pr = Pr
        self._Pc = Pc
        self._P = P
        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        self._iter = None
        self._m = None
        self._v = None
        self._error_rate = error_rate
        self._node_comm = node_comm
        self._forward_sum_comm = MPI.COMM_NULL
        for i in range(Pr):
            include = [self.getProcessorId((i, j)) for j in range(Pc)]
            new_group = self._node_comm.Get_group().Incl(include)
            new_comm = self._node_comm.Create(new_group)
            if new_comm != MPI.COMM_NULL:
                self._forward_sum_comm = new_comm
        self._backprop_sum_comm = MPI.COMM_NULL
        for i in range(Pc):
            include = [self.getProcessorId((j, i)) for j in range(Pr)]
            new_group = self._node_comm.Get_group().Incl(include)
            new_comm = self._node_comm.Create(new_group)
            if new_comm != MPI.COMM_NULL:
                self._backprop_sum_comm = new_comm
        self._forward_total_time = []
        self._forward_tosink_time = []
        self._forward_dot_time = []
        self._forward_sum_time = []
        self._backprop_total_time = []
        self._backprop_tosink_time = []
        self._backprop_dot_time = []
        self._backprop_sum_time = []
        self._update_time = []

    def getGridId(self, node_rank):
        if node_rank >= self._Pr * self._Pc:
            return (-1, -1) #current processor is not used in this layer
        return (node_rank // self._Pc, node_rank % self._Pc)
    
    def getProcessorId(self, grid):
        r, c = grid
        if r == -1 and c == -1:
            return self._Pr * self._Pc #current processor is not used in this layer
        return r * self._Pc + c
    
    def init(self, layer_index):
        r, c = self.getGridId(self._node_comm.rank)
        """
        if not read:
            self._W = np.random.random_sample(self._M_seg) * 0.01
        else:
        """
        if r > -1 and c > -1:
            self._W = np.load("./init/W_" + str(r) + "_" + str(c//2) + "_" + str(layer_index) + ".npy")
            self._W = self._W[:, (c%2)*self._M_seg[1]:(c%2+1)*self._M_seg[1]]
            """
            self._W = np.random.random_sample(self._M_seg) * 0.01
            """
        self._iter = 0
        self._m = np.zeros(self._total_M_seg, dtype='d').reshape(self._M_seg)
        self._v = np.zeros(self._total_M_seg, dtype='d').reshape(self._M_seg)
        self._forward_total_time = []
        self._forward_tosink_time = []
        self._forward_dot_time = []
        self._forward_sum_time = []
        self._backprop_total_time = []
        self._backprop_tosink_time = []
        self._backprop_dot_time = []
        self._backprop_sum_time = []
        self._update_time = []
    
    @abc.abstractmethod
    def forwardCompute(self, x, generate_error):
        pass
    
    @abc.abstractmethod
    def backpropCompute(self, dy, generate_error):
        pass
    
    @abc.abstractmethod
    def update(self, x, dy, generate_error):
        pass
    
    def forward(self, x, tag, generate_error, test):
        forward_start = datetime.now()
        self._x = x
        r, c = self.getGridId(self._node_comm.rank)
        if -1 < c and c < self._Pc:
            #introduce error to W, compute unsum_y_seg
            forward_cal_start = datetime.now()
            unsum_y_seg = self.forwardCompute(x, generate_error, test)
            self._forward_dot_time.append(datetime.now() - forward_cal_start)
            
            #reduction(summation)
            forward_sum_start = datetime.now()
            y_seg = np.empty(self._total_N_seg, dtype='d').reshape(self._N_seg)
            self._forward_sum_comm.Reduce(unsum_y_seg, y_seg, op=MPI.SUM, root=0)
            self._forward_sum_time.append(datetime.now() - forward_sum_start)
            
        #get coded_y and send it to all processors
        counts = ((self._total_N_seg,) + (0,) * (self._Pc - 1)) * self._Pr
        counts += (0,) * (self._P - (self._Pr * self._Pc))
        dspls = ()
        for i in range(self._Pr):
            dspls += (i * self._total_N_seg,) + ((i + 1) * self._total_N_seg,) * (self._Pc - 1)
        dspls += (self._Pr * self._total_N_seg,) * (self._P - (self._Pr * self._Pc))
        if self._forward_sum_comm != MPI.COMM_NULL:
            if self._forward_sum_comm.rank == 0:
                senddata = y_seg
            else:
                senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        else:
            senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        y_segs = np.empty(self._total_N_seg * self._Pr, dtype='d').reshape((self._Pr,) + self._N_seg)
        self._node_comm.Allgatherv([senddata, counts[self._node_comm.rank]], \
                                   [y_segs, counts, dspls, MPI.DOUBLE])
        
        #get y
        y = y_segs.reshape(self._N)
        self._forward_total_time.append(datetime.now() - forward_start)
        return y, False
    
    def backprop(self, dy, tag, generate_error, update):
        backprop_start = datetime.now()
        self._dy = dy
        r, c = self.getGridId(self._node_comm.rank)
        if -1 < r and r < self._Pr:
            #introduce error to W, compute unsum_dx_seg
            backprop_cal_start = datetime.now()
            unsum_dx_seg = self.backpropCompute(dy, generate_error)
            self._backprop_dot_time.append(datetime.now() - backprop_cal_start)
            
            #reduction(summation)
            backprop_sum_start = datetime.now()
            dx_seg = np.empty(self._total_pre_N_seg, dtype='d').reshape(self._pre_N_seg)
            self._backprop_sum_comm.Reduce(unsum_dx_seg, dx_seg, op=MPI.SUM, root=0)
            self._backprop_sum_time.append(datetime.now() - backprop_sum_start)
            
        #get coded_dx and send it to all processors
        counts = (self._total_pre_N_seg,) * self._Pc
        counts += (0,) * (self._P - self._Pc)
        dspls = ()
        for i in range(self._Pc):
            dspls += (i * self._total_pre_N_seg,)
        dspls += (self._Pc * self._total_pre_N_seg,) * (self._P - self._Pc)
        if self._backprop_sum_comm != MPI.COMM_NULL:
            if self._backprop_sum_comm.rank == 0:
                senddata = dx_seg
            else:
                senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        else:
            senddata = np.zeros(counts[self._node_comm.rank], dtype='d')
        dx_segs = np.empty(self._total_pre_N_seg * self._Pc, dtype='d').reshape((self._Pc,) + self._pre_N_seg)
        self._node_comm.Allgatherv([senddata, counts[self._node_comm.rank]], \
                                   [dx_segs, counts, dspls, MPI.DOUBLE])
        
        #get dx
        dx = dx_segs.reshape(self._pre_N)
        fail = False

        #update with error
        if update:
            update_start = datetime.now()
            self.update(self._x, self._dy, generate_error)
            self._update_time.append(datetime.now() - update_start)
        self._backprop_total_time.append(datetime.now() - backprop_start)
        return dx, fail

    def saveW(self, processor_id, node_index):
        pass
    
    def loadW(self, processor_id, node_index):
        pass
    
    def outputTime(self, data_size, iter=100, id=0):
        file = open(str(id) + str(rank) + '_' + str(self.getGridId(self._node_comm.rank)) + '_uncoded_time.csv', 'w')
        file.write("_forward_total_time, _forward_bcast_time,_forward_tosink_time,_forward_dot_time,_forward_sum_time,_backprop_total_time,_backprop_bcast_time,_backprop_tosink_time,_backprop_dot_time,_backprop_sum_time,_update_time\n")
        #print(len(self._forward_total_time))
        for i in range(data_size * iter):
            if self._forward_total_time:
                file.write(str(self._forward_total_time[i].total_seconds()))
            file.write(',')
            if self._forward_bcast_time:
                file.write(str(self._forward_bcast_time[i].total_seconds()))
            file.write(',')
            if self._forward_tosink_time:
                file.write(str(self._forward_tosink_time[i].total_seconds()))
            file.write(',')
            if self._forward_dot_time:
                file.write(str(self._forward_dot_time[i].total_seconds()))
            file.write(',')
            if self._forward_sum_time:
                file.write(str(self._forward_sum_time[i].total_seconds()))
            file.write(',')
            if self._backprop_total_time:
                file.write(str(self._backprop_total_time[i].total_seconds()))
            file.write(',')
            if self._backprop_bcast_time:
                file.write(str(self._backprop_bcast_time[i].total_seconds()))
            file.write(',')
            if self._backprop_tosink_time:
                file.write(str(self._backprop_tosink_time[i].total_seconds()))
            file.write(',')
            if self._backprop_dot_time:
                file.write(str(self._backprop_dot_time[i].total_seconds()))
            file.write(',')
            if self._backprop_sum_time:
                file.write(str(self._backprop_sum_time[i].total_seconds()))
            file.write(',')
            if self._update_time:
                file.write(str(self._update_time[i].total_seconds()))
            file.write('\n')
    
    def encodeTest(self):
        for j in range(16):
            if rank == j:
                print(self.getGridId(self._node_comm.rank), self._W)
            time.sleep(0.2)


class DecentralUncodedFc(DecentralUncoded):
    def forwardCompute(self, x, generate_error, test):
        r, c = self.getGridId(self._node_comm.rank)
        if not test:
            random_number = np.random.random_sample()
            if random_number < self._error_rate and generate_error:
                error = scipy.sparse.rand(self._M_seg[0], self._M_seg[1], density=0.005)
                error = 2.0 * error - error.ceil()
                self._W += error.toarray() * 5.0
                print(np.amax(error.toarray()), np.amin(error.toarray()))
                print("forward error")
        unsum_y_seg = np.dot(self._W, x[c * self._pre_N_seg[0] : (c+1) * self._pre_N_seg[0]])
        return unsum_y_seg
    
    def backpropCompute(self, dy, generate_error):
        r, c = self.getGridId(self._node_comm.rank)
        random_number = np.random.random_sample()
        if random_number < self._error_rate and generate_error:
            error = scipy.sparse.rand(self._M_seg[0], self._M_seg[1], density=0.005)
            error = 2.0 * error - error.ceil()
            self._W += error.toarray() * 5.0
            print(np.amax(error.toarray()), np.amin(error.toarray()))
            print("backprop error")
        unsum_dx_seg = np.dot(dy[r * self._N_seg[0] : (r+1) * self._N_seg[0]], self._W)
        return unsum_dx_seg
    
    def update(self, x, dy, generate_error):
        r, c = self.getGridId(self._node_comm.rank)
        if r == -1 and c == -1:
            return
        x_seg = x[c * self._pre_N_seg[0] : (c+1) * self._pre_N_seg[0]]
        dy_seg = dy[r * self._N_seg[0] : (r+1) * self._N_seg[0]]
        self._iter += 1
        rate = 1.0
        """
        if 500 < self._iter <= 1000:
            rate = 1.0
        elif 1000 < self._iter <= 1500:
            rate = 1.0
        elif 1500 < self._iter <= 2000:
            rate = 0.1
        elif 2000 < self._iter:
            rate = 0.1
        """
        self._W += rate * self._alpha * np.outer(dy_seg, x_seg)
        """
        self._m = self._beta1 * self._m + self._alpha * np.outer(dy_seg, x_seg)
        self._W += self._m
        """
        random_number = np.random.random_sample()
        if random_number < self._error_rate and generate_error:
            error = scipy.sparse.rand(self._M_seg[0], self._M_seg[1], density=0.005)
            error = 2.0 * error - error.ceil()
            self._W += error.toarray() * 5.0
            print(np.amax(error.toarray()), np.amin(error.toarray()))
            print("update error")


if __name__ == "__main__":
    graph = Graph(40)
    
    #decentral Mds DNN
    if sys.argv[1] == "mds" and sys.argv[2] == "fc":
        replication = False
        test_node1 = graph.addNode(None, None, (10000,), (784,), {'type':'DecentralMdsFc', 'Pr':7, 'qr':5, 'Pc':6, 'qc':4, 'P':38, 'alpha':0.005, 'beta1':0., 'beta2':0.999, 'M':(10000, 784), 'error_rate':0.0003})
        bias_node1 = graph.addNode(None, None, 10000, 10000, {'type': 'DecentralBias', 'P':38, 'alpha':0.005, 'beta1':0., 'beta2':0.999})
        #sig_node1 = graph.addNode(None, None, 1200, 1200, {'type':'Sigmoid'})
        relu_node1 = graph.addNode(None, None, 10000, 10000, {'type':'DecentralRelu', 'P':38})
        
        test_node2 = graph.addNode(None, None, (10000,), (10000,), {'type':'DecentralMdsFc', 'Pr':7, 'qr':5, 'Pc':6, 'qc':4, 'P':38, 'alpha':0.005, 'beta1':0., 'beta2':0.999, 'M':(10000, 10000), 'error_rate':0.0003})
        bias_node2 = graph.addNode(None, None, 10000, 10000, {'type': 'DecentralBias', 'P':38, 'alpha':0.005, 'beta1':0., 'beta2':0.999})
        #sig_node2 = graph.addNode(None, None, 1200, 1200, {'type':'Sigmoid'})
        relu_node2 = graph.addNode(None, None, 10000, 10000, {'type':'DecentralRelu', 'P':38})
        
        test_node3 = graph.addNode(None, None, (10,), (10000,), {'type':'DecentralMdsFc', 'Pr':7, 'qr':5, 'Pc':6, 'qc':4, 'P':38, 'alpha':0.005, 'beta1':0., 'beta2':0.999, 'M':(10, 10000), 'error_rate':0.0003})
        bias_node3 = graph.addNode(None, None, 10, 10, {'type': 'DecentralBias', 'P':38, 'alpha':0.005, 'beta1':0., 'beta2':0.999})

    
    #decentral Replica DNN
    if sys.argv[1] == "replica" and sys.argv[2] == "fc":
        replication = True
        test_node1 = graph.addNode(None, None, (10000,), (784,), {'type':'DecentralReplicaFc', 'Pr':5, 'Pc':4, 'P':40, 'alpha':0.005, 'beta1':0., 'beta2':0.999, 'M':(10000, 784), 'error_rate':0.0003})
        bias_node1 = graph.addNode(None, None, 10000, 10000, {'type': 'DecentralBias', 'P':40, 'alpha':0.005, 'beta1':0., 'beta2':0.999})
        #sig_node1 = graph.addNode(None, None, 1200, 1200, {'type':'Sigmoid'})
        relu_node1 = graph.addNode(None, None, 10000, 10000, {'type':'DecentralRelu', 'P':40})
        
        test_node2 = graph.addNode(None, None, (10000,), (10000,), {'type':'DecentralReplicaFc', 'Pr':5, 'Pc':4, 'P':40, 'alpha':0.005, 'beta1':0., 'beta2':0.999, 'M':(10000, 10000), 'error_rate':0.0003})
        bias_node2 = graph.addNode(None, None, 10000, 10000, {'type': 'DecentralBias', 'P':40, 'alpha':0.005, 'beta1':0., 'beta2':0.999})
        #sig_node2 = graph.addNode(None, None, 1200, 1200, {'type':'Sigmoid'})
        relu_node2 = graph.addNode(None, None, 10000, 10000, {'type':'DecentralRelu', 'P':40})
        
        test_node3 = graph.addNode(None, None, (10,), (10000,), {'type':'DecentralReplicaFc', 'Pr':5, 'Pc':4, 'P':40, 'alpha':0.005, 'beta1':0., 'beta2':0.999, 'M':(10, 10000), 'error_rate':0.0003})
        bias_node3 = graph.addNode(None, None, 10, 10, {'type': 'DecentralBias', 'P':40, 'alpha':0.005, 'beta1':0., 'beta2':0.999})
    
    
    #decentral Uncoded DNN
    if sys.argv[1] == "uncoded" and sys.argv[2] == "fc":
        replication = False
        test_node1 = graph.addNode(None, None, (10000,), (784,), {'type':'DecentralUncodedFc', 'Pr':5, 'Pc':8, 'P':40, 'alpha':0.005, 'beta1':0., 'beta2':0.999, 'M':(10000, 784), 'error_rate':0.0003})
        bias_node1 = graph.addNode(None, None, 10000, 10000, {'type': 'DecentralBias', 'P':40, 'alpha':0.005, 'beta1':0., 'beta2':0.999})
        #sig_node1 = graph.addNode(None, None, 1200, 1200, {'type':'Sigmoid'})
        relu_node1 = graph.addNode(None, None, 10000, 10000, {'type':'DecentralRelu', 'P':40})
        
        test_node2 = graph.addNode(None, None, (10000,), (10000,), {'type':'DecentralUncodedFc', 'Pr':5, 'Pc':8, 'P':40, 'alpha':0.005, 'beta1':0., 'beta2':0.999, 'M':(10000, 10000), 'error_rate':0.0003})
        bias_node2 = graph.addNode(None, None, 10000, 10000, {'type': 'DecentralBias', 'P':40, 'alpha':0.005, 'beta1':0., 'beta2':0.999})
        #sig_node2 = graph.addNode(None, None, 1200, 1200, {'type':'Sigmoid'})
        relu_node2 = graph.addNode(None, None, 10000, 10000, {'type':'DecentralRelu', 'P':40})
        
        test_node3 = graph.addNode(None, None, (10,), (10000,), {'type':'DecentralUncodedFc', 'Pr':5, 'Pc':8, 'P':40, 'alpha':0.005, 'beta1':0., 'beta2':0.999, 'M':(10, 10000), 'error_rate':0.0003})
        bias_node3 = graph.addNode(None, None, 10, 10, {'type': 'DecentralBias', 'P':40, 'alpha':0.005, 'beta1':0., 'beta2':0.999})
    
    #get dataset (MNIST)
    train_size = int(sys.argv[5])
    X = mnistLoad.readImages(MNIST_PATH + 'train-images.idx3-ubyte', train_size) / 128.0
    Y = mnistLoad.readLabels(MNIST_PATH + 'train-labels.idx1-ubyte', train_size)
    
    #test time or accuracy
    if sys.argv[4] == "-time":
        #initialization
        checkpoint_iter = int(sys.argv[3])
        graph.init()
        
        #training
        start = datetime.now()
        loss_list, time_list = graph.fitSeq(X, Y, train_size, iter=1, checkpoint_iter=checkpoint_iter, replication=replication)
        if rank == 0:
            delta = datetime.now() - start
            print("training time (seconds): " + str(delta.total_seconds()))
        """    
        open('single_time.txt', 'w').write(str(delta) + str('\n'))
        open('total_time.txt', 'a').write(str(delta) + str('\n'))
        """
        #output measured time to files
        if rank == 0:
            loss_file = open(OUTPUT_PATH + "loss.txt", 'w')
            time_file = open(OUTPUT_PATH + "time.txt", 'w')
            for i in range(len(loss_list)):
                loss_file.write(str(loss_list[i]) + "\n")
                time_file.write(str(time_list[i].total_seconds()) + "\n")
            loss_file.close()
            time_file.close()
        
        #classification testing
        test_size = int(sys.argv[6])
        X = mnistLoad.readImages(MNIST_PATH + 't10k-images.idx3-ubyte', test_size) / 128.0
        Y = mnistLoad.readLabels(MNIST_PATH + 't10k-labels.idx1-ubyte', test_size)
        
        predict_Y = graph.predict(X, test_size, checkpoint_iter=checkpoint_iter)
        if rank == 0:
            correct = 0.0
            for i in range(test_size):
                if predict_Y[i] == Y[i]:
                    correct += 1
            print("accuracy:", correct/test_size)
    
    elif sys.argv[4] == "-accuracy":
        #initialization
        checkpoint_iter = int(sys.argv[3])
        graph.init()
        round = int(sys.argv[7])
        X = X.reshape((round, train_size/round, X.shape[-1]))
        Y = Y.reshape((round, train_size/round))
        
        #load test set
        test_size = int(sys.argv[6])
        test_X = mnistLoad.readImages(MNIST_PATH + 't10k-images.idx3-ubyte', test_size) / 128.0
        test_Y = mnistLoad.readLabels(MNIST_PATH + 't10k-labels.idx1-ubyte', test_size)
        
        #training and testing by batch
        accuracy = []
        loss_list = []
        time_list = []
        
        predict_Y = graph.predict(test_X, test_size, checkpoint_iter=checkpoint_iter)
        if rank == 0:
            correct = 0.0
            for i in range(test_size):
                if predict_Y[i] == test_Y[i]:
                    correct += 1
            print("accuracy:", correct/test_size)
            accuracy.append(correct/test_size)
        
        start = datetime.now()
        for i in range(round):
            if i == 0:
                init_check = True
                tmp_replication = replication
            else:
                init_check = False
                tmp_replication = False
            loss_list_round, time_list_round = graph.fitSeq(X[i], Y[i], train_size/round, iter=1, checkpoint_iter=checkpoint_iter, replication=tmp_replication, init_check=init_check)
            if loss_list_round is not None:
                loss_list += loss_list_round
                time_list += time_list_round
            predict_Y = graph.predict(test_X, test_size, checkpoint_iter=checkpoint_iter)
            if rank == 0:
                correct = 0.0
                for i in range(test_size):
                    if predict_Y[i] == test_Y[i]:
                        correct += 1
                print("accuracy:", correct/test_size)
                accuracy.append(correct/test_size)
        
        if rank == 0:
            delta = datetime.now() - start
            print("training time (seconds): " + str(delta.total_seconds()))
        """
        open('single_time.txt', 'w').write(str(delta) + str('\n'))
        open('total_time.txt', 'a').write(str(delta) + str('\n'))
        """
        #output measured time to files
        if rank == 0:
            loss_file = open(OUTPUT_PATH + "loss.txt", 'w')
            time_file = open(OUTPUT_PATH + "time.txt", 'w')
            for i in range(len(loss_list)):
                loss_file.write(str(loss_list[i]) + "\n")
                time_file.write(str(time_list[i].total_seconds()) + "\n")
            loss_file.close()
            time_file.close()
            accuracy_file = open(OUTPUT_PATH + "accuracy.txt", 'w')
            for i in range(len(accuracy)):
                accuracy_file.write(str(accuracy[i]) + "\n")
            accuracy_file.close()
