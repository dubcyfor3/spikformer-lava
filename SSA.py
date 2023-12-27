import sys
import typing as ty
sys.path.append("~/lava/src")

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import Monitor
import torch
import torch.nn as nn

class SpikeInput(AbstractProcess):
    """randomly generate spikes for input neurons"""

    def __init__(self,
                 shape,
                 ):
        super().__init__()
        self.shape = shape
        self.spikes_out = OutPort(shape=shape)  # Input spikes to the networks

@implements(proc=SpikeInput, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeInputModel(PyLoihiProcessModel):
    spikes_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)

    def run_spk(self):
        # generate a random bool matrix
        self.spikes = np.random.randint(2, size=self.spikes_out.shape).astype(np.bool_)
        self.spikes_out.send(self.spikes)

class MatMul(AbstractProcess):
    '''
    input: (B, N, M, K) 
            (B, N, K, P)
    output: (B, N, M, P)

    '''
    def __init__(self, shape):
        super().__init__()
        (B, N, M, K, P) = shape
        self.mat_in_A = InPort(shape=(B, N, M, K))
        self.mat_in_B = InPort(shape=(B, N, K, P))
        self.mat_out = OutPort(shape=(B, N, M, P))

@implements(proc=MatMul, protocol=LoihiProtocol)
@requires(CPU)
class PyMatMulModel(PyLoihiProcessModel):
    mat_in_A: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    mat_in_B: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, precision=32)


    def run_spk(self):
        mat_in_A = self.mat_in_A.recv()
        mat_in_B = self.mat_in_B.recv()
        # numpy to torch tensor
        mat_in_A = torch.from_numpy(mat_in_A)
        mat_in_B = torch.from_numpy(mat_in_B)
        mat_result = torch.matmul(mat_in_A, mat_in_B).numpy()
        self.mat_out.send(mat_result)

class Linear(AbstractProcess):
    '''
    input: (TB, N, C)
    output: (TB, N, C)
    T = 1
    '''
    def __init__(self, shape):
        super().__init__()
        (TB, N, C) = shape
        self.shape = shape
        self.mat_in = InPort(shape=(TB, N, C))
        self.weight = Var(shape=(C, C), init=0)
        self.bias = Var(shape=(C,), init=0)
        self.mat_out = OutPort(shape=(TB, N, C))

@implements(proc=Linear, protocol=LoihiProtocol)
@requires(CPU)
class PyLinearModel(PyLoihiProcessModel):
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.linear = nn.Linear(proc_params._parameters.get("shape")[1], proc_params._parameters.get("shape")[2])
        # fill weight with proc_params._parameters.get("weight")
        self.linear.weight.data = torch.from_numpy(proc_params._parameters.get("weight"))
        self.linear.bias.data = torch.from_numpy(proc_params._parameters.get("bias"))


    def run_spk(self):
        mat_in = self.mat_in.recv()
        mat_in = torch.from_numpy(mat_in)
        mat_out = self.linear(mat_in).detach().numpy()
        self.mat_out.send(mat_out)

class BatchNorm1d(AbstractProcess):
    '''
    input (TB, N, C)
    output (TB, N, C)
    normalize over C
    T = 1
    '''

    def __init__(self, shape):
        super().__init__()
        (TB, N, C) = shape
        self.shape = shape
        self.mat_in = InPort(shape=(TB, N, C))
        self.mat_out = OutPort(shape=(TB, N, C))

@implements(proc=BatchNorm1d, protocol=LoihiProtocol)
@requires(CPU)
class PyBatchNorm1dModel(PyLoihiProcessModel):
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.bn = nn.BatchNorm1d(proc_params._parameters.get("shape")[1])
        # initialize the gamma and beta
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()


    def run_spk(self):
        mat_in = self.mat_in.recv()
        mat_in = torch.from_numpy(mat_in)
        mat_out = self.bn(mat_in).detach().numpy()
        self.mat_out.send(mat_out)

class MyLIF(AbstractProcess):
    '''
    input: (B, N, C)
    output: (B, N, C)
    B * N * C neurons
    '''

class Transpose(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        (M, N) = shape
        self.mat_in = InPort(shape=(M, N))
        self.mat_out = OutPort(shape=(N, M))

@implements(proc=Transpose, protocol=LoihiProtocol)
@requires(CPU)
class PyTransposeModel(PyLoihiProcessModel):
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, precision=32)


    def run_spk(self):
        mat_in = self.mat_in.recv()
        mat_result = mat_in.T
        self.mat_out.send(mat_result)

class SSA(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        (TB, N, C) = shape
        self.shape = shape
        self.mat_in_x = InPort(shape=(TB, N, C))
        self.mat_out = OutPort(shape=(TB, N, C))

@implements(proc=SSA, protocol=LoihiProtocol)
@requires(CPU)
class PySSAModel(AbstractSubProcessModel):
    def __init__(self, proc):

        (TB, N, C) = proc.shape
        self.linear_q = Linear(shape=proc.shape)
        self.linear_k = Linear(shape=proc.shape)
        self.linear_v = Linear(shape=proc.shape)
        self.bn_q = BatchNorm1d(shape=proc.shape)
        self.bn_k = BatchNorm1d(shape=proc.shape)
        self.bn_v = BatchNorm1d(shape=proc.shape)
        self.lif_q = LIF(shape=(TB, N, C))
        self.lif_k = LIF(shape=(TB, N, C))
        self.lif_v = LIF(shape=(TB, N, C))
        self.attn_0 = MatMul(shape=(TB, N, C))

        proc.mat_in_x.connect(self.linear_q.mat_in)
        proc.mat_in_x.connect(self.linear_k.mat_in)
        proc.mat_in_x.connect(self.linear_v.mat_in)
        self.linear_q.mat_out.connect(self.bn_q.mat_in)
        self.linear_k.mat_out.connect(self.bn_k.mat_in)
        self.linear_v.mat_out.connect(self.bn_v.mat_in)
        self.bn_q.mat_out.connect(self.lif_q.a_in)
        self.bn_k.mat_out.connect(self.lif_k.a_in)
        self.bn_v.mat_out.connect(self.lif_v.a_in)

class SpikingSelfAttention(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        (N, dim) = shape
        self.shape = shape
        # self.mat_in_x = InPort(shape=(N * dim,))
        # self.mat_in_q = InPort(shape=(N, dim))
        # self.mat_in_k = InPort(shape=(N, dim))
        # self.mat_in_v = InPort(shape=(N, dim))
        self.mat_out = OutPort(shape=(N, dim))

@implements(proc=SpikingSelfAttention, protocol=LoihiProtocol)
@requires(CPU)
class PySpikingSelfAttentionModel(AbstractSubProcessModel):
    def __init__(self, proc):
        self.input_q = SpikeInput((proc.shape[0], proc.shape[1]))
        self.input_k = SpikeInput((proc.shape[1], proc.shape[0]))
        self.input_v = SpikeInput((proc.shape[0], proc.shape[1]))
        self.matmul_q_k = MatMul(shape=(proc.shape[0], proc.shape[1], proc.shape[0]))
        self.matmul_attn_v = MatMul(shape=(proc.shape[0], proc.shape[0], proc.shape[1]))

        self.input_q.spikes_out.connect(self.matmul_q_k.mat_in_A)
        self.input_k.spikes_out.connect(self.matmul_q_k.mat_in_B)
        self.matmul_q_k.mat_out.connect(self.matmul_attn_v.mat_in_A)
        self.input_v.spikes_out.connect(self.matmul_attn_v.mat_in_B)

        self.matmul_attn_v.mat_out.connect(proc.mat_out)


class Generate_MatMul(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.mat_result = Var(shape=(self.shape[0], self.shape[2]), init=0)
        self.mat_out = OutPort(shape=(self.shape[0], self.shape[2]))

@implements(proc=Generate_MatMul, protocol=LoihiProtocol)
@requires(CPU)
class PyGenerate_MatMulModel(AbstractSubProcessModel):
    def __init__(self, proc):
        self.input_q = SpikeInput((proc.shape[0], proc.shape[1]))
        self.input_k = SpikeInput((proc.shape[1], proc.shape[2]))
        self.matmul_layer = MatMul(shape=proc.shape)

        self.input_q.spikes_out.connect(self.matmul_layer.mat_in_A)
        self.input_k.spikes_out.connect(self.matmul_layer.mat_in_B)
        self.matmul_layer.mat_out.connect(proc.mat_out)

        proc.vars.mat_result.alias(self.matmul_layer.vars.mat_result)



class Dense(AbstractProcess):
    """Dense connections between neurons.
    Realizes the following abstract behavior:
    a_out = W * s_in
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.shape = Var(shape=(2,), init=shape)
        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, init=kwargs.pop("weights", 0))

@implements(proc=Dense, protocol=LoihiProtocol)
@requires(CPU)
class PyDenseModel(PyLoihiProcessModel):
    shape: np.ndarray = LavaPyType(np.ndarray, int)
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    weights: np.ndarray = LavaPyType(np.ndarray, float)
    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.shape = proc_params._parameters.get("shape", (1, 1))
        self.weights = proc_params._parameters.get("weights", 0)
        self.linear = nn.Linear(self.shape[1], self.shape[0])
        self.linear.weight.data = torch.from_numpy(self.weights)
        self.linear.bias.data.zero_()

    def run_spk(self):
        s_in = self.s_in.recv()
        # a_out = self.weights[:, s_in].sum(axis=1)
        a_out = self.linear(torch.from_numpy(s_in.astype(np.float32))).detach().numpy()
        self.a_out.send(a_out)


class LIF(AbstractProcess):
    """Leaky-Integrate-and-Fire (LIF) neural Process.
    LIF dynamics abstracts to:
    u[t] = u[t-1] * (1-du) + a_in              # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias_mant  # neuron voltage
    s_out = v[t] > vth                         # spike if threshold is exceeded
    v[t] = 0                                   # reset at spike
    Parameters
    ----------
    du: Inverse of decay time-constant for current decay.
    dv: Inverse of decay time-constant for voltage decay.
    bias: Neuron bias.
    vth: Neuron threshold voltage, exceeding which, the neuron will spike.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias_mant = kwargs.pop("bias_mant", 0)
        vth = kwargs.pop("vth", 10)

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_mant = Var(shape=shape, init=bias_mant)
        self.vth = Var(shape=(1,), init=vth)

@implements(proc=LIF, protocol=LoihiProtocol)
@requires(CPU)
class PyLifModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
    du: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)
    vth: float = LavaPyType(float, float)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        self.u[:] = self.u * (1 - self.du)
        self.u[:] += a_in_data
        self.v[:] = self.v * (1 - self.dv) + self.u + self.bias_mant
        s_out = self.v >= self.vth
        self.v[s_out] = 0  # Reset voltage to 0
        self.s_out.send(s_out)

class OutputProcess(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.mat_in = InPort(shape=(shape[0], shape[1]))
        self.mat_result = Var(shape=(shape[0], shape[1]), init=0)

@implements(proc=OutputProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputProcessModel(PyLoihiProcessModel):
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    mat_result: np.ndarray = LavaPyType(np.ndarray, int)

    def run_spk(self):
        mat_in = self.mat_in.recv()
        self.mat_result[:] = mat_in

class DenseLayer(AbstractProcess):
    """Combines Dense and LIF Processes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias_mant = kwargs.pop("bias_mant", 0)
        bias_exp = kwargs.pop("bias_exp", 0)
        vth = kwargs.pop("vth", 10)
        weights = kwargs.pop("weights", 0)

        self.s_in = InPort(shape=(shape[1],))
        self.s_out = OutPort(shape=(shape[0],))

        self.weights = Var(shape=shape, init=weights)
        self.u = Var(shape=(shape[0],), init=0)
        self.v = Var(shape=(shape[0],), init=0)
        self.bias_mant = Var(shape=(shape[0],), init=bias_mant)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.vth = Var(shape=(1,), init=vth)


@implements(proc=DenseLayer, protocol=LoihiProtocol)
class SubDenseLayerModel(AbstractSubProcessModel):

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""

        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        shape = proc.proc_params.get("shape", (1, 1))
        weights = proc.proc_params.get("weights", (1, 1))
        bias_mant = proc.proc_params.get("bias_mant", (1, 1))
        vth = proc.proc_params.get("vth", (1, 1))

        shape = weights.shape
        self.dense = Dense(shape=shape, weights=weights)
        self.lif = LIF(shape=(shape[0], ), bias_mant=bias_mant, vth=vth)

        # Connect the parent InPort to the InPort of the Dense child-Process.
        proc.in_ports.s_in.connect(self.dense.in_ports.s_in)

        # Connect the OutPort of the Dense child-Process to the InPort of the
        # LIF child-Process.
        self.dense.out_ports.a_out.connect(self.lif.in_ports.a_in)

        # Connect the OutPort of the LIF child-Process to the OutPort of the
        # parent Process.
        self.lif.out_ports.s_out.connect(proc.out_ports.s_out)

        proc.vars.u.alias(self.lif.vars.u)
        proc.vars.v.alias(self.lif.vars.v)
        proc.vars.bias_mant.alias(self.lif.vars.bias_mant)
        proc.vars.du.alias(self.lif.vars.du)
        proc.vars.dv.alias(self.lif.vars.dv)
        proc.vars.vth.alias(self.lif.vars.vth)
        proc.vars.weights.alias(self.dense.vars.weights)

def test_denselayer():
    dim = (3, 3)
    # Create the weight matrix.
    weights0 = np.zeros(shape=dim)
    weights0[1,1]=1
    # convert to float32
    weights0 = weights0.astype(np.float32)
    weights1 = weights0
    # Instantiate two DenseLayers.
    layer0 = DenseLayer(shape=dim, weights=weights0, bias_mant=4, vth=10)
    layer1 = DenseLayer(shape=dim, weights=weights1, bias_mant=4, vth=10)
    # Connect the first DenseLayer to the second DenseLayer.
    layer0.s_out.connect(layer1.s_in)

    print('Layer 1 weights: \n', layer1.weights.get(),'\n')
    print('\n ----- \n')

    rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=True)

    for t in range(9):
        # Run the entire network of Processes.
        layer1.run(condition=RunSteps(num_steps=1), run_cfg=rcfg)
        print('t: ',t)
        print('Layer 0 v: ', layer0.v.get())
        print('Layer 1 u: ', layer1.u.get())
        print('Layer 1 v: ', layer1.v.get())
        #print('Layer 1 spikes: ', layer1.spikes.get())
        print('\n ----- \n')

    layer1.stop()

def test_matmul():
    dim = (4, 4, 4)
    input_q = SpikeInput((dim[0], dim[1]))
    input_k = SpikeInput((dim[1], dim[2]))
    matmul_layer = MatMul(shape=dim)
    input_q.spikes_out.connect(matmul_layer.mat_in_A)
    input_k.spikes_out.connect(matmul_layer.mat_in_B)

    rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=False)

    for t in range(9):
        # Run the entire network of Processes.
        matmul_layer.run(condition=RunSteps(num_steps=1), run_cfg=rcfg)
        print('t: ',t)
        print('matmul_layer result: ', matmul_layer.mat_result.get())
        print('\n ----- \n')

def test_subprocess():
    dim = (4, 4, 4)
    proc = Generate_MatMul(shape=dim)
    rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=True)
    for t in range(9):
        proc.run(condition=RunSteps(num_steps=1), run_cfg=rcfg)
        print('matmul_layer result: ', proc.mat_result.get())
        print('\n ----- \n')

def test_SSA():
    dim = (4, 3)
    proc = SpikingSelfAttention(shape=dim)
    output = OutputProcess(shape=dim)
    proc.mat_out.connect(output.mat_in)
    rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=True)
    for t in range(9):
        proc.run(condition=RunSteps(num_steps=1), run_cfg=rcfg)
        print('SSA result at time step',t,': \n', output.mat_result.get())
        print('\n ----- \n')
    proc.stop()

if __name__ == "__main__":
    # test_subprocess()
    test_SSA()
    # test_denselayer()
