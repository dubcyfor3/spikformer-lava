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

class Residual(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        self.tensor_in_x = InPort(shape=shape)
        self.tensor_in_x1 = InPort(shape=shape)
        self.tensor_out = OutPort(shape=shape)

@implements(proc=Residual, protocol=LoihiProtocol)
@requires(CPU)
class PyResidualModel(PyLoihiProcessModel):
    tensor_in_x: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    tensor_in_x1: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    tensor_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)
    def run_spk(self):
        tensor_in_x = self.tensor_in_x.recv()
        tensor_in_x1 = self.tensor_in_x1.recv()
        tensor_in_x = torch.from_numpy(tensor_in_x).float()
        tensor_in_x1 = torch.from_numpy(tensor_in_x1).float()
        tensor_out = tensor_in_x + tensor_in_x1
        tensor_out = tensor_out.detach().numpy()
        self.tensor_out.send(tensor_out)

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

class MatMul_bool2float(AbstractProcess):
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

@implements(proc=MatMul_bool2float, protocol=LoihiProtocol)
@requires(CPU)
class PyMatMulModel(PyLoihiProcessModel):
    mat_in_A: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    mat_in_B: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)


    def run_spk(self):
        mat_in_A = self.mat_in_A.recv()
        mat_in_B = self.mat_in_B.recv()
        # numpy to torch tensor
        mat_in_A = torch.from_numpy(mat_in_A)
        mat_in_A = mat_in_A.float()
        mat_in_B = torch.from_numpy(mat_in_B)
        mat_in_B = mat_in_B.float()
        mat_result = torch.matmul(mat_in_A, mat_in_B).numpy()
        self.mat_out.send(mat_result)

class MatMul_floatbool2float(AbstractProcess):
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

@implements(proc=MatMul_floatbool2float, protocol=LoihiProtocol)
@requires(CPU)
class PyMatMul_floatbool2floatModel(PyLoihiProcessModel):
    mat_in_A: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)
    mat_in_B: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)


    def run_spk(self):
        mat_in_A = self.mat_in_A.recv()
        mat_in_B = self.mat_in_B.recv()
        # numpy to torch tensor
        mat_in_A = torch.from_numpy(mat_in_A)
        mat_in_B = torch.from_numpy(mat_in_B)
        mat_in_A = mat_in_A.float()
        mat_in_B = mat_in_B.float()
        mat_result = torch.matmul(mat_in_A, mat_in_B).numpy()
        self.mat_out.send(mat_result)

class Linear(AbstractProcess):
    '''
    input: (TB, N, C_input)
    output: (TB, N, C_output)
    T = 1
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        (TB, N, C_input, C_output) = shape
        self.shape = shape
        self.mat_in = InPort(shape=(TB, N, C_input))
        # self.weight = Var(shape=(C, C), init=weight)
        # self.bias = Var(shape=(C,), init=bias)
        self.mat_out = OutPort(shape=(TB, N, C_output))

@implements(proc=Linear, protocol=LoihiProtocol)
@requires(CPU)
class PyLinearModel(PyLoihiProcessModel):
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)
    # weight: np.ndarray = LavaPyType(np.ndarray, float)
    # bias: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.linear = nn.Linear(proc_params._parameters.get("shape")[2], proc_params._parameters.get("shape")[3])
        weight = proc_params._parameters.get("weight")
        bias = proc_params._parameters.get("bias")
        assert weight.shape == self.linear.weight.data.shape
        assert bias.shape == self.linear.bias.data.shape
        self.linear.weight.data = torch.from_numpy(weight).float()
        self.linear.bias.data = torch.from_numpy(bias).float()


    def run_spk(self):
        mat_in = self.mat_in.recv()
        mat_in = torch.from_numpy(mat_in).float()
        mat_out = self.linear(mat_in).detach().numpy()
        self.mat_out.send(mat_out)

class BatchNorm1d(AbstractProcess):
    '''
    input (TB, N, C)
    output (TB, N, C)
    normalize over C
    T = 1
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        (TB, N, C) = shape
        self.shape = shape
        self.mat_in = InPort(shape=(TB, N, C))
        self.mat_out = OutPort(shape=(TB, N, C))

@implements(proc=BatchNorm1d, protocol=LoihiProtocol)
@requires(CPU)
class PyBatchNorm1dModel(PyLoihiProcessModel):
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.bn = nn.BatchNorm1d(proc_params._parameters.get("shape")[2])
        beta = proc_params._parameters.get("beta")
        gamma = proc_params._parameters.get("gamma")
        # initialize the gamma and beta
        self.bn.weight.data = torch.from_numpy(gamma).float()
        self.bn.bias.data = torch.from_numpy(beta).float()


    def run_spk(self):
        mat_in = self.mat_in.recv()
        mat_in = torch.from_numpy(mat_in)
        mat_in = mat_in.float()
        mat_in = mat_in.transpose(1, 2)
        mat_out = self.bn(mat_in)
        mat_out = mat_out.transpose(1, 2).detach().numpy()
        self.mat_out.send(mat_out)

class Transpose(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        self.dim = Var(shape=(1,), init=shape.__len__())
        shape_out = [i for i in shape]
        shape_out[-1], shape_out[-2] = shape_out[-2], shape_out[-1]
        shape_out = tuple(shape_out)
        self.mat_in = InPort(shape=shape)
        self.mat_out = OutPort(shape=shape_out)

@implements(proc=Transpose, protocol=LoihiProtocol)
@requires(CPU)
class PyTransposeModel(PyLoihiProcessModel):
    dim: np.ndarray = LavaPyType(np.ndarray, int)
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)


    def run_spk(self):
        mat_in = self.mat_in.recv()
        target_dim = [i for i in range(self.dim[0])]
        target_dim[-1], target_dim[-2] = target_dim[-2], target_dim[-1]
        target_dim = tuple(target_dim)
        mat_result = np.transpose(mat_in, target_dim)
        self.mat_out.send(mat_result)

class SplitMultiHead(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        (B, N, C, num_heads) = shape
        self.shape = Var(shape=(4,), init=shape)
        self.mat_in = InPort(shape=(B, N, C))
        self.mat_out = OutPort(shape=(B, num_heads, N, C//num_heads))

@implements(proc=SplitMultiHead, protocol=LoihiProtocol)
@requires(CPU)
class PySplitMultiHeadModel(PyLoihiProcessModel):
    shape: np.ndarray = LavaPyType(np.ndarray, int)
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)

    def run_spk(self):
        mat_in = self.mat_in.recv()
        mat_result = np.split(mat_in, self.shape[3], axis=2)
        mat_result = np.stack(mat_result, axis=2)
        mat_result = np.transpose(mat_result, (0, 2, 1, 3))
        self.mat_out.send(mat_result)

class ConcatMultiHead(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        (B, N, C, num_heads) = shape
        self.shape = Var(shape=(4,), init=shape)
        self.mat_in = InPort(shape=(B, num_heads, N, C//num_heads))
        self.mat_out = OutPort(shape=(B, N, C))

@implements(proc=ConcatMultiHead, protocol=LoihiProtocol)
@requires(CPU)
class PyConcatMultiHeadModel(PyLoihiProcessModel):
    shape: np.ndarray = LavaPyType(np.ndarray, int)
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)

    def run_spk(self):
        mat_in = self.mat_in.recv()
        mat_result = np.transpose(mat_in, (0, 2, 1, 3))
        mat_result = np.reshape(mat_result, (self.shape[0], self.shape[1], self.shape[2]))
        self.mat_out.send(mat_result)

class Scale(AbstractProcess):
    def __init__(self, shape, scale):
        super().__init__()
        self.shape = shape
        (A, B, C, D) = shape
        self.mat_in = InPort(shape=(A, B, C, D))
        self.mat_out = OutPort(shape=(A, B, C, D))
        self.scale = Var(shape=(1,), init=scale)

@implements(proc=Scale, protocol=LoihiProtocol)
@requires(CPU)
class PyScaleModel(PyLoihiProcessModel):
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)
    scale: float = LavaPyType(float, float)

    def run_spk(self):
        mat_in = self.mat_in.recv()
        mat_result = mat_in * self.scale
        self.mat_out.send(mat_result)

class SSA(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape")
        (TB, N, C) = shape
        num_heads = kwargs.get("num_heads")

        self.shape = (TB, N, C)
        self.num_heads = num_heads

        self.q_linear_weight = kwargs.get("weight_q_linear")
        self.q_linear_bias = kwargs.get("bias_q_linear")
        self.k_linear_weight = kwargs.get("weight_k_linear")
        self.k_linear_bias = kwargs.get("bias_k_linear")
        self.v_linear_weight = kwargs.get("weight_v_linear")
        self.v_linear_bias = kwargs.get("bias_v_linear")
        self.proj_linear_weight = kwargs.get("weight_proj_linear")
        self.proj_linear_bias = kwargs.get("bias_proj_linear")

        self.q_bn_beta = kwargs.get("beta_q_bn")
        self.q_bn_gamma = kwargs.get("gamma_q_bn")
        self.k_bn_beta = kwargs.get("beta_k_bn")
        self.k_bn_gamma = kwargs.get("gamma_k_bn")
        self.v_bn_beta = kwargs.get("beta_v_bn")
        self.v_bn_gamma = kwargs.get("gamma_v_bn")
        self.proj_bn_beta = kwargs.get("beta_proj_bn")
        self.proj_bn_gamma = kwargs.get("gamma_proj_bn")

        # self.q_linear_weight = Var(shape=(C, C), init=q_linear_weight)
        # self.q_linear_bias = Var(shape=(C,), init=q_linear_bias)
        # self.k_linear_weight = Var(shape=(C, C), init=k_linear_weight)
        # self.k_linear_bias = Var(shape=(C,), init=k_linear_bias)
        # self.v_linear_weight = Var(shape=(C, C), init=v_linear_weight)
        # self.v_linear_bias = Var(shape=(C,), init=v_linear_bias)
        # self.proj_linear_weight = Var(shape=(C, C), init=proj_linear_weight)
        # self.proj_linear_bias = Var(shape=(C,), init=proj_linear_bias)

        self.lif_q_u = Var(shape=(TB, N, C), init=0)
        self.lif_q_v = Var(shape=(TB, N, C), init=0)
        self.lif_q_bias_mant = Var(shape=(TB, N, C), init=0)
        self.lif_q_du = Var(shape=(1,), init=0)
        self.lif_q_dv = Var(shape=(1,), init=0)
        self.lif_q_vth = Var(shape=(1,), init=0)

        self.lif_k_u = Var(shape=(TB, N, C), init=0)
        self.lif_k_v = Var(shape=(TB, N, C), init=0)
        self.lif_k_bias_mant = Var(shape=(TB, N, C), init=0)
        self.lif_k_du = Var(shape=(1,), init=0)
        self.lif_k_dv = Var(shape=(1,), init=0)
        self.lif_k_vth = Var(shape=(1,), init=0)

        self.lif_v_u = Var(shape=(TB, N, C), init=0)
        self.lif_v_v = Var(shape=(TB, N, C), init=0)
        self.lif_v_bias_mant = Var(shape=(TB, N, C), init=0)
        self.lif_v_du = Var(shape=(1,), init=0)
        self.lif_v_dv = Var(shape=(1,), init=0)
        self.lif_v_vth = Var(shape=(1,), init=0)

        self.lif_attn_u = Var(shape=(TB, N, C), init=0)
        self.lif_attn_v = Var(shape=(TB, N, C), init=0)
        self.lif_attn_bias_mant = Var(shape=(TB, N, C), init=0)
        self.lif_attn_du = Var(shape=(1,), init=0)
        self.lif_attn_dv = Var(shape=(1,), init=0)
        self.lif_attn_vth = Var(shape=(1,), init=0)

        self.lif_proj_u = Var(shape=(TB, N, C), init=0)
        self.lif_proj_v = Var(shape=(TB, N, C), init=0)
        self.lif_proj_bias_mant = Var(shape=(TB, N, C), init=0)
        self.lif_proj_du = Var(shape=(1,), init=0)
        self.lif_proj_dv = Var(shape=(1,), init=0)
        self.lif_proj_vth = Var(shape=(1,), init=0)


        self.mat_in_x = InPort(shape=(TB, N, C))
        self.mat_out = OutPort(shape=(TB, N, C))

@implements(proc=SSA, protocol=LoihiProtocol)
@requires(CPU)
class PySSAModel(AbstractSubProcessModel):
    def __init__(self, proc):

        (TB, N, C) = proc.shape
        self.linear_q = Linear(shape=(TB, N, C, C), weight=proc.q_linear_weight, bias=proc.q_linear_bias)
        self.linear_k = Linear(shape=(TB, N, C, C), weight=proc.k_linear_weight, bias=proc.k_linear_bias)
        self.linear_v = Linear(shape=(TB, N, C, C), weight=proc.v_linear_weight, bias=proc.v_linear_bias)
        self.linear_proj = Linear(shape=(TB, N, C, C), weight=proc.proj_linear_weight, bias=proc.proj_linear_bias)

        self.bn_q = BatchNorm1d(shape=proc.shape, gamma = proc.q_bn_gamma, beta = proc.q_bn_beta)
        self.bn_k = BatchNorm1d(shape=proc.shape, gamma = proc.k_bn_gamma, beta = proc.k_bn_beta)
        self.bn_v = BatchNorm1d(shape=proc.shape, gamma = proc.v_bn_gamma, beta = proc.v_bn_beta)
        self.bn_proj = BatchNorm1d(shape=proc.shape, gamma = proc.proj_bn_gamma, beta = proc.proj_bn_beta)

        self.lif_q = LIF(shape=(TB, N, C))
        self.lif_k = LIF(shape=(TB, N, C))
        self.lif_v = LIF(shape=(TB, N, C))
        self.lif_attn = LIF(shape=(TB, N, C))
        self.lif_proj = LIF(shape=(TB, N, C))

        self.split_head_q = SplitMultiHead(shape=(TB, N, C, proc.num_heads))
        self.split_head_k = SplitMultiHead(shape=(TB, N, C, proc.num_heads))
        self.split_head_v = SplitMultiHead(shape=(TB, N, C, proc.num_heads))
        self.transpose_k = Transpose(shape=(TB, proc.num_heads, N, C//proc.num_heads))

        self.attn_0 = MatMul_bool2float(shape=(TB, proc.num_heads, N, C//proc.num_heads, N))
        self.scale = Scale(shape=(TB, proc.num_heads, N, N), scale=0.125)
        self.attn_1 = MatMul_floatbool2float(shape=(TB, proc.num_heads, N, N, C//proc.num_heads))

        self.concat_head = ConcatMultiHead(shape=(TB, N, C, proc.num_heads))
        self.residual = Residual(shape=(TB, N, C))

        proc.mat_in_x.connect(self.linear_q.mat_in)
        proc.mat_in_x.connect(self.linear_k.mat_in)
        proc.mat_in_x.connect(self.linear_v.mat_in)

        self.linear_q.mat_out.connect(self.bn_q.mat_in)
        self.linear_k.mat_out.connect(self.bn_k.mat_in)
        self.linear_v.mat_out.connect(self.bn_v.mat_in)

        self.bn_q.mat_out.connect(self.lif_q.a_in)
        self.bn_k.mat_out.connect(self.lif_k.a_in)
        self.bn_v.mat_out.connect(self.lif_v.a_in)

        self.lif_q.s_out.connect(self.split_head_q.mat_in)
        self.lif_k.s_out.connect(self.split_head_k.mat_in)
        self.lif_v.s_out.connect(self.split_head_v.mat_in)

        self.split_head_k.mat_out.connect(self.transpose_k.mat_in)
        self.split_head_q.mat_out.connect(self.attn_0.mat_in_A)
        self.transpose_k.mat_out.connect(self.attn_0.mat_in_B)

        self.attn_0.mat_out.connect(self.scale.mat_in)
        self.scale.mat_out.connect(self.attn_1.mat_in_A)
        self.split_head_v.mat_out.connect(self.attn_1.mat_in_B)

        self.attn_1.mat_out.connect(self.concat_head.mat_in)
        self.concat_head.mat_out.connect(self.lif_attn.a_in)

        self.lif_attn.s_out.connect(self.linear_proj.mat_in)
        self.linear_proj.mat_out.connect(self.bn_proj.mat_in)
        self.bn_proj.mat_out.connect(self.lif_proj.a_in)
        self.lif_proj.s_out.connect(self.residual.tensor_in_x)
        proc.mat_in_x.connect(self.residual.tensor_in_x1)
        self.residual.tensor_out.connect(proc.mat_out)

        # setting alias of sub-processes
        proc.vars.lif_q_u.alias(self.lif_q.vars.u)
        proc.vars.lif_q_v.alias(self.lif_q.vars.v)
        proc.vars.lif_q_bias_mant.alias(self.lif_q.vars.bias_mant)
        proc.vars.lif_q_du.alias(self.lif_q.vars.du)
        proc.vars.lif_q_dv.alias(self.lif_q.vars.dv)
        proc.vars.lif_q_vth.alias(self.lif_q.vars.vth)

        proc.vars.lif_k_u.alias(self.lif_k.vars.u)
        proc.vars.lif_k_v.alias(self.lif_k.vars.v)
        proc.vars.lif_k_bias_mant.alias(self.lif_k.vars.bias_mant)
        proc.vars.lif_k_du.alias(self.lif_k.vars.du)
        proc.vars.lif_k_dv.alias(self.lif_k.vars.dv)
        proc.vars.lif_k_vth.alias(self.lif_k.vars.vth)

        proc.vars.lif_v_u.alias(self.lif_v.vars.u)
        proc.vars.lif_v_v.alias(self.lif_v.vars.v)
        proc.vars.lif_v_bias_mant.alias(self.lif_v.vars.bias_mant)
        proc.vars.lif_v_du.alias(self.lif_v.vars.du)
        proc.vars.lif_v_dv.alias(self.lif_v.vars.dv)
        proc.vars.lif_v_vth.alias(self.lif_v.vars.vth)

        proc.vars.lif_attn_u.alias(self.lif_attn.vars.u)
        proc.vars.lif_attn_v.alias(self.lif_attn.vars.v)
        proc.vars.lif_attn_bias_mant.alias(self.lif_attn.vars.bias_mant)
        proc.vars.lif_attn_du.alias(self.lif_attn.vars.du)
        proc.vars.lif_attn_dv.alias(self.lif_attn.vars.dv)
        proc.vars.lif_attn_vth.alias(self.lif_attn.vars.vth)

        proc.vars.lif_proj_u.alias(self.lif_proj.vars.u)
        proc.vars.lif_proj_v.alias(self.lif_proj.vars.v)
        proc.vars.lif_proj_bias_mant.alias(self.lif_proj.vars.bias_mant)
        proc.vars.lif_proj_du.alias(self.lif_proj.vars.du)
        proc.vars.lif_proj_dv.alias(self.lif_proj.vars.dv)
        proc.vars.lif_proj_vth.alias(self.lif_proj.vars.vth)

        # proc.vars.q_linear_weight.alias(self.linear_q.vars.weight)
        # proc.vars.q_linear_bias.alias(self.linear_q.vars.bias)
        # proc.vars.k_linear_weight.alias(self.linear_k.vars.weight)
        # proc.vars.k_linear_bias.alias(self.linear_k.vars.bias)
        # proc.vars.v_linear_weight.alias(self.linear_v.vars.weight)
        # proc.vars.v_linear_bias.alias(self.linear_v.vars.bias)
        # proc.vars.proj_linear_weight.alias(self.linear_proj.vars.weight)
        # proc.vars.proj_linear_bias.alias(self.linear_proj.vars.bias)

class InputGenerator(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        (TB, N, C) = shape
        self.mat_out = OutPort(shape=(TB, N, C))

@implements(proc=InputGenerator, protocol=LoihiProtocol)
@requires(CPU)
class PyInputGeneratorModel(PyLoihiProcessModel):
    mat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)

    def run_spk(self):
        # randomly generate 0 or 1 for input
        mat_result = np.random.randint(2, size=self.mat_out.shape)
        self.mat_out.send(mat_result)

class OutputReceiver(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        (TB, N, C) = shape
        self.mat_in = InPort(shape=(TB, N, C))
        self.mat_result = Var(shape=(TB, N, C), init=0)

@implements(proc=OutputReceiver, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputReceiverModel(PyLoihiProcessModel):
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    mat_result: np.ndarray = LavaPyType(np.ndarray, bool)

    def run_spk(self):
        mat_in = self.mat_in.recv()
        self.mat_result[:] = mat_in


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
        # self.weights = Var(shape=shape, init=kwargs.pop("weights", 0))

@implements(proc=Dense, protocol=LoihiProtocol)
@requires(CPU)
class PyDenseModel(PyLoihiProcessModel):
    shape: np.ndarray = LavaPyType(np.ndarray, int)
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    # weights: np.ndarray = LavaPyType(np.ndarray, float)
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
        dv = kwargs.pop("dv", 0.5)
        bias_mant = kwargs.pop("bias_mant", 0)
        vth = kwargs.pop("vth", 1)

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=np.zeros(shape))
        self.v = Var(shape=shape, init=np.zeros(shape))
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
        self.u[:] = a_in_data
        self.v[:] = (self.v + self.u) * (1 - self.dv) + self.bias_mant
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
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
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

        # self.weights = Var(shape=shape, init=weights)
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
        # proc.vars.weights.alias(self.dense.vars.weights)

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

    # print('Layer 1 weights: \n', layer1.weights.get(),'\n')
    # print('\n ----- \n')

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

def test_whole_block():
    shape = (4, 5, 6) # (TB, N, C)
    num_heads = 3
    # generate random weights and bias
    weight = np.random.rand(shape[2], shape[2])
    bias = np.random.rand(shape[2])

    gamma = np.ones(shape[2])
    beta = np.zeros(shape[2])

    this_block = SSA(shape=shape,
                     weight_q_linear=weight,
                     bias_q_linear=bias,
                     weight_k_linear=weight,
                     bias_k_linear=bias,
                     weight_v_linear=weight,
                     bias_v_linear=bias,
                     weight_proj_linear=weight,
                     bias_proj_linear=bias,
                        beta_q_bn=beta,
                        gamma_q_bn=gamma,
                        beta_k_bn=beta,
                        gamma_k_bn=gamma,
                        beta_v_bn=beta,
                        gamma_v_bn=gamma,
                        beta_proj_bn=beta,
                        gamma_proj_bn=gamma,
                     num_heads=num_heads)

    input_process = InputGenerator(shape=shape)
    output_process = OutputReceiver(shape=shape)
    input_process.mat_out.connect(this_block.mat_in_x)
    this_block.mat_out.connect(output_process.mat_in)

    rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=True)

    for t in range(9):
        # Run the entire network of Processes.
        this_block.run(condition=RunSteps(num_steps=1), run_cfg=rcfg)
        print('t: ',t)
        print('this_block result: ', output_process.mat_result.get())
        print('\n ----- \n')

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
    test_whole_block()
    # test_denselayer()
