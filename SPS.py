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

from SSA import LIF, Transpose, Residual

class BatchNorm2d(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        (TB, C, H, W) = shape
        self.tensor_in_x = InPort(shape=(TB, C, H, W))
        self.tensor_out = OutPort(shape=(TB, C, H, W))

@implements(proc=BatchNorm2d, protocol=LoihiProtocol)
@requires(CPU)
class PyBatchNorm2dModel(PyLoihiProcessModel):
    tensor_in_x: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)
    tensor_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        shape = proc_params._parameters.get("shape")
        self.bn = nn.BatchNorm2d(num_features=shape[1])
        self.bn.eval()
        gamma = proc_params._parameters.get("gamma")
        beta = proc_params._parameters.get("beta")
        running_mean = proc_params._parameters.get("running_mean")
        running_var = proc_params._parameters.get("running_var")
        assert gamma.shape == self.bn.weight.data.shape
        assert beta.shape == self.bn.bias.data.shape
        self.bn.weight.data = torch.from_numpy(gamma).float()
        self.bn.bias.data = torch.from_numpy(beta).float()
        self.bn.running_mean = torch.from_numpy(running_mean).float()
        self.bn.running_var = torch.from_numpy(running_var).float()

    def run_spk(self):
        tensor_in_x = self.tensor_in_x.recv()
        tensor_in_x = torch.from_numpy(tensor_in_x).float()
        tensor_out = self.bn(tensor_in_x).detach().numpy()
        self.tensor_out.send(tensor_out)

class Conv2D_init(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        (TB, C_in, C_out, H, W) = shape
        self.tensor_in_x = InPort(shape=(TB, C_in, H, W))
        self.tensor_out = OutPort(shape=(TB, C_out, H, W))

@implements(proc=Conv2D_init, protocol=LoihiProtocol)
@requires(CPU)
class PyConv2D_initModel(PyLoihiProcessModel):
    tensor_in_x: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)
    tensor_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        (TB, C_in, C_out, H, W) = proc_params._parameters.get("shape")
        self.conv2d = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=3, stride=1, padding=1, bias=False)
        weight = proc_params._parameters.get("weight")
        assert weight.shape == self.conv2d.weight.data.shape
        self.conv2d.weight.data = torch.from_numpy(weight).float()

    def run_spk(self):
        tensor_in_x = self.tensor_in_x.recv()
        tensor_in_x = torch.from_numpy(tensor_in_x).float()
        tensor_out = self.conv2d(tensor_in_x).detach().numpy()
        self.tensor_out.send(tensor_out)

class Conv2D(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        (TB, C_in, C_out, H, W) = shape
        self.tensor_in_x = InPort(shape=(TB, C_in, H, W))
        self.tensor_out = OutPort(shape=(TB, C_out, H, W))

@implements(proc=Conv2D, protocol=LoihiProtocol)
@requires(CPU)
class PyConv2DModel(PyLoihiProcessModel):
    tensor_in_x: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    tensor_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        (TB, C_in, C_out, H, W) = proc_params._parameters.get("shape")
        self.conv2d = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=3, stride=1, padding=1, bias=False)
        weight = proc_params._parameters.get("weight")
        assert weight.shape == self.conv2d.weight.data.shape
        self.conv2d.weight.data = torch.from_numpy(weight).float()

    def run_spk(self):
        tensor_in_x = self.tensor_in_x.recv()
        tensor_in_x = torch.from_numpy(tensor_in_x).float()
        tensor_out = self.conv2d(tensor_in_x).detach().numpy()
        self.tensor_out.send(tensor_out)

class MaxPool2D(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        (TB, C, H_in, W_in, H_out, W_out) = shape
        self.tensor_in_x = InPort(shape=(TB, C, H_in, W_in))
        self.tensor_out = OutPort(shape=(TB, C, H_out, W_out))

@implements(proc=MaxPool2D, protocol=LoihiProtocol)
@requires(CPU)
class PyMaxPool2DModel(PyLoihiProcessModel):
    tensor_in_x: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    tensor_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)


    def run_spk(self):
        tensor_in_x = self.tensor_in_x.recv()
        tensor_in_x = torch.from_numpy(tensor_in_x).float()
        tensor_out= self.maxpool2d(tensor_in_x)
        tensor_out = tensor_out.bool()
        tensor_out = tensor_out.detach().numpy()
        self.tensor_out.send(tensor_out)

class Flatten(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        (TB, C, H, W) = shape
        self.tensor_in_x = InPort(shape=(TB, C, H, W))
        self.tensor_out = OutPort(shape=(TB, C, H*W))

@implements(proc=Flatten, protocol=LoihiProtocol)
@requires(CPU)
class PyFlattenModel(PyLoihiProcessModel):
    tensor_in_x: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)
    tensor_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)
    def run_spk(self):
        tensor_in_x = self.tensor_in_x.recv()
        tensor_in_x = torch.from_numpy(tensor_in_x).float()
        tensor_out = tensor_in_x.flatten(start_dim=-2)
        tensor_out = tensor_out.detach().numpy()
        self.tensor_out.send(tensor_out)


class SPS(AbstractProcess):

    def __init__(self, **kwargs):
        '''
        input: (batch_size, in_channels, image_size_h, image_size_w),
        output: (batch_size, image_size_h*image_size_w//16, embed_dims)
        '''
        super().__init__()
        self.shape = kwargs.get("shape")
        (TB,) = self.shape
        self.image_size_h = kwargs.get("image_size_h")
        self.image_size_w = kwargs.get("image_size_w")
        self.patch_size = kwargs.get("patch_size")
        self.in_channels = kwargs.get("in_channels")
        self.embed_dims = kwargs.get("embed_dims")

        self.proj_conv_weight = kwargs.get("proj_conv_weight")
        self.proj_bn_gamma = kwargs.get("proj_bn_gamma")
        self.proj_bn_beta = kwargs.get("proj_bn_beta")
        self.proj_bn_running_mean = kwargs.get("proj_bn_running_mean")
        self.proj_bn_running_var = kwargs.get("proj_bn_running_var")

        self.proj1_conv_weight = kwargs.get("proj1_conv_weight")
        self.proj1_bn_gamma = kwargs.get("proj1_bn_gamma")
        self.proj1_bn_beta = kwargs.get("proj1_bn_beta")
        self.proj1_bn_running_mean = kwargs.get("proj1_bn_running_mean")
        self.proj1_bn_running_var = kwargs.get("proj1_bn_running_var")

        self.proj2_conv_weight = kwargs.get("proj2_conv_weight")
        self.proj2_bn_gamma = kwargs.get("proj2_bn_gamma")
        self.proj2_bn_beta = kwargs.get("proj2_bn_beta")
        self.proj2_bn_running_mean = kwargs.get("proj2_bn_running_mean")
        self.proj2_bn_running_var = kwargs.get("proj2_bn_running_var")

        self.proj3_conv_weight = kwargs.get("proj3_conv_weight")
        self.proj3_bn_gamma = kwargs.get("proj3_bn_gamma")
        self.proj3_bn_beta = kwargs.get("proj3_bn_beta")
        self.proj3_bn_running_mean = kwargs.get("proj3_bn_running_mean")
        self.proj3_bn_running_var = kwargs.get("proj3_bn_running_var")

        self.rpe_conv_weight = kwargs.get("rpe_conv_weight")
        self.rpe_bn_gamma = kwargs.get("rpe_bn_gamma")
        self.rpe_bn_beta = kwargs.get("rpe_bn_beta")
        self.rpe_bn_running_mean = kwargs.get("rpe_bn_running_mean")
        self.rpe_bn_running_var = kwargs.get("rpe_bn_running_var")

        self.lif_proj_u = Var(shape=(TB, self.embed_dims//8, self.image_size_h, self.image_size_w), init=0)
        self.lif_proj_v = Var(shape=(TB, self.embed_dims//8, self.image_size_h, self.image_size_w), init=0)
        self.lif_proj_bias_mant = Var(shape=(TB, self.embed_dims//8, self.image_size_h, self.image_size_w), init=0)
        self.lif_proj_du = Var(shape=(1,), init=0)
        self.lif_proj_dv = Var(shape=(1,), init=0)
        self.lif_proj_vth = Var(shape=(1,), init=0)

        self.lif_proj1_u = Var(shape=(TB, self.embed_dims//4, self.image_size_h, self.image_size_w), init=0)
        self.lif_proj1_v = Var(shape=(TB, self.embed_dims//4, self.image_size_h, self.image_size_w), init=0)
        self.lif_proj1_bias_mant = Var(shape=(TB, self.embed_dims//4, self.image_size_h, self.image_size_w), init=0)
        self.lif_proj1_du = Var(shape=(1,), init=0)
        self.lif_proj1_dv = Var(shape=(1,), init=0)
        self.lif_proj1_vth = Var(shape=(1,), init=0)

        self.lif_proj2_u = Var(shape=(TB, self.embed_dims//2, self.image_size_h, self.image_size_w), init=0)
        self.lif_proj2_v = Var(shape=(TB, self.embed_dims//2, self.image_size_h, self.image_size_w), init=0)
        self.lif_proj2_bias_mant = Var(shape=(TB, self.embed_dims//2, self.image_size_h, self.image_size_w), init=0)
        self.lif_proj2_du = Var(shape=(1,), init=0)
        self.lif_proj2_dv = Var(shape=(1,), init=0)
        self.lif_proj2_vth = Var(shape=(1,), init=0)

        self.lif_proj3_u = Var(shape=(TB, self.embed_dims, self.image_size_h//2, self.image_size_w//2), init=0)
        self.lif_proj3_v = Var(shape=(TB, self.embed_dims, self.image_size_h//2, self.image_size_w//2), init=0)
        self.lif_proj3_bias_mant = Var(shape=(TB, self.embed_dims, self.image_size_h//2, self.image_size_w//2), init=0)
        self.lif_proj3_du = Var(shape=(1,), init=0)
        self.lif_proj3_dv = Var(shape=(1,), init=0)
        self.lif_proj3_vth = Var(shape=(1,), init=0)

        self.lif_rpe_u = Var(shape=(TB, self.embed_dims, self.image_size_h//4, self.image_size_w//4), init=0)
        self.lif_rpe_v = Var(shape=(TB, self.embed_dims, self.image_size_h//4, self.image_size_w//4), init=0)
        self.lif_rpe_bias_mant = Var(shape=(TB, self.embed_dims, self.image_size_h//4, self.image_size_w//4), init=0)
        self.lif_rpe_du = Var(shape=(1,), init=0)
        self.lif_rpe_dv = Var(shape=(1,), init=0)
        self.lif_rpe_vth = Var(shape=(1,), init=0)

        self.tensor_in_x = InPort(shape=(TB, self.in_channels, self.image_size_h, self.image_size_w))
        self.tensor_out = OutPort(shape=(TB, self.image_size_h*self.image_size_w//16, self.embed_dims))

@implements(proc=SPS, protocol=LoihiProtocol)
@requires(CPU)
class PySPSModel(AbstractSubProcessModel):
    def __init__(self, proc):
        (TB,) = proc.shape

        self.proj_conv = Conv2D_init(shape=(TB, proc.in_channels, proc.embed_dims//8, proc.image_size_h, proc.image_size_w), weight=proc.proj_conv_weight)
        self.proj_bn = BatchNorm2d(shape=(TB, proc.embed_dims//8, proc.image_size_h, proc.image_size_w), gamma=proc.proj_bn_gamma, beta=proc.proj_bn_beta, running_mean=proc.proj_bn_running_mean, running_var=proc.proj_bn_running_mean)
        self.proj_lif = LIF(shape=(TB, proc.embed_dims//8, proc.image_size_h, proc.image_size_w))

        self.proj1_conv = Conv2D(shape=(TB, proc.embed_dims//8, proc.embed_dims//4, proc.image_size_h, proc.image_size_w), weight=proc.proj1_conv_weight)
        self.proj1_bn = BatchNorm2d(shape=(TB, proc.embed_dims//4, proc.image_size_h, proc.image_size_w), gamma=proc.proj1_bn_gamma, beta=proc.proj1_bn_beta, running_mean=proc.proj1_bn_running_mean, running_var=proc.proj1_bn_running_mean)
        self.proj1_lif = LIF(shape=(TB, proc.embed_dims//4, proc.image_size_h, proc.image_size_w))

        self.proj2_conv = Conv2D(shape=(TB, proc.embed_dims//4, proc.embed_dims//2, proc.image_size_h, proc.image_size_w), weight=proc.proj2_conv_weight)
        self.proj2_bn = BatchNorm2d(shape=(TB, proc.embed_dims//2, proc.image_size_h, proc.image_size_w), gamma=proc.proj2_bn_gamma, beta=proc.proj2_bn_beta, running_mean=proc.proj2_bn_running_mean, running_var=proc.proj2_bn_running_mean)
        self.proj2_lif = LIF(shape=(TB, proc.embed_dims//2, proc.image_size_h, proc.image_size_w))
        self.maxpool2 = MaxPool2D(shape=(TB, proc.embed_dims//2, proc.image_size_h, proc.image_size_w, proc.image_size_h//2, proc.image_size_w//2))

        self.proj3_conv = Conv2D(shape=(TB, proc.embed_dims//2, proc.embed_dims, proc.image_size_h//2, proc.image_size_w//2), weight=proc.proj3_conv_weight)
        self.proj3_bn = BatchNorm2d(shape=(TB, proc.embed_dims, proc.image_size_h//2, proc.image_size_w//2), gamma=proc.proj3_bn_gamma, beta=proc.proj3_bn_beta, running_mean=proc.proj3_bn_running_mean, running_var=proc.proj3_bn_running_mean)
        self.proj3_lif = LIF(shape=(TB, proc.embed_dims, proc.image_size_h//2, proc.image_size_w//2))
        self.maxpool3 = MaxPool2D(shape=(TB, proc.embed_dims, proc.image_size_h//2, proc.image_size_w//2, proc.image_size_h//4, proc.image_size_w//4))

        self.rpe_conv = Conv2D(shape=(TB, proc.embed_dims, proc.embed_dims, proc.image_size_h//4, proc.image_size_w//4), weight=proc.rpe_conv_weight)
        self.rpe_bn = BatchNorm2d(shape=(TB, proc.embed_dims, proc.image_size_h//4, proc.image_size_w//4), gamma=proc.rpe_bn_gamma, beta=proc.rpe_bn_beta, running_mean=proc.rpe_bn_running_mean, running_var=proc.rpe_bn_running_mean)
        self.rpe_lif = LIF(shape=(TB, proc.embed_dims, proc.image_size_h//4, proc.image_size_w//4))

        self.residual = Residual(shape=(TB, proc.embed_dims, proc.image_size_h//4, proc.image_size_w//4))
        self.flatten = Flatten(shape=(TB, proc.embed_dims, proc.image_size_h//4, proc.image_size_w//4))
        self.transpose = Transpose(shape=(TB, proc.embed_dims, proc.image_size_h*proc.image_size_w//16))

        proc.tensor_in_x.connect(self.proj_conv.tensor_in_x)
        self.proj_conv.tensor_out.connect(self.proj_bn.tensor_in_x)
        self.proj_bn.tensor_out.connect(self.proj_lif.a_in)
        self.proj_lif.s_out.connect(self.proj1_conv.tensor_in_x)
        self.proj1_conv.tensor_out.connect(self.proj1_bn.tensor_in_x)
        self.proj1_bn.tensor_out.connect(self.proj1_lif.a_in)
        self.proj1_lif.s_out.connect(self.proj2_conv.tensor_in_x)
        self.proj2_conv.tensor_out.connect(self.proj2_bn.tensor_in_x)
        self.proj2_bn.tensor_out.connect(self.proj2_lif.a_in)
        self.proj2_lif.s_out.connect(self.maxpool2.tensor_in_x)
        self.maxpool2.tensor_out.connect(self.proj3_conv.tensor_in_x)
        self.proj3_conv.tensor_out.connect(self.proj3_bn.tensor_in_x)
        self.proj3_bn.tensor_out.connect(self.proj3_lif.a_in)
        self.proj3_lif.s_out.connect(self.maxpool3.tensor_in_x)
        self.maxpool3.tensor_out.connect(self.rpe_conv.tensor_in_x)
        self.rpe_conv.tensor_out.connect(self.rpe_bn.tensor_in_x)
        self.rpe_bn.tensor_out.connect(self.rpe_lif.a_in)
        self.rpe_lif.s_out.connect(self.residual.tensor_in_x)
        self.maxpool3.tensor_out.connect(self.residual.tensor_in_x1)
        self.residual.tensor_out.connect(self.flatten.tensor_in_x)
        self.flatten.tensor_out.connect(self.transpose.mat_in)
        self.transpose.mat_out.connect(proc.tensor_out)

        proc.vars.lif_proj_u.alias(self.proj_lif.vars.u)
        proc.vars.lif_proj_v.alias(self.proj_lif.vars.v)
        proc.vars.lif_proj_bias_mant.alias(self.proj_lif.vars.bias_mant)
        proc.vars.lif_proj_du.alias(self.proj_lif.vars.du)
        proc.vars.lif_proj_dv.alias(self.proj_lif.vars.dv)
        proc.vars.lif_proj_vth.alias(self.proj_lif.vars.vth)

        proc.vars.lif_proj1_u.alias(self.proj1_lif.vars.u)
        proc.vars.lif_proj1_v.alias(self.proj1_lif.vars.v)
        proc.vars.lif_proj1_bias_mant.alias(self.proj1_lif.vars.bias_mant)
        proc.vars.lif_proj1_du.alias(self.proj1_lif.vars.du)
        proc.vars.lif_proj1_dv.alias(self.proj1_lif.vars.dv)
        proc.vars.lif_proj1_vth.alias(self.proj1_lif.vars.vth)

        proc.vars.lif_proj2_u.alias(self.proj2_lif.vars.u)
        proc.vars.lif_proj2_v.alias(self.proj2_lif.vars.v)
        proc.vars.lif_proj2_bias_mant.alias(self.proj2_lif.vars.bias_mant)
        proc.vars.lif_proj2_du.alias(self.proj2_lif.vars.du)
        proc.vars.lif_proj2_dv.alias(self.proj2_lif.vars.dv)
        proc.vars.lif_proj2_vth.alias(self.proj2_lif.vars.vth)

        proc.vars.lif_proj3_u.alias(self.proj3_lif.vars.u)
        proc.vars.lif_proj3_v.alias(self.proj3_lif.vars.v)
        proc.vars.lif_proj3_bias_mant.alias(self.proj3_lif.vars.bias_mant)
        proc.vars.lif_proj3_du.alias(self.proj3_lif.vars.du)
        proc.vars.lif_proj3_dv.alias(self.proj3_lif.vars.dv)
        proc.vars.lif_proj3_vth.alias(self.proj3_lif.vars.vth)

        proc.vars.lif_rpe_u.alias(self.rpe_lif.vars.u)
        proc.vars.lif_rpe_v.alias(self.rpe_lif.vars.v)
        proc.vars.lif_rpe_bias_mant.alias(self.rpe_lif.vars.bias_mant)
        proc.vars.lif_rpe_du.alias(self.rpe_lif.vars.du)
        proc.vars.lif_rpe_dv.alias(self.rpe_lif.vars.dv)
        proc.vars.lif_rpe_vth.alias(self.rpe_lif.vars.vth)

class InputGenerator(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        self.tensor_out = OutPort(shape=shape)

@implements(proc=InputGenerator, protocol=LoihiProtocol)
@requires(CPU)
class PyInputGeneratorModel(PyLoihiProcessModel):
    tensor_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)
    def run_spk(self):
        shape = self.tensor_out.shape
        tensor_out = np.random.rand(*shape)
        self.tensor_out.send(tensor_out)

class OutputReceiver(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape")
        self.tensor_in_x = InPort(shape=shape)
        self.tensor_result = Var(shape=shape, init=0)

@implements(proc=OutputReceiver, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputReceiverModel(PyLoihiProcessModel):
    tensor_in_x: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)
    tensor_result: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    def run_spk(self):
        tensor_in_x = self.tensor_in_x.recv()
        self.tensor_result = tensor_in_x

def test_SPS():
    shape = (4,) # (TB, N, C)
    image_size_h = 32
    image_size_w = 32
    patch_size = 4
    in_channels = 3
    embed_dims = 256

    proj_conv_weight = np.random.rand(embed_dims//8, in_channels, 3, 3)
    proj_bn_gamma = np.ones(embed_dims//8)
    proj_bn_beta = np.zeros(embed_dims//8)

    proj1_conv_weight = np.random.rand(embed_dims//4, embed_dims//8, 3, 3)
    proj1_bn_gamma = np.ones(embed_dims//4)
    proj1_bn_beta = np.zeros(embed_dims//4)

    proj2_conv_weight = np.random.rand(embed_dims//2, embed_dims//4, 3, 3)
    proj2_bn_gamma = np.ones(embed_dims//2)
    proj2_bn_beta = np.zeros(embed_dims//2)

    proj3_conv_weight = np.random.rand(embed_dims, embed_dims//2, 3, 3)
    proj3_bn_gamma = np.ones(embed_dims)
    proj3_bn_beta = np.zeros(embed_dims)

    rpe_conv_weight = np.random.rand(embed_dims, embed_dims, 3, 3)
    rpe_bn_gamma = np.ones(embed_dims)
    rpe_bn_beta = np.zeros(embed_dims)

    SPS_block = SPS(shape=shape,
                    image_size_h=image_size_h,
                    image_size_w=image_size_w,
                    patch_size=patch_size,
                    in_channels=in_channels,
                    embed_dims=embed_dims,
                    proj_conv_weight=proj_conv_weight,
                    proj_bn_gamma=proj_bn_gamma,
                    proj_bn_beta=proj_bn_beta,
                    proj1_conv_weight=proj1_conv_weight,
                    proj1_bn_gamma=proj1_bn_gamma,
                    proj1_bn_beta=proj1_bn_beta,
                    proj2_conv_weight=proj2_conv_weight,
                    proj2_bn_gamma=proj2_bn_gamma,
                    proj2_bn_beta=proj2_bn_beta,
                    proj3_conv_weight=proj3_conv_weight,
                    proj3_bn_gamma=proj3_bn_gamma,
                    proj3_bn_beta=proj3_bn_beta,
                    rpe_conv_weight=rpe_conv_weight,
                    rpe_bn_gamma=rpe_bn_gamma,
                    rpe_bn_beta=rpe_bn_beta)

    input_process = InputGenerator(shape=(shape[0], in_channels, image_size_h, image_size_w))
    output_process = OutputReceiver(shape=(shape[0], image_size_h*image_size_w//16, embed_dims))

    input_process.tensor_out.connect(SPS_block.tensor_in_x)
    SPS_block.tensor_out.connect(output_process.tensor_in_x)

    rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=True)

    for t in range(9):
        # Run the entire network of Processes.
        SPS_block.run(condition=RunSteps(num_steps=1), run_cfg=rcfg)
        print('t: ',t)
        print('this_block result: ', output_process.tensor_result.get())
        print('\n ----- \n')


if __name__ == "__main__":
    test_SPS()
