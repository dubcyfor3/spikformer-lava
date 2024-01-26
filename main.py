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
from timm.data import create_dataset
from SPS import SPS
from SSA import SSA, Linear
from MLP import SpikingMLP


class ImageInput(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        self.shape = kwargs.get("shape")
        self.image_out = OutPort(shape=self.shape)

@implements(proc=ImageInput, protocol=LoihiProtocol)
@requires(CPU)
class PyImageInputModel(PyLoihiProcessModel):
    image_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)

    def run_spk(self):
        image_out = torch.rand(self.image_out.shape)
        self.image_out.send(image_out)

class ClassOutput(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_time_steps_per_imgage = kwargs.get("num_time_steps")
        self.shape = kwargs.get("shape")
        self.num_time_steps_per_image = Var(shape=(1,), init=num_time_steps_per_imgage)
        self.tensor_in = InPort(shape=self.shape)
        self.tensor_stored = Var(shape=(num_time_steps_per_imgage, *self.shape))
        self.classify_result = Var(shape=(self.shape[0], 10), init=-1)

@implements(proc=ClassOutput, protocol=LoihiProtocol)
@requires(CPU)
class PyClassOutputModel(PyLoihiProcessModel):
    num_time_steps_per_image: int = LavaPyType(int, int, precision=32)
    tensor_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)
    tensor_stored: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    classify_result: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.shape = proc_params._parameters.get("shape")
        self.weight = proc_params._parameters.get("weight")
        self.bias = proc_params._parameters.get("bias")
        input_dim = self.shape[-1]
        self.linear_classfier = nn.Linear(input_dim, 10)
        self.linear_classfier.weight.data = torch.from_numpy(self.weight).float()
        self.linear_classfier.bias.data = torch.from_numpy(self.bias).float()

    def post_guard(self):
        if self.time_step % self.num_time_steps_per_image == 0:
            return True
        else:
            return False

    def run_post_mgmt(self):
        tensor_stored = torch.from_numpy(self.tensor_stored).float()
        tensor_in = tensor_stored.mean(dim=0)
        self.classify_result = self.linear_classfier(tensor_in).detach().numpy()
        self.tensor_stored = np.zeros(self.tensor_stored.shape)


    def run_spk(self):
        tensor_in = self.tensor_in.recv()
        self.tensor_stored[self.time_step % self.num_time_steps_per_image] = tensor_in


class Mean(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shape = kwargs.get("shape")
        self.dim = kwargs.get("dim")
        shape_out = tuple([dim_len for i,dim_len in enumerate(self.shape) if i != self.dim])
        self.tensor_in = InPort(shape=self.shape)
        self.tensor_out = OutPort(shape=shape_out)

@implements(proc=Mean, protocol=LoihiProtocol)
@requires(CPU)
class PyMeanModel(PyLoihiProcessModel):
    tensor_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=32)
    tensor_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.dim = proc_params._parameters.get("dim")

    def run_spk(self):
        tensor_in = self.tensor_in.recv()
        tensor_in = torch.from_numpy(tensor_in)
        tensor_out = torch.mean(tensor_in, dim=self.dim).numpy()
        self.tensor_out.send(tensor_out)

def test_spikformer():
    timestep = 4
    batch_size = 8
    image_size_h = 16
    image_size_w = 16
    patch_size = 4
    in_channels = 3
    embed_dims = 64 # 384
    hidden_dim = 64*4 # 384*4
    num_heads = 4 # 12
    depth = 4

    # set seed for reproducibility
    np.random.seed(0)
    # weight for SPS
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

    SPS_block = SPS(shape=(batch_size,),
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

    classifier_weight = np.random.rand(10, embed_dims)
    classifier_bias = np.random.rand(10)

    class_output = ClassOutput(shape=(batch_size, embed_dims), num_time_steps=timestep, weight=classifier_weight, bias=classifier_bias)
    mean_output = Mean(shape=(batch_size, image_size_h*image_size_w//16, embed_dims), dim=1)
    image_input = ImageInput(shape=(batch_size, in_channels, image_size_h, image_size_w))

    block_list_SSA = []
    block_list_MLP = []
    for i in range(depth):
        weight = np.random.rand(embed_dims, embed_dims)
        bias = np.random.rand(embed_dims)

        gamma = np.ones(embed_dims)
        beta = np.zeros(embed_dims)

        this_SSA = SSA(shape=(batch_size, image_size_h*image_size_w//16, embed_dims),
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

        block_list_SSA.append(this_SSA)

        fc1_linear_weight = np.random.rand(hidden_dim, embed_dims)
        fc1_linear_bias = np.random.rand(hidden_dim)
        fc2_linear_weight = np.random.rand(embed_dims, hidden_dim)
        fc2_linear_bias = np.random.rand(embed_dims)

        fc1_bn_gamma = np.ones(hidden_dim)
        fc1_bn_beta = np.zeros(hidden_dim)
        fc2_bn_gamma = np.ones(embed_dims)
        fc2_bn_beta = np.zeros(embed_dims)

        this_MLP = SpikingMLP(shape=(batch_size, image_size_h*image_size_w//16, embed_dims, hidden_dim, embed_dims),
                 fc1_linear_weight=fc1_linear_weight,
                 fc1_linear_bias=fc1_linear_bias,
                 fc2_linear_weight=fc2_linear_weight,
                 fc2_linear_bias=fc2_linear_bias,
                 fc1_bn_gamma=fc1_bn_gamma,
                 fc1_bn_beta=fc1_bn_beta,
                 fc2_bn_gamma=fc2_bn_gamma,
                 fc2_bn_beta=fc2_bn_beta)

        block_list_MLP.append(this_MLP)

    SPS_block.tensor_out.connect(block_list_SSA[0].mat_in_x)

    image_input.image_out.connect(SPS_block.tensor_in_x)

    for i in range(depth):
        block_list_SSA[i].mat_out.connect(block_list_MLP[i].tensor_in_x)
        if i != depth-1:
            block_list_MLP[i].tensor_out.connect(block_list_SSA[i+1].mat_in_x)

    block_list_MLP[-1].tensor_out.connect(mean_output.tensor_in)
    mean_output.tensor_out.connect(class_output.tensor_in)

    rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=True)

    for t in range(timestep):
        SPS_block.run(condition=RunSteps(num_steps=1), run_cfg=rcfg)
        print('t: ',t)
        print('classfy result: ', class_output.classify_result.get())
        print('\n ----- \n')

if __name__ == '__main__':
    test_spikformer()
