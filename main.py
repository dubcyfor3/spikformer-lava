import sys
import os
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
        dict = torch.load('input_data.pth')
        image_out = dict['input_tensor'].cpu().numpy()
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

def test_spikformer(checkpoint=None):
    timestep = 4
    batch_size = 8
    image_size_h = 32
    image_size_w = 32
    patch_size = 4
    in_channels = 3
    embed_dims = 384 # 384
    hidden_dim = 384*4 # 384*4
    num_heads = 12 # 12
    depth = 4

    # set seed for reproducibility
    np.random.seed(0)
    # weight for SPS
    # proj_conv_weight = np.random.rand(embed_dims//8, in_channels, 3, 3)
    # proj_bn_gamma = np.ones(embed_dims//8)
    # proj_bn_beta = np.zeros(embed_dims//8)

    proj_conv_weight = checkpoint['patch_embed.proj_conv.weight'].numpy()
    proj_bn_gamma = checkpoint['patch_embed.proj_bn.weight'].numpy()
    proj_bn_beta = checkpoint['patch_embed.proj_bn.bias'].numpy()
    proj_bn_running_mean = checkpoint['patch_embed.proj_bn.running_mean'].numpy()
    proj_bn_running_var = checkpoint['patch_embed.proj_bn.running_var'].numpy()

    # proj1_conv_weight = np.random.rand(embed_dims//4, embed_dims//8, 3, 3)
    # proj1_bn_gamma = np.ones(embed_dims//4)
    # proj1_bn_beta = np.zeros(embed_dims//4)


    proj1_conv_weight = checkpoint['patch_embed.proj_conv1.weight'].numpy()
    proj1_bn_gamma = checkpoint['patch_embed.proj_bn1.weight'].numpy()
    proj1_bn_beta = checkpoint['patch_embed.proj_bn1.bias'].numpy()
    proj1_bn_running_mean = checkpoint['patch_embed.proj_bn1.running_mean'].numpy()
    proj1_bn_running_var = checkpoint['patch_embed.proj_bn1.running_var'].numpy()


    # proj2_conv_weight = np.random.rand(embed_dims//2, embed_dims//4, 3, 3)
    # proj2_bn_gamma = np.ones(embed_dims//2)
    # proj2_bn_beta = np.zeros(embed_dims//2)

    proj2_conv_weight = checkpoint['patch_embed.proj_conv2.weight'].numpy()
    proj2_bn_gamma = checkpoint['patch_embed.proj_bn2.weight'].numpy()
    proj2_bn_beta = checkpoint['patch_embed.proj_bn2.bias'].numpy()
    proj2_bn_running_mean = checkpoint['patch_embed.proj_bn2.running_mean'].numpy()
    proj2_bn_running_var = checkpoint['patch_embed.proj_bn2.running_var'].numpy()

    # proj3_conv_weight = np.random.rand(embed_dims, embed_dims//2, 3, 3)
    # proj3_bn_gamma = np.ones(embed_dims)
    # proj3_bn_beta = np.zeros(embed_dims)

    proj3_conv_weight = checkpoint['patch_embed.proj_conv3.weight'].numpy()
    proj3_bn_gamma = checkpoint['patch_embed.proj_bn3.weight'].numpy()
    proj3_bn_beta = checkpoint['patch_embed.proj_bn3.bias'].numpy()
    proj3_bn_running_mean = checkpoint['patch_embed.proj_bn3.running_mean'].numpy()
    proj3_bn_running_var = checkpoint['patch_embed.proj_bn3.running_var'].numpy()

    # rpe_conv_weight = np.random.rand(embed_dims, embed_dims, 3, 3)
    # rpe_bn_gamma = np.ones(embed_dims)
    # rpe_bn_beta = np.zeros(embed_dims)

    rpe_conv_weight = checkpoint['patch_embed.rpe_conv.weight'].numpy()
    rpe_bn_gamma = checkpoint['patch_embed.rpe_bn.weight'].numpy()
    rpe_bn_beta = checkpoint['patch_embed.rpe_bn.bias'].numpy()
    rpe_bn_running_mean = checkpoint['patch_embed.rpe_bn.running_mean'].numpy()
    rpe_bn_running_var = checkpoint['patch_embed.rpe_bn.running_var'].numpy()


    SPS_block = SPS(shape=(batch_size,),
                image_size_h=image_size_h,
                image_size_w=image_size_w,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dims=embed_dims,
                proj_conv_weight=proj_conv_weight,
                proj_bn_gamma=proj_bn_gamma,
                proj_bn_beta=proj_bn_beta,
                proj_bn_running_mean=proj_bn_running_mean,
                proj_bn_running_var=proj_bn_running_var,
                proj1_conv_weight=proj1_conv_weight,
                proj1_bn_gamma=proj1_bn_gamma,
                proj1_bn_beta=proj1_bn_beta,
                proj1_bn_running_mean=proj1_bn_running_mean,
                proj1_bn_running_var=proj1_bn_running_var,
                proj2_conv_weight=proj2_conv_weight,
                proj2_bn_gamma=proj2_bn_gamma,
                proj2_bn_beta=proj2_bn_beta,
                proj2_bn_running_mean=proj2_bn_running_mean,
                proj2_bn_running_var=proj2_bn_running_var,
                proj3_conv_weight=proj3_conv_weight,
                proj3_bn_gamma=proj3_bn_gamma,
                proj3_bn_beta=proj3_bn_beta,
                proj3_bn_running_mean=proj3_bn_running_mean,
                proj3_bn_running_var=proj3_bn_running_var,
                rpe_conv_weight=rpe_conv_weight,
                rpe_bn_gamma=rpe_bn_gamma,
                rpe_bn_beta=rpe_bn_beta,
                rpe_bn_running_mean=rpe_bn_running_mean,
                rpe_bn_running_var=rpe_bn_running_var)

    classifier_weight = checkpoint['head.weight'].numpy()
    classifier_bias = checkpoint['head.bias'].numpy()

    class_output = ClassOutput(shape=(batch_size, embed_dims), num_time_steps=timestep, weight=classifier_weight, bias=classifier_bias)
    mean_output = Mean(shape=(batch_size, image_size_h*image_size_w//16, embed_dims), dim=1)
    image_input = ImageInput(shape=(batch_size, in_channels, image_size_h, image_size_w))

    block_list_SSA = []
    block_list_MLP = []
    for i in range(depth):
        # weight = np.random.rand(embed_dims, embed_dims)
        # bias = np.random.rand(embed_dims)

        # gamma = np.ones(embed_dims)
        # beta = np.zeros(embed_dims)

        this_SSA = SSA(shape=(batch_size, image_size_h*image_size_w//16, embed_dims),
                     weight_q_linear=checkpoint['block.{}.attn.q_linear.weight'.format(i)].numpy(),
                     bias_q_linear=checkpoint['block.{}.attn.q_linear.bias'.format(i)].numpy(),
                     weight_k_linear=checkpoint['block.{}.attn.k_linear.weight'.format(i)].numpy(),
                     bias_k_linear=checkpoint['block.{}.attn.k_linear.bias'.format(i)].numpy(),
                     weight_v_linear=checkpoint['block.{}.attn.v_linear.weight'.format(i)].numpy(),
                     bias_v_linear=checkpoint['block.{}.attn.v_linear.bias'.format(i)].numpy(),
                     weight_proj_linear=checkpoint['block.{}.attn.proj_linear.weight'.format(i)].numpy(),
                     bias_proj_linear=checkpoint['block.{}.attn.proj_linear.bias'.format(i)].numpy(),
                        beta_q_bn=checkpoint['block.{}.attn.q_bn.bias'.format(i)].numpy(),
                        gamma_q_bn=checkpoint['block.{}.attn.q_bn.weight'.format(i)].numpy(),
                        running_mean_q_bn=checkpoint['block.{}.attn.q_bn.running_mean'.format(i)].numpy(),
                        running_var_q_bn=checkpoint['block.{}.attn.q_bn.running_var'.format(i)].numpy(),
                        beta_k_bn=checkpoint['block.{}.attn.k_bn.bias'.format(i)].numpy(),
                        gamma_k_bn=checkpoint['block.{}.attn.k_bn.weight'.format(i)].numpy(),
                        running_mean_k_bn=checkpoint['block.{}.attn.k_bn.running_mean'.format(i)].numpy(),
                        running_var_k_bn=checkpoint['block.{}.attn.k_bn.running_var'.format(i)].numpy(),
                        beta_v_bn=checkpoint['block.{}.attn.v_bn.bias'.format(i)].numpy(),
                        gamma_v_bn=checkpoint['block.{}.attn.v_bn.weight'.format(i)].numpy(),
                        running_mean_v_bn=checkpoint['block.{}.attn.v_bn.running_mean'.format(i)].numpy(),
                        running_var_v_bn=checkpoint['block.{}.attn.v_bn.running_var'.format(i)].numpy(),
                        beta_proj_bn=checkpoint['block.{}.attn.proj_bn.bias'.format(i)].numpy(),
                        gamma_proj_bn=checkpoint['block.{}.attn.proj_bn.weight'.format(i)].numpy(),
                        running_mean_proj_bn=checkpoint['block.{}.attn.proj_bn.running_mean'.format(i)].numpy(),
                        running_var_proj_bn=checkpoint['block.{}.attn.proj_bn.running_var'.format(i)].numpy(),
                     num_heads=num_heads)

        block_list_SSA.append(this_SSA)

        # fc1_linear_weight = np.random.rand(hidden_dim, embed_dims)
        # fc1_linear_bias = np.random.rand(hidden_dim)
        # fc2_linear_weight = np.random.rand(embed_dims, hidden_dim)
        # fc2_linear_bias = np.random.rand(embed_dims)

        # fc1_bn_gamma = np.ones(hidden_dim)
        # fc1_bn_beta = np.zeros(hidden_dim)
        # fc2_bn_gamma = np.ones(embed_dims)
        # fc2_bn_beta = np.zeros(embed_dims)

        this_MLP = SpikingMLP(shape=(batch_size, image_size_h*image_size_w//16, embed_dims, hidden_dim, embed_dims),
                 fc1_linear_weight=checkpoint['block.{}.mlp.fc1_linear.weight'.format(i)].numpy(),
                 fc1_linear_bias=checkpoint['block.{}.mlp.fc1_linear.bias'.format(i)].numpy(),
                 fc2_linear_weight=checkpoint['block.{}.mlp.fc2_linear.weight'.format(i)].numpy(),
                 fc2_linear_bias=checkpoint['block.{}.mlp.fc2_linear.bias'.format(i)].numpy(),
                 fc1_bn_gamma=checkpoint['block.{}.mlp.fc1_bn.weight'.format(i)].numpy(),
                 fc1_bn_beta=checkpoint['block.{}.mlp.fc1_bn.bias'.format(i)].numpy(),
                 fc1_bn_running_mean=checkpoint['block.{}.mlp.fc1_bn.running_mean'.format(i)].numpy(),
                 fc1_bn_running_var=checkpoint['block.{}.mlp.fc1_bn.running_var'.format(i)].numpy(),
                 fc2_bn_gamma=checkpoint['block.{}.mlp.fc2_bn.weight'.format(i)].numpy(),
                 fc2_bn_beta=checkpoint['block.{}.mlp.fc2_bn.bias'.format(i)].numpy(),
                 fc2_bn_running_mean=checkpoint['block.{}.mlp.fc2_bn.running_mean'.format(i)].numpy(),
                 fc2_bn_running_var=checkpoint['block.{}.mlp.fc2_bn.running_var'.format(i)].numpy())

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

    SPS_block.stop()

if __name__ == '__main__':
    checkpoint_path = 'model_best.pth.tar'
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
            # print all keys
            # print(checkpoint.keys())
    test_spikformer(checkpoint=checkpoint)
