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

from SSA import Linear, BatchNorm1d, LIF, InputGenerator, OutputReceiver

class SpikingMLP(AbstractProcess):
    
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape")
        (TB, N, C_input, C_hidden, C_output) = shape
        self.shape = shape
        self.fc1_linear_weight = kwargs.get("fc1_linear_weight")
        self.fc1_linear_bias = kwargs.get("fc1_linear_bias")
        self.fc2_linear_weight = kwargs.get("fc2_linear_weight")
        self.fc2_linear_bias = kwargs.get("fc2_linear_bias")
        self.fc1_bn_gamma = kwargs.get("fc1_bn_gamma")
        self.fc1_bn_beta = kwargs.get("fc1_bn_beta")
        self.fc2_bn_gamma = kwargs.get("fc2_bn_gamma")
        self.fc2_bn_beta = kwargs.get("fc2_bn_beta")

        self.lif_fc1_u = Var(shape=(TB, N, C_hidden), init=0)
        self.lif_fc1_v = Var(shape=(TB, N, C_hidden), init=0)
        self.lif_fc1_bias_mant = Var(shape=(TB, N, C_hidden), init=0)
        self.lif_fc1_du = Var(shape=(1,), init=0)
        self.lif_fc1_dv = Var(shape=(1,), init=0)
        self.lif_fc1_vth = Var(shape=(1,), init=0)

        self.lif_fc2_u = Var(shape=(TB, N, C_output), init=0)
        self.lif_fc2_v = Var(shape=(TB, N, C_output), init=0)
        self.lif_fc2_bias_mant = Var(shape=(TB, N, C_output), init=0)
        self.lif_fc2_du = Var(shape=(1,), init=0)
        self.lif_fc2_dv = Var(shape=(1,), init=0)
        self.lif_fc2_vth = Var(shape=(1,), init=0)

        self.tensor_in_x = InPort(shape=(TB, N, C_input))
        self.tensor_out = OutPort(shape=(TB, N, C_output))

@implements(proc=SpikingMLP, protocol=LoihiProtocol)
@requires(CPU)
class PySpikingMLPModel(AbstractSubProcessModel):
    def __init__(self, proc):

        (TB, N, C_input, C_hidden, C_output) = proc.shape
        self.fc1_linear = Linear(shape=(TB, N, C_input, C_hidden), weight=proc.fc1_linear_weight, bias=proc.fc1_linear_bias)
        self.fc1_bn = BatchNorm1d(shape=(TB, N, C_hidden), gamma=proc.fc1_bn_gamma, beta=proc.fc1_bn_beta)
        self.fc1_lif = LIF(shape=(TB, N, C_hidden))

        self.fc2_linear = Linear(shape=(TB, N, C_hidden, C_output), weight=proc.fc2_linear_weight, bias=proc.fc2_linear_bias)
        self.fc2_bn = BatchNorm1d(shape=(TB, N, C_output), gamma=proc.fc2_bn_gamma, beta=proc.fc2_bn_beta)
        self.fc2_lif = LIF(shape=(TB, N, C_output))

        proc.tensor_in_x.connect(self.fc1_linear.mat_in)
        self.fc1_linear.mat_out.connect(self.fc1_bn.mat_in)
        self.fc1_bn.mat_out.connect(self.fc1_lif.a_in)
        self.fc1_lif.s_out.connect(self.fc2_linear.mat_in)
        self.fc2_linear.mat_out.connect(self.fc2_bn.mat_in)
        self.fc2_bn.mat_out.connect(self.fc2_lif.a_in)
        self.fc2_lif.s_out.connect(proc.tensor_out)

        proc.vars.lif_fc1_u.alias(self.fc1_lif.vars.u)
        proc.vars.lif_fc1_v.alias(self.fc1_lif.vars.v)
        proc.vars.lif_fc1_bias_mant.alias(self.fc1_lif.vars.bias_mant)
        proc.vars.lif_fc1_du.alias(self.fc1_lif.vars.du)
        proc.vars.lif_fc1_dv.alias(self.fc1_lif.vars.dv)
        proc.vars.lif_fc1_vth.alias(self.fc1_lif.vars.vth)

        proc.vars.lif_fc2_u.alias(self.fc2_lif.vars.u)
        proc.vars.lif_fc2_v.alias(self.fc2_lif.vars.v)
        proc.vars.lif_fc2_bias_mant.alias(self.fc2_lif.vars.bias_mant)
        proc.vars.lif_fc2_du.alias(self.fc2_lif.vars.du)
        proc.vars.lif_fc2_dv.alias(self.fc2_lif.vars.dv)
        proc.vars.lif_fc2_vth.alias(self.fc2_lif.vars.vth)


def test_MLP_block():
    shape = (4, 5, 6, 7, 6) # (TB, N, C_input, C_hidden, C_output)
    (TB, N, C_input, C_hidden, C_output) = shape
    # generate random weights and bias
    fc1_linear_weight = np.random.rand(shape[3], shape[2])
    fc1_linear_bias = np.random.rand(shape[3])
    fc2_linear_weight = np.random.rand(shape[4], shape[3])
    fc2_linear_bias = np.random.rand(shape[4])

    fc1_bn_gamma = np.ones(shape[3])
    fc1_bn_beta = np.zeros(shape[3])
    fc2_bn_gamma = np.ones(shape[4])
    fc2_bn_beta = np.zeros(shape[4])

    MLP_block = SpikingMLP(shape=shape, 
                     fc1_linear_weight=fc1_linear_weight, 
                     fc1_linear_bias=fc1_linear_bias, 
                     fc2_linear_weight=fc2_linear_weight, 
                     fc2_linear_bias=fc2_linear_bias, 
                     fc1_bn_gamma=fc1_bn_gamma, 
                     fc1_bn_beta=fc1_bn_beta, 
                     fc2_bn_gamma=fc2_bn_gamma, 
                     fc2_bn_beta=fc2_bn_beta)
    
    input_process = InputGenerator(shape=(TB, N, C_input))
    output_process = OutputReceiver(shape=(TB, N, C_output))

    input_process.mat_out.connect(MLP_block.tensor_in_x)
    MLP_block.tensor_out.connect(output_process.mat_in)

    rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=True)

    for t in range(9):
        # Run the entire network of Processes.
        MLP_block.run(condition=RunSteps(num_steps=1), run_cfg=rcfg)
        print('t: ',t)
        print('MLP_block result: ', output_process.mat_result.get())
        print('\n ----- \n')





if __name__ == "__main__":
    test_MLP_block()