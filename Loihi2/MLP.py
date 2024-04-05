import sys
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU, LMT
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.c.type import LavaCType, LavaCDataType, COutPort, CInPort
from lava.magma.core.model.c.model import CLoihiProcessModel
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import Monitor

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense 
from lava.proc.sparse.process import Sparse 
from scipy.sparse import spmatrix, csr_matrix
from lava.utils.profiler import Profiler

class SpikeGenerator(AbstractProcess):
    """Spike generator process provides spikes to subsequent Processes.

    Parameters
    ----------
    shape: tuple
        defines the dimensionality of the generated spikes per timestep
    spike_prob: int
        spike probability in percent
    """
    def __init__(self, shape: tuple, spike_prob: int) -> None:        
        super().__init__()
        self.spike_prob = Var(shape=(1, ), init=spike_prob)
        # multiply all the elements of the shape tuple
        out_shape = np.prod(shape)
        self.s_out = OutPort(shape=(out_shape,))

@implements(proc=SpikeGenerator, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeGeneratorModel(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)
    spike_prob: int = LavaPyType(int, int)

    def run_spk(self):
        self.s_out.send(np.random.rand(self.s_out.shape[0]) < self.spike_prob / 100)


# @implements(proc=SpikeGenerator, protocol=LoihiProtocol)
# @requires(LMT)
# class CSpikeGeneratorModel(CLoihiProcessModel):
#     """Spike Generator process model in C."""
#     spike_prob: Var = LavaCType(cls=int, d_type=LavaCDataType.INT32)
#     s_out: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    
#     @property
#     def source_file_name(self):
#         return "spike_generator.c"

class SpikingMLPLayer(AbstractProcess):

    def __init__(self, **kwargs):
        super().__init__()
        
        self.shape = kwargs.get("shape")
        (C_input, C_output) = self.shape
        w_d = kwargs.get("fc_linear_weight")
        assert w_d.shape == (C_output, C_input)
        w_i = np.eye(C_output).astype(np.int32)
        b_d = kwargs.get("fc_bias")
        assert b_d.shape == (C_output,)
        b_i = np.random.rand(C_output).astype(np.int32)

        self.spikes_in = InPort(shape=(C_input,))
        self.spikes_out = OutPort(shape=(C_output,))
        self.w_dense = Var(shape=w_d.shape, init=w_d)
        self.b_bn = Var(shape=(w_d.shape[0],), init=b_d)
        self.w_i = Var(shape=w_i.shape, init=w_i)
        self.b_lif = Var(shape=(w_i.shape[0],), init=b_i)


        # Up-level currents and voltages of LIF Processes
        # for resetting (see at the end of the tutorial)
        self.bn_u = Var(shape=(w_d.shape[0],), init=0)
        self.bn_v = Var(shape=(w_d.shape[0],), init=0)
        self.lif_u = Var(shape=(w_i.shape[0],), init=0)
        self.lif_v = Var(shape=(w_i.shape[0],), init=0)

@implements(proc=SpikingMLPLayer, protocol=LoihiProtocol)
@requires(CPU)
class PySpikingMLPLayerModel(AbstractSubProcessModel):
    def __init__(self, proc):
        shape = proc.shape
        (C_input, C_output) = shape
        self.dense = Dense(weights=proc.w_dense.init)
        self.bn = LIF(shape=(C_output,), bias_mant=proc.b_bn.init, vth=1,
                        dv=1, du=2)
        self.eye = Sparse(weights=proc.w_i.init)
        self.lif = LIF(shape=(C_output,), bias_mant=proc.b_lif.init, vth=1,
                        dv=1, du=2)

        proc.spikes_in.connect(self.dense.s_in)
        self.dense.a_out.connect(self.bn.a_in)
        self.bn.s_out.connect(self.eye.s_in)
        self.eye.a_out.connect(self.lif.a_in)
        self.lif.s_out.connect(proc.spikes_out)
        
        # Create aliases of SubProcess variables
        proc.bn_u.alias(self.bn.u)
        proc.bn_v.alias(self.bn.v)
        proc.lif_u.alias(self.lif.u)
        proc.lif_v.alias(self.lif.v)


class SpikingMLP(AbstractProcess):

    def __init__(self, **kwargs):
        super().__init__()
        
        self.shape = kwargs.get("shape")
        (C_input, C_hidden, C_output) = self.shape
        w0 = kwargs.get("fc1_linear_weight")
        w1 = np.eye(C_hidden).astype(np.int32)
        w2 = kwargs.get("fc2_linear_weight")
        w3 = np.eye(C_output).astype(np.int32)
        b1 = np.random.rand(C_hidden).astype(np.int32)
        b2 = np.random.rand(C_hidden).astype(np.int32)
        b3 = np.random.rand(C_output).astype(np.int32)
        b4 = np.random.rand(C_output).astype(np.int32)

        self.spikes_in = InPort(shape=(C_input,))
        self.spikes_out = OutPort(shape=(C_output,))
        self.w_dense0 = Var(shape=w0.shape, init=w0)
        self.b_lif1 = Var(shape=(w0.shape[0],), init=b1)
        self.w_sparse1 = Var(shape=w1.shape, init=w1)
        self.b_lif2 = Var(shape=(w1.shape[0],), init=b2)
        self.w_dense2 = Var(shape=w2.shape, init=w2)
        self.b_lif3 = Var(shape=(w2.shape[0],), init=b3)
        self.w_sparse3 = Var(shape=w3.shape, init=w3)
        self.b_lif4 = Var(shape=(w3.shape[0],), init=b4)

        # Up-level currents and voltages of LIF Processes
        # for resetting (see at the end of the tutorial)
        self.lif1_u = Var(shape=(w0.shape[0],), init=0)
        self.lif1_v = Var(shape=(w0.shape[0],), init=0)
        self.lif2_u = Var(shape=(w1.shape[0],), init=0)
        self.lif2_v = Var(shape=(w1.shape[0],), init=0)
        self.lif3_u = Var(shape=(w2.shape[0],), init=0)
        self.lif3_v = Var(shape=(w2.shape[0],), init=0)
        self.lif4_u = Var(shape=(w3.shape[0],), init=0)
        self.lif4_v = Var(shape=(w3.shape[0],), init=0)

@implements(proc=SpikingMLP, protocol=LoihiProtocol)
@requires(CPU)
class PySpikingMLPModel(AbstractSubProcessModel):
    def __init__(self, proc):
        shape = proc.shape
        (C_input, C_hidden, C_output) = shape
        self.dense0 = Dense(weights=proc.w_dense0.init)
        self.lif1 = LIF(shape=(C_hidden,), bias_mant=proc.b_lif1.init, vth=1,
                        dv=1, du=2)
        self.sparse1 = Sparse(weights=proc.w_sparse1.init)
        self.lif2 = LIF(shape=(C_hidden,), bias_mant=proc.b_lif2.init, vth=1,
                        dv=1, du=2)
        self.dense2 = Dense(weights=proc.w_dense2.init)
        self.lif3 = LIF(shape=(C_output,), bias_mant=proc.b_lif3.init, vth=1,
                        dv=1, du=2)
        self.sparse3 = Sparse(weights=proc.w_sparse3.init)
        self.lif4 = LIF(shape=(C_output,), bias_mant=proc.b_lif4.init, vth=1,
                        dv=1, du=2)

        proc.spikes_in.connect(self.dense0.s_in)
        self.dense0.a_out.connect(self.lif1.a_in)
        self.lif1.s_out.connect(self.sparse1.s_in)
        self.sparse1.a_out.connect(self.lif2.a_in)
        self.lif2.s_out.connect(self.dense2.s_in)
        self.dense2.a_out.connect(self.lif3.a_in)
        self.lif3.s_out.connect(self.sparse3.s_in)
        self.sparse3.a_out.connect(self.lif4.a_in)
        self.lif4.s_out.connect(proc.spikes_out)
        
        # Create aliases of SubProcess variables
        proc.lif1_u.alias(self.lif1.u)
        proc.lif1_v.alias(self.lif1.v)
        proc.lif2_u.alias(self.lif2.u)
        proc.lif2_v.alias(self.lif2.v)
        proc.lif3_u.alias(self.lif3.u)
        proc.lif3_v.alias(self.lif3.v)
        proc.lif4_u.alias(self.lif4.u)
        proc.lif4_v.alias(self.lif4.v)

# class SpikingMLP_bug(AbstractProcess):

#     def __init__(self, **kwargs):
#         super().__init__()

#         shape = kwargs.get("shape")
#         (C_input, C_hidden, C_output) = shape
#         self.shape = shape
#         fc1_linear_weight = kwargs.get("fc1_linear_weight")
#         fc2_linear_weight = kwargs.get("fc2_linear_weight")


#         self.spikes_in = InPort(shape=(C_input,))
#         self.spikes_out = OutPort(shape=(C_output,))

#         self.fc1_w = Var(shape=fc1_linear_weight.shape, init=fc1_linear_weight)
#         self.fc2_w = Var(shape=fc2_linear_weight.shape, init=fc2_linear_weight)
#         w_identity_1 = np.eye(C_hidden).astype(np.int32)
#         self.fc1_sp_w = Var(shape=w_identity_1.shape, init=w_identity_1)
#         w_identity_2 = np.eye(C_output).astype(np.int32)
#         self.fc2_sp_w = Var(shape=w_identity_2.shape, init=w_identity_2)

#         self.lif_fc1_u = Var(shape=(C_hidden,), init=0)
#         self.lif_fc1_v = Var(shape=(C_hidden,), init=0)
#         self.bn_fc1_u = Var(shape=(C_hidden,), init=0)
#         self.bn_fc1_v = Var(shape=(C_hidden,), init=0)
#         self.lif_fc2_u = Var(shape=(C_output,), init=0)
#         self.lif_fc2_v = Var(shape=(C_output,), init=0)
#         self.bn_fc2_u = Var(shape=(C_output,), init=0)
#         self.bn_fc2_v = Var(shape=(C_output,), init=0)


# @implements(proc=SpikingMLP_bug, protocol=LoihiProtocol)
# @requires(CPU)
# class PySpikingMLP_bugModel(AbstractSubProcessModel):
#     def __init__(self, proc):

#         (C_input, C_hidden, C_output) = proc.shape

#         self.fc1_linear = Dense(weights=proc.fc1_w.init)
#         self.fc1_bn = LIF(shape=(C_hidden,), bias_mant=0, vth=10, du=1, dv=2)
#         self.fc1_sp = Sparse(weights=proc.fc1_sp_w.init)
#         self.fc1_lif = LIF(shape=(C_hidden,), bias_mant=0, vth=10, du=1, dv=2)

#         self.fc2_linear = Dense(weights=proc.fc2_w.init)
#         self.fc2_bn = LIF(shape=(C_output,), bias_mant=0, vth=10, du=1, dv=2)
#         self.fc2_sp = Sparse(weights=proc.fc2_sp_w.init)
#         self.fc2_lif = LIF(shape=(C_output,), bias_mant=0, vth=10, du=1, dv=2)

#         proc.spikes_in.connect(self.fc1_linear.s_in)
#         self.fc1_linear.a_out.connect(self.fc1_bn.a_in)
#         self.fc1_bn.s_out.connect(self.fc1_sp.s_in)
#         self.fc1_sp.a_out.connect(self.fc1_lif.a_in)
#         self.fc1_lif.s_out.connect(self.fc2_linear.s_in)
#         self.fc2_linear.a_out.connect(self.fc2_bn.a_in)
#         self.fc2_bn.s_out.connect(self.fc2_sp.s_in)
#         self.fc2_sp.a_out.connect(self.fc2_lif.a_in)
#         self.fc2_lif.s_out.connect(proc.spikes_out)
        

#         proc.lif_fc1_u.alias(self.fc1_lif.u)
#         proc.lif_fc1_v.alias(self.fc1_lif.v)
#         proc.lif_fc2_u.alias(self.fc2_lif.u)
#         proc.lif_fc2_v.alias(self.fc2_lif.v)
#         proc.bn_fc1_u.alias(self.fc1_bn.u)
#         proc.bn_fc1_v.alias(self.fc1_bn.v)
#         proc.bn_fc2_u.alias(self.fc2_bn.u)
#         proc.bn_fc2_v.alias(self.fc2_bn.v)

class OutputReceiver(AbstractProcess):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.mat_in = InPort(shape=(shape))
        self.mat_result = Var(shape=(shape), init=0)

@implements(proc=OutputReceiver, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputReceiverModel(PyLoihiProcessModel):
    mat_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    mat_result: np.ndarray = LavaPyType(np.ndarray, bool)

    def run_spk(self):
        mat_in = self.mat_in.recv()
        self.mat_result[:] = mat_in


def test_MLP_block():
    shape = (5, 6, 7) # (TB, N, C_input, C_hidden, C_output)
    (C_input, C_hidden, C_output) = shape
    # generate random weights and bias
    fc1_linear_weight = np.random.rand(C_hidden, C_input).astype(np.int32)
    fc2_linear_weight = np.random.rand(C_output, C_hidden).astype(np.int32)


    MLP_block = SpikingMLPLayer(shape=(C_input, C_output), fc_linear_weight=np.random.rand(C_output, C_input).astype(np.int32), fc_bias=np.random.rand(C_output).astype(np.int32))

    input_process = SpikeGenerator(shape=(C_input,), spike_prob=50)
    # output_process = OutputReceiver(shape=(C_output,))


    input_process.s_out.connect(MLP_block.spikes_in)
    # MLP_block.spikes_out.connect(output_process.mat_in)

    rcfg = Loihi2HwCfg(select_sub_proc_model=True)


    profiler = Profiler.init(rcfg)
    profiler.execution_time_probe(num_steps=10)
    profiler.energy_probe(num_steps=10)
    profiler.activity_probe()
    # Run the entire network of Processes.
    MLP_block.run(condition=RunSteps(num_steps=10), run_cfg=rcfg)
    
    MLP_block.stop()
    profiler.plot_activity(file='activity_test.png')
    print(f"Total execution time: {np.round(np.sum(profiler.execution_time), 6)} s")
    print(profiler.execution_time[:20])
    print(f"Total power: {np.round(profiler.power, 6)} W")
    print(f"Total energy: {np.round(profiler.energy, 6)} J")
    print(f"Static energy: {np.round(profiler.static_energy, 6)} J")  





if __name__ == "__main__":
    test_MLP_block()
    # dense = Sparse(weights=np.eye(10))
    # lif = LIF(shape=(10, ), vth=1)

    # dense.a_out.connect(lif.a_in)
    # rcfg = Loihi2HwCfg(select_sub_proc_model=False)
    # dense.run(condition=RunSteps(num_steps=10), run_cfg=rcfg)
    # dense.stop()
    # print('finish')