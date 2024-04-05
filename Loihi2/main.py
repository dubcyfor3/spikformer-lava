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

from lava.proc import embedded_io as eio
from MLP import SpikingMLPLayer, SpikeGenerator

# consider B = 1 first
# input: spikes (B, N, C) 

# SSA: B * N * 3 spiking MLP layer for q k v
# SSA: an adapter aggregate B * N * 3 * C spike tensor and generate B * N * C spike tensor
# SSA:

# output: spikes (B, N, C)

class SpikeGeneratorqkv8(AbstractProcess):
    def __init__(self, channle_dim, spike_prob):
        super().__init__()
        self.spike_prob = Var(shape=(1,), init=spike_prob)
        self.s_out_q_0 = OutPort(shape=(channle_dim,))
        self.s_out_q_1 = OutPort(shape=(channle_dim,))
        self.s_out_q_2 = OutPort(shape=(channle_dim,))
        self.s_out_q_3 = OutPort(shape=(channle_dim,))
        self.s_out_q_4 = OutPort(shape=(channle_dim,))
        self.s_out_q_5 = OutPort(shape=(channle_dim,))
        self.s_out_q_6 = OutPort(shape=(channle_dim,))
        self.s_out_q_7 = OutPort(shape=(channle_dim,))
        self.s_out_k_0 = OutPort(shape=(channle_dim,))
        self.s_out_k_1 = OutPort(shape=(channle_dim,))
        self.s_out_k_2 = OutPort(shape=(channle_dim,))
        self.s_out_k_3 = OutPort(shape=(channle_dim,))
        self.s_out_k_4 = OutPort(shape=(channle_dim,))
        self.s_out_k_5 = OutPort(shape=(channle_dim,))
        self.s_out_k_6 = OutPort(shape=(channle_dim,))
        self.s_out_k_7 = OutPort(shape=(channle_dim,))
        self.s_out_v_0 = OutPort(shape=(channle_dim,))
        self.s_out_v_1 = OutPort(shape=(channle_dim,))
        self.s_out_v_2 = OutPort(shape=(channle_dim,))
        self.s_out_v_3 = OutPort(shape=(channle_dim,))
        self.s_out_v_4 = OutPort(shape=(channle_dim,))
        self.s_out_v_5 = OutPort(shape=(channle_dim,))
        self.s_out_v_6 = OutPort(shape=(channle_dim,))
        self.s_out_v_7 = OutPort(shape=(channle_dim,))

@implements(proc=SpikeGeneratorqkv8, protocol=LoihiProtocol)
@requires(LMT)
class CSpikeGeneratorqkv8Model(CLoihiProcessModel):
    spike_prob: Var = LavaCType(cls=int, d_type=LavaCDataType.INT32)
    s_out_q_0: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_q_1: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_q_2: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_q_3: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_q_4: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_q_5: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_q_6: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_q_7: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_k_0: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_k_1: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_k_2: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_k_3: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_k_4: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_k_5: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_k_6: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_k_7: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_v_0: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_v_1: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_v_2: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_v_3: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_v_4: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_v_5: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_v_6: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    s_out_v_7: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    
    @property
    def source_file_name(self):
        return "spike_generator_qkv8.c"
    

class SpikeAdapter8to1(AbstractProcess):
    def __init__(self, channel_dim):
        super().__init__()
        self.s_in_0 = InPort(shape=(channel_dim,))
        self.s_in_1 = InPort(shape=(channel_dim,))
        self.s_in_2 = InPort(shape=(channel_dim,))
        self.s_in_3 = InPort(shape=(channel_dim,))
        self.s_in_4 = InPort(shape=(channel_dim,))
        self.s_in_5 = InPort(shape=(channel_dim,))
        self.s_in_6 = InPort(shape=(channel_dim,))
        self.s_in_7 = InPort(shape=(channel_dim,))
        self.s_out = OutPort(shape=(8 * channel_dim, ))

@implements(proc=SpikeAdapter8to1, protocol=LoihiProtocol)
@requires(LMT)
class CSpikeAdapter8to1Model(CLoihiProcessModel):
    s_in_0: CInPort = LavaCType(cls=CInPort, d_type=LavaCDataType.INT32)
    s_in_1: CInPort = LavaCType(cls=CInPort, d_type=LavaCDataType.INT32)
    s_in_2: CInPort = LavaCType(cls=CInPort, d_type=LavaCDataType.INT32)
    s_in_3: CInPort = LavaCType(cls=CInPort, d_type=LavaCDataType.INT32)
    s_in_4: CInPort = LavaCType(cls=CInPort, d_type=LavaCDataType.INT32)
    s_in_5: CInPort = LavaCType(cls=CInPort, d_type=LavaCDataType.INT32)
    s_in_6: CInPort = LavaCType(cls=CInPort, d_type=LavaCDataType.INT32)
    s_in_7: CInPort = LavaCType(cls=CInPort, d_type=LavaCDataType.INT32)
    s_out: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    
    @property
    def source_file_name(self):
        return "spike_adapter_8to1.c"

class SpikeAdapter1to8(AbstractProcess):
    def __init__(self, channel_dim):
        super().__init__()
        self.s_in = InPort(shape=(8 * channel_dim, ))
        self.s_out_0 = OutPort(shape=(channel_dim,))
        self.s_out_1 = OutPort(shape=(channel_dim,))
        self.s_out_2 = OutPort(shape=(channel_dim,))
        self.s_out_3 = OutPort(shape=(channel_dim,))
        self.s_out_4 = OutPort(shape=(channel_dim,))
        self.s_out_5 = OutPort(shape=(channel_dim,))
        self.s_out_6 = OutPort(shape=(channel_dim,))
        self.s_out_7 = OutPort(shape=(channel_dim,))

@implements(proc=SpikeAdapter1to8, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeAdapter1to8Model(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)
    s_out_0: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)
    s_out_1: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)
    s_out_2: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)
    s_out_3: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)
    s_out_4: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)
    s_out_5: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)
    s_out_6: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)
    s_out_7: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)

    def run_spk(self):
        data_in = self.s_in.recv()
        length = data_in.shape[0] // 8
        cur_pos = 0
        self.s_out_0.send(data_in[cur_pos:cur_pos + length])
        cur_pos += length
        self.s_out_1.send(data_in[cur_pos:cur_pos + length])
        cur_pos += length
        self.s_out_2.send(data_in[cur_pos:cur_pos + length])
        cur_pos += length
        self.s_out_3.send(data_in[cur_pos:cur_pos + length])
        cur_pos += length
        self.s_out_4.send(data_in[cur_pos:cur_pos + length])
        cur_pos += length
        self.s_out_5.send(data_in[cur_pos:cur_pos + length])
        cur_pos += length
        self.s_out_6.send(data_in[cur_pos:cur_pos + length])
        cur_pos += length
        self.s_out_7.send(data_in[cur_pos:cur_pos + length])



# @implements(proc=SpikeAdapter1to8, protocol=LoihiProtocol)
# @requires(LMT)
# class CSpikeAdapter1to8Model(CLoihiProcessModel):
#     s_in: CInPort = LavaCType(cls=CInPort, d_type=LavaCDataType.INT32)
#     s_out_0: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
#     s_out_1: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
#     s_out_2: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
#     s_out_3: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
#     s_out_4: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
#     s_out_5: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
#     s_out_6: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
#     s_out_7: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    
#     @property
#     def source_file_name(self):
#         return "spike_adapter_1to8.c"

def test_mini_spikformer():
    batch_size = 1
    sequence_length = 8
    embedding_dimension = 16
    depth = 2

    input_process = SpikeGeneratorqkv8(channle_dim=embedding_dimension, spike_prob=50)

    block_list_SSA = []
    block_list_MLP = []
    for block_idx in range(depth):
        batch_list_SSA = []
        batch_list_MLP = []
        for batch_idx in range(batch_size):
            sequence_list_SSA = []
            sequence_list_MLP = []
            for sequence_idx in range(sequence_length):
                process_pack_SSA = []
                q_linear = SpikingMLPLayer(shape=(embedding_dimension, embedding_dimension), 
                                           fc_linear_weight=np.random.rand(embedding_dimension, embedding_dimension).astype(np.int32),
                                             fc_bias=np.random.rand(embedding_dimension).astype(np.int32),
                )
                k_linear = SpikingMLPLayer(shape=(embedding_dimension, embedding_dimension), 
                                           fc_linear_weight=np.random.rand(embedding_dimension, embedding_dimension).astype(np.int32),
                                             fc_bias=np.random.rand(embedding_dimension).astype(np.int32),
                )
                v_linear = SpikingMLPLayer(shape=(embedding_dimension, embedding_dimension), 
                                           fc_linear_weight=np.random.rand(embedding_dimension, embedding_dimension).astype(np.int32),
                                             fc_bias=np.random.rand(embedding_dimension).astype(np.int32),
                )
                process_pack_SSA.append(q_linear)
                process_pack_SSA.append(k_linear)
                process_pack_SSA.append(v_linear)
                sequence_list_SSA.append(process_pack_SSA)
                process_pack_MLP = []
                sequence_list_MLP.append(process_pack_MLP)
            batch_list_SSA.append(sequence_list_SSA)
            batch_list_MLP.append(sequence_list_MLP)
        block_list_MLP.append(batch_list_MLP)
        block_list_SSA.append(batch_list_SSA)

    input_process.s_out_q_0.connect(block_list_SSA[0][0][0][0].spikes_in)
    input_process.s_out_q_1.connect(block_list_SSA[0][0][1][0].spikes_in)
    input_process.s_out_q_2.connect(block_list_SSA[0][0][2][0].spikes_in)
    input_process.s_out_q_3.connect(block_list_SSA[0][0][3][0].spikes_in)
    input_process.s_out_q_4.connect(block_list_SSA[0][0][4][0].spikes_in)
    input_process.s_out_q_5.connect(block_list_SSA[0][0][5][0].spikes_in)
    input_process.s_out_q_6.connect(block_list_SSA[0][0][6][0].spikes_in)
    input_process.s_out_q_7.connect(block_list_SSA[0][0][7][0].spikes_in)

    input_process.s_out_k_0.connect(block_list_SSA[0][0][0][1].spikes_in)
    input_process.s_out_k_1.connect(block_list_SSA[0][0][1][1].spikes_in)
    input_process.s_out_k_2.connect(block_list_SSA[0][0][2][1].spikes_in)
    input_process.s_out_k_3.connect(block_list_SSA[0][0][3][1].spikes_in)
    input_process.s_out_k_4.connect(block_list_SSA[0][0][4][1].spikes_in)
    input_process.s_out_k_5.connect(block_list_SSA[0][0][5][1].spikes_in)
    input_process.s_out_k_6.connect(block_list_SSA[0][0][6][1].spikes_in)
    input_process.s_out_k_7.connect(block_list_SSA[0][0][7][1].spikes_in)

    input_process.s_out_v_0.connect(block_list_SSA[0][0][0][2].spikes_in)
    input_process.s_out_v_1.connect(block_list_SSA[0][0][1][2].spikes_in)
    input_process.s_out_v_2.connect(block_list_SSA[0][0][2][2].spikes_in)
    input_process.s_out_v_3.connect(block_list_SSA[0][0][3][2].spikes_in)
    input_process.s_out_v_4.connect(block_list_SSA[0][0][4][2].spikes_in)
    input_process.s_out_v_5.connect(block_list_SSA[0][0][5][2].spikes_in)
    input_process.s_out_v_6.connect(block_list_SSA[0][0][6][2].spikes_in)
    input_process.s_out_v_7.connect(block_list_SSA[0][0][7][2].spikes_in)

    run_cfg = Loihi2HwCfg(select_sub_proc_model=True)
    profiler = Profiler.init(run_cfg)
    profiler.execution_time_probe(num_steps=10)
    profiler.energy_probe(num_steps=10)
    profiler.activity_probe()
    # Run the entire network of Processes.
    input_process.run(condition=RunSteps(num_steps=10), run_cfg=run_cfg)
    
    input_process.stop()
    profiler.plot_activity(file='activity_test_spikformer.png')
    print(f"Total execution time: {np.round(np.sum(profiler.execution_time), 6)} s")
    print(profiler.execution_time[:20])
    print(f"Total power: {np.round(profiler.power, 6)} W")
    print(f"Total energy: {np.round(profiler.energy, 6)} J")
    print(f"Static energy: {np.round(profiler.static_energy, 6)} J")  

def test_spikformer():
    batch_size = 1
    sequence_length = 64
    embedding_dimension = 16
    depth = 1

    # input: spikes (B, N, C)
    input_process = SpikeGenerator(shape=(64 * embedding_dimension, ), spike_prob=50)

    input_adapter_l0 = SpikeAdapter1to8(channel_dim=8 * embedding_dimension)
    input_adapter_l1 = []
    py2nx_adapter = []
    
    input_process.s_out.connect(input_adapter_l0.s_in)
    for i in range(8):
        adapter = SpikeAdapter1to8(channel_dim=embedding_dimension)
        input_adapter_l1.append(adapter)

    for i in range(2):
        adapter = eio.spike.PyToNxAdapter(shape=(embedding_dimension, ))
        py2nx_adapter.append(adapter)

    input_adapter_l0.s_out_0.connect(input_adapter_l1[0].s_in)
    input_adapter_l0.s_out_1.connect(input_adapter_l1[1].s_in)
    input_adapter_l0.s_out_2.connect(input_adapter_l1[2].s_in)
    input_adapter_l0.s_out_3.connect(input_adapter_l1[3].s_in)
    input_adapter_l0.s_out_4.connect(input_adapter_l1[4].s_in)
    input_adapter_l0.s_out_5.connect(input_adapter_l1[5].s_in)
    input_adapter_l0.s_out_6.connect(input_adapter_l1[6].s_in)
    input_adapter_l0.s_out_7.connect(input_adapter_l1[7].s_in)

    # test_process = SpikingMLPLayer(shape=(embedding_dimension, embedding_dimension), 
    #                                fc_linear_weight=np.random.rand(embedding_dimension, embedding_dimension).astype(np.int32),
    #                                fc_bias=np.random.rand(embedding_dimension).astype(np.int32),
    # )

    # input_adapter_l0.s_out_0.connect(test_process.spikes_in)
    # decoder block
    block_list_SSA = []
    block_list_MLP = []
    for block_idx in range(depth):
        batch_list_SSA = []
        batch_list_MLP = []
        for batch_idx in range(batch_size):
            sequence_list_SSA = []
            sequence_list_MLP = []
            for sequence_idx in range(sequence_length):
                process_pack_SSA = []
                q_linear = SpikingMLPLayer(shape=(embedding_dimension, embedding_dimension), 
                                           fc_linear_weight=np.random.rand(embedding_dimension, embedding_dimension).astype(np.int32),
                                             fc_bias=np.random.rand(embedding_dimension).astype(np.int32),
                )
                process_pack_SSA.append(q_linear)
                sequence_list_SSA.append(process_pack_SSA)
                process_pack_MLP = []
                sequence_list_MLP.append(process_pack_MLP)
            batch_list_SSA.append(sequence_list_SSA)
            batch_list_MLP.append(sequence_list_MLP)
        block_list_MLP.append(batch_list_MLP)
        block_list_SSA.append(batch_list_SSA)

    input_adapter_l1[0].s_out_0.connect(py2nx_adapter[0].inp)
    input_adapter_l1[1].s_out_1.connect(py2nx_adapter[1].inp)
    py2nx_adapter[0].out.connect(block_list_SSA[0][0][0][0].spikes_in)
    py2nx_adapter[1].out.connect(block_list_SSA[0][0][1][0].spikes_in)
    # for i in range(8):
    #     input_adapter_l1[i].s_out_0.connect(py2nx_adapter[0 + i * 8].inp)
    #     input_adapter_l1[i].s_out_1.connect(py2nx_adapter[1 + i * 8].inp)
    #     input_adapter_l1[i].s_out_2.connect(py2nx_adapter[2 + i * 8].inp)
    #     input_adapter_l1[i].s_out_3.connect(py2nx_adapter[3 + i * 8].inp)
    #     input_adapter_l1[i].s_out_4.connect(py2nx_adapter[4 + i * 8].inp)
    #     input_adapter_l1[i].s_out_5.connect(py2nx_adapter[5 + i * 8].inp)
    #     input_adapter_l1[i].s_out_6.connect(py2nx_adapter[6 + i * 8].inp)
    #     input_adapter_l1[i].s_out_7.connect(py2nx_adapter[7 + i * 8].inp)

    # for i in range(64):
    #     py2nx_adapter[i].out.connect(block_list_SSA[0][0][i][0].spikes_in)



    run_cfg = Loihi2HwCfg(select_sub_proc_model=True, exception_proc_model_map={eio.spike.PyToNxAdapter: eio.spike.PyToNxAdapterModel})
    profiler = Profiler.init(run_cfg)
    profiler.execution_time_probe(num_steps=10)
    profiler.energy_probe(num_steps=10)
    profiler.activity_probe()
    # Run the entire network of Processes.
    input_process.run(condition=RunSteps(num_steps=10), run_cfg=run_cfg)
    
    input_process.stop()
    profiler.plot_activity(file='activity_test_spikformer.png')
    print(f"Total execution time: {np.round(np.sum(profiler.execution_time), 6)} s")
    print(profiler.execution_time[:20])
    print(f"Total power: {np.round(profiler.power, 6)} W")
    print(f"Total energy: {np.round(profiler.energy, 6)} J")
    print(f"Static energy: {np.round(profiler.static_energy, 6)} J")  

if __name__ == '__main__':
    test_mini_spikformer()