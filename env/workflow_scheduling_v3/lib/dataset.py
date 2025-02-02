
import inspect
import os
import sys
import numpy as np
from env.workflow_scheduling_v3.lib.buildDAGfromXML import buildGraph
from env.workflow_scheduling_v3.lib.get_DAGlongestPath import get_longestPath_nodeWeighted

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

dataset_30 = ['CyberShake_30']  # test instance 1
dataset_50 = ['CyberShake_50', 'Montage_50', 'Inspiral_50', 'Sipht_60']  # test instance 2
dataset_100 = ['CyberShake_100', 'Montage_100', 'Inspiral_100', 'Sipht_100']  # test instance 3
dataset_1000 = ['CyberShake_1000', 'Montage_1000', 'Inspiral_1000', 'Sipht_1000']  # test instance 4

dataset_dict = {'S': dataset_30, 'M': dataset_50, 'L': dataset_100, 'XL': dataset_1000}


class dataset:
    def __init__(self, arg):
        if arg not in dataset_dict:
            raise NotImplementedError
        self.wset = []
        self.wsetTotProcessTime = []
        for i, j in zip(['CyberShake'], dataset_dict[arg]):
            dag, wsetProcessTime = buildGraph(f'{i}', parentdir + f'/workflow_scheduling_v3/dax/{j}.xml')

            # Ya added: print the detailed info of dag
            import networkx as nx
            # adj_matrix = nx.adjacency_matrix(dag).todense()
            # print(adj_matrix)
            # for node, data in dag.nodes(data=True):
            #     print(f"Node {node}: {data}")

            self.wset.append(dag)
            self.wsetTotProcessTime.append(wsetProcessTime)

        # the maximum processing time of a DAG from entrance to exit
        self.wsetSlowestT = []
        for app in self.wset:
            self.wsetSlowestT.append(get_longestPath_nodeWeighted(app))

        self.wsetBeta = []
        for app in self.wset:
            self.wsetBeta.append(2)

        self.vmVCPU = [2, 4, 8, 16, 32, 48]  # EC2 m5
        # self.vmVCPU = [2]  # EC2 m5

        self.request = np.array([1]) * 0.01  # Poisson distribution: the default is 0.01, lets test 10.0 1.0 and 0.1

        # Base VM cost per CPU per region
        # Min CPU's provided from services is usually 2, so we define the base fee as this divide by 2 for 1 CPU
        # Assumption: extra costs only takes the cheapest, smallest VM into account.
        # Keys are region_ids
        self.vm_basefee = {
            0: 0.048,  # East, USA, N.Virginia
            1: 0.06,   # Southeast, Australia, Sydney
            2: 0.0555  # West, Europe, London
        }

        # Inter-region communication delays
        self.region_map = {
            0: "us-east-1",
            1: "ap-southeast-2",
            2: "eu-west-2"
        }

        # Bandwidth values for each vCPU VM Type in Gigbits per second (Gbps)
        self.bandwidth_map = {
            2: 8,  # Assume these are 8 as documentation says "Up to 10 Gbps"
            4: 8,
            8: 8,
            16: 8,
            32: 10,
            48: 12
        }

        # TODO: Incorporate these into EXECUTION_TIME calculation
        # Data Communication transmittion time - amount of data + latency
        # amount of data transferred between dependent tasks
        # D / bandwidth (bandwitdh for intercontinental connections)

        # Data transfer, Latency. Cross check with the simulator that the total cost/time accounts for this
        # Can do this by hand, and emulating the code.

        # Inter-region communication delays in milliseconds (ms)
        self.latency_map = {
            0: {  # From N. Virginia
                1: 197,  # Latency to Sydney
                2: 75  # Latency to London
            },
            1: {  # From Sydney
                0: 197,  # Latency to N. Virginia
                2: 264  # Latency to London
            },
            2: {  # From London
                0: 75,  # Latency to N. Virginia
                1: 264  # Latency to Sydney
            }
        }

        # Inter-region data transfer costs per GB
        self.data_transfer_cost_map = {
            0: 0.02,
            1: 0.098,
            2: 0.02
        }

