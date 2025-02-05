
"""
    1. Changed reset() function, added argument: ep_num
    2. added: state_dict_list["removeVM"] = self.vm_queues_id.index(self.VMtobeRemove)
        if self.VMtobeRemove in self.vm_queues_id else None
"""

from env.workflow_scheduling_v3.lib.cloud_env_maxPktNum import cloud_simulator
import numpy as np
import env.workflow_scheduling_v3.lib.dataset as dataset


class WFEnv(cloud_simulator):
    def __init__(self, name, args, testMatrix):

        # Create a train_dataset under a specific seed
        wf_types = len(dataset.dataset_dict[args["wf_size"]])
        if args["generateWay"] == 'fixed':
            train_dataset = np.random.randint(0, wf_types, (1, args['evalNum'], args['wf_num']))
            train_dataset = np.array([list(train_dataset[0]) for _ in range(args['dataGen']+1)])
            train_dataset = train_dataset.astype(np.int64)
        else:   # by rotation
            train_dataset = np.random.randint(0, wf_types, (args['dataGen']+1, args['evalNum'], args['wf_num']))
            train_dataset = train_dataset.astype(np.int64)
        # valid_dataset = np.random.randint(0, wf_types, (1, args['validNum'], args['wf_num'])).astype(np.int64)

        # Ya added: load the fixed test_dataset
        test_dataset = testMatrix
        self.testMatrix = testMatrix

        # Setup
        config = {"traffic pattern": args['traffic_pattern'], "seed": args['seed'], "gamma": args['gamma'],
                  "envid": 0, "wf_size": args["wf_size"], "wf_num": args["wf_num"],
                  "trainSet": train_dataset, 'testSet': test_dataset,
                  "distributed_cloud_enabled": args["distributed_cloud_enabled"]}
        super(WFEnv, self).__init__(config)
        self.name = name

    def reset(self, seed=None, ep_num=None, train_or_test=None):
        super(WFEnv, self).reset(seed, ep_num, train_or_test)
        self.step_curr = self.numTimestep
        state_dict_list = {}
        state_dict = {}
        s = self.state_info_construct()
        state_dict["state"] = np.array(s)
        state_dict_list["0"] = state_dict
        return state_dict_list

    def step(self, action):
        r, usr_respTime, usr_received_appNum, usr_sent_pktNum, d = super(WFEnv, self).step(action["0"])
        state_dict_list = {}
        state_dict = {}
        s = self.state_info_construct()
        info = [usr_respTime, usr_received_appNum, usr_sent_pktNum]

        state_dict["state"] = np.array(s)
        state_dict["reward"] = r
        state_dict["done"] = d
        state_dict["info"] = info
        state_dict_list["0"] = state_dict

        if hasattr(self, "VMtobeRemove"):
            # get the vm index
            state_dict_list["removeVM"] = self.vm_queues_id.index(self.VMtobeRemove) if self.VMtobeRemove in self.vm_queues_id else None

        return state_dict_list, r, d, info

    def get_agent_ids(self):
        return ["0"]

    def close(self):
        super(WFEnv, self).close()

