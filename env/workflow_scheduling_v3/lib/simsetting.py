import inspect
import os
import re
import sys
import logging
import numpy as np

from main import debug_mode

logging.basicConfig(
    level=logging.DEBUG if debug_mode else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.workflow_scheduling_v3.lib.dataset import dataset

traffic_density = {"CONSTANT": 1,
                   "LINEAR_INCREASE": [0.1, 0.00025],  # [base_density, increased rate]
                   "LINEAR_DECREASE": [1, -0.00025],
                   "PERIODIC": {0: 0.65, 1: 0.55, 2: 0.35, 3: 0.25, 4: 0.2, 5: 0.16, 6: 0.16, 7: 0.2, 8: 0.4, 9: 0.55,
                                10: 0.65, 11: 0.75, 12: 0.79, 13: 0.79, 14: 0.85, 15: 0.86, 16: 0.85, 17: 0.83, 18: 0.8,
                                19: 0.79, 20: 0.76, 21: 0.76, 22: 0.69, 23: 0.6}}

traffic_type = ["CONSTANT", "LINEAR_INCREASE", "LINEAR_DECREASE", "PERIODIC"]
traffic_dist = ["EVEN", "UNEVEN"]


class Setting(object):
    state_info_sample_period = 50  # state will be recoreded every 50 seconds
    dataformat = "pickle"
    ac_ob_info_required = False
    epsilon = 0  # 0.1  # used in RL for exploration, epsilon greedy
    is_wf_trace_record = False
    save_nn_iteration_frequency = 1  # every 20 timesteps

    def __init__(self, args):
        self.traf_type = args["traffic pattern"]
        self.traf_dist = "EVEN"
        self.seed = args["seed"]
        if "REINFORCE learning rate" in args.keys():
            self.REINFORCE_learn_rate = args["REINFORCE learning rate"]
        if "hist_len" in args.keys():
            self.history_len = args["hist_len"]
        else:
            self.history_len = 2  # the number of history data (response time & utilization) for each variable will be used
        self.timeStep = 1800
        self.respTime_update_interval = 0.5  # (sec) the time interval used in averaging the response time
        self.util_update_interval = 0.5
        self.arrival_rate_update_interval = self.timeStep
        self.warmupPeriod = 30  # unit: second
        self.envid = args["envid"]
        self.gamma = args["gamma"]
        self.pkt_trace_sample_freq = 10
        self.VMpayInterval = 60 * 60

        # For DDMWS, add option to run simulation in distributed setting and new penalties
        self.distributed_cloud_enabled = args['distributed_cloud_enabled']
        self.dataScalingFactor = args["data_scaling_factor"]  # Between 0.2 - 1.0
        self.latencyPenaltyFactor = args["latency_penalty_factor"]  # Between 0.2 - 1.0
        self.regionMismatchPenaltyFactor = args["region_mismatch_penalty_factor"]  # Experiment with values (0.1 - 1.0)

        self.dataset = dataset(args["wf_size"], self.distributed_cloud_enabled, self.dataScalingFactor)

        # setting for total workflow number
        self.WorkflowNum = args["wf_num"]
        self._init(self.envid)
        # used for gd scheduling inherit from cpp
        self.dlt = 1.0  # self.dlt * avg_resp_time / util
        self.mu = [100.0] * 5
        self.beta = 0.1  # self.beta * synchronization_cost

    def _init(self, num):
        if num == 0:
            # [0] for default US single_region_id
            # [0, 1, 2] for distributed regions in DDMWS that maps to a distinct region
            # Extend/change the number and what types of regions used in self.region_map in dataset.py
            region_ids = [0, 1, 2] if self.distributed_cloud_enabled else [0]

            latency_matrix = np.array([region_ids])
            latency = np.multiply(latency_matrix, 0.5)  # TODO: Incorporate real latency between the regions here
            logger.debug(f"Latency: {latency}")
            self.candidate = region_ids
            self.dcNum = len(self.candidate)   # default: 3
            self.usrNum = latency.shape[0]  # default: 1
            self.candidate.sort()
            self.usr2dc = latency[:, self.candidate]
            logger.debug(f"usrNum Latency: {self.usrNum}")

        else:
            assert (num == 0), "Please set envid to 0!"

        self.wrfNum = len(self.dataset.wset)  # the number of workflow types
        self.arrival_rate = {}  # {usr1: {app1: arrivalRate, app2: arrivalRate}} --> {usr1: {app1: 0.1, app2: 0.1}}
        for i in range(self.usrNum):
            self.arrival_rate[i] = {}
            for a in range(len(self.dataset.wset)):
                self.arrival_rate[i][a] = self.dataset.request[i]
        self.dueTimeCoef = np.ones((self.usrNum, self.wrfNum)) / max(self.dataset.vmVCPU) * self.gamma  # coefficient used to get different due time for each app from each user
        self.totWrfNum = self.WorkflowNum

    def get_individual_arrival_rate(self, time, usrcenter, app):
        if self.traf_type == "CONSTANT":
            den = traffic_density[self.traf_type]  # default:1
        else:
            if re.match(r"^LINEAR.", self.traf_type):
                den = traffic_density[self.traf_type][0] + traffic_density[self.traf_type][-1] * time
            elif self.traf_type == "PERIODIC":
                hr = int(time / 75) % 24  # we consider two periods in one hour
                den = traffic_density[self.traf_type][hr]
            else:
                print("cannot get the arrival rate!!!!!!!!")
                den = None
        return den * self.arrival_rate[usrcenter][app]  # default:0.1
