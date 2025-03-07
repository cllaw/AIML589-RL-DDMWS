
"""
    1. Changed reset() function, added argument: ep_num
    2. self.indEVALindex = ep_num, individual is evaluated on ep_num-th instance in on dataGen
"""
from eval_rl import debug_mode
import numpy as np
# import pandas as pd
import csv
import math
import os, sys, inspect, random, copy
import gymnasium
import torch
import logging

logging.basicConfig(
    level=logging.DEBUG if debug_mode else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


from env.workflow_scheduling_v3.lib.stats import Stats
from env.workflow_scheduling_v3.lib.poissonSampling import one_sample_poisson
from env.workflow_scheduling_v3.lib.vm import VM
from env.workflow_scheduling_v3.lib.workflow import Workflow
from env.workflow_scheduling_v3.lib.simqueue import SimQueue
from env.workflow_scheduling_v3.lib.simsetting import Setting
from env.workflow_scheduling_v3.lib.cal_rank import calPSD


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
vmidRange = 10000


def ensure_dir_exist(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_csv_header(file, header):
    ensure_dir_exist(file)
    with open(file, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(header)


def write_csv_data(file, data):
    ensure_dir_exist(file)
    with open(file, 'a', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(data)


class cloud_simulator(object):

    def __init__(self, args):
        self.set = Setting(args)
        self.baseHEFT = None

        self.trainSet = args["trainSet"]
        self.testSet = args["testSet"]  # Ya added
        self.train_or_test = None  # Ya added

        self.GENindex = None
        self.indEVALindex = None
        self.TaskRule = None  # input th task selection rule here, if it has

        if self.set.is_wf_trace_record:
            self.df = {}
            __location__ = os.getcwd() + '\Saved_Results'
            self.pkt_trace_file = os.path.join(__location__, r'allocation_trace_%s_seed%s_arr%s_gamma%s.csv' % (args["algo"],  args["seed"], args["arrival rate"], args["gamma"]))
            write_csv_header(self.pkt_trace_file, ['Workflow ID', 'Workflow Pattern', 'Workflow Arrival Time', 'Workflow Finish Time', 'Workflow Deadline', 'Workflow Deadline Penalty',
                                                   'Task Index', 'Task Size', 'Task Execution Time', 'Task Ready Time', 'Task Start Time', 'Task Finish Time',
                                                   'VM ID', 'VM speed', 'Price', 'VM Rent Start Time', 'VM Rent End Time', 'VM Pending Index'])  # 6 + 6 + 6 columns

        # TODO: Make this work dynamically so we can test older models with different numbers of states
        num_states = 10
        self.observation_space = gymnasium.spaces.Box(low=0, high=10000, shape=(num_states + self.set.history_len,))
        # self.observation_space = gymnasiumnasium.spaces.Box(low=0, high=10000, shape=(9 + self.set.history_len,))  # Ya added
        self.action_space = gymnasium.spaces.Discrete(n=100)  # n is a placeholder

    def close(self):
        print("Environment id %s is closed" % self.set.envid)

    def _init(self):
        self.appSubDeadline = {}        # { app: {task: task_sub_deadline} } used as a state feature
        self.usr_queues = []            # [usr1:[workflows, ...], usr2:[workflows, ...]], e.g., user1 stores 30 workflows
        self.vm_queues = []             # [VM1, VM2, ...] each VM is a class
        self.vm_queues_id = []          # the vmid of each VM in self.vm_queues
        self.vm_queues_cpu = []
        self.vm_queues_rentEndTime = []
        self.usrNum = self.set.usrNum   # useless for one cloud
        self.dcNum = self.set.dcNum     # useless for one cloud
        self.wrfNum = self.set.wrfNum
        self.totWrfNum = self.set.totWrfNum
        self.VMtypeNum = len(self.set.dataset.vmVCPU)  # number of VM types
        self.numTimestep = 0            # indicate how many timesteps have been processed
        self.completedWF = 0
        self.VMRemainingTime = {}       # {vmid1:time, vmid2:time}
        self.VMRemainAvaiTime = {}      # reamin available time  = leased time period - vm_total_execute_time
        self.VMrentInfos = {}           # {VMid: [rent start time, rent end time]}
        self.notNormalized_arr_hist = np.zeros((self.usrNum, self.wrfNum, self.set.history_len)) 
        self.VMcost = 0
        self.SLApenalty = 0
        self.region_mismatch_penalty = 0
        self.wrfIndex = 0
        self.usrcurrentTime = np.zeros(self.usrNum)  # Used to record the current moment of the user
        self.remainWrfNum = 0           # Record the number of packets remained in VMs
        self.missDeadlineNum = 0
        self.region_mismatch_count = 0
        self.VMrentHours = 0
        self.VMexecHours = 0  

        # IMPORTANT: used to get the ready task for the next time step
        self.firstvmWrfLeaveTime = []  # Record the current timestamp on each VM
        self.firstusrWrfGenTime = np.zeros(self.usrNum)  # Arrival time of the first inactive workflow in each user's workflow set

        self.uselessAllocation = 0
        self.VMtobeRemove = None

        self.usr_respTime = np.zeros((self.usrNum, self.wrfNum)) 
        self.usr_received_wrfNum = np.zeros((self.usrNum, self.wrfNum)) 
        self.usr_sent_pktNum = np.zeros((self.usrNum, self.dcNum))

        # upload all workflows with their arrival time to the 'self.firstusrWrfGenTime'
        # for i in range(self.usrNum):
        #     self.usr_queues.append(SimQueue())
        #     workflowsIDs = self.trainSet[self.GENindex][self.indEVALindex]
        #     for appID in workflowsIDs:
        #         self.workflow_generator(i, appID)
        #     self.firstusrWrfGenTime[i] = self.usr_queues[i].getFirstPktEnqueueTime()

        # Ya added
        for i in range(self.usrNum):
            if self.train_or_test == "train":
                Data_Set = self.trainSet
            elif self.train_or_test == "test":
                Data_Set = self.testSet
            self.usr_queues.append(SimQueue())
            workflowsIDs = Data_Set[self.GENindex][self.indEVALindex]
            for appID in workflowsIDs:
                self.workflow_generator(i, appID)
            self.firstusrWrfGenTime[i] = self.usr_queues[i].getFirstPktEnqueueTime()

        self.nextUsr, self.nextTimeStep = self.get_nextWrfFromUsr()
        self.PrenextTimeStep = self.nextTimeStep
        self.nextisUsr = True
        self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstPkt()  # obtain the root task of the first workflow in the self.nextUsr
        temp = self.nextWrf.get_allnextTask(self.finishTask)  # Get all real successor tasks of the virtual workflow root task
                
        self.dispatchParallelTaskNum = 0
        self.nextTask = temp[self.dispatchParallelTaskNum]
        if len(temp) > 1:  # the next task has parallel successor tasks
            self.isDequeue = False
            self.isNextTaskParallel = True
        else:
            self.isDequeue = True  # decide whether the nextWrf should be dequeued
            self.isNextTaskParallel = False

        self.stat = Stats(self.set)

    # Generate one workflow at one time
    def workflow_generator(self, usr, appID):
        wrf = self.set.dataset.wset[appID]
        nextArrivalTime = one_sample_poisson(self.set.get_individual_arrival_rate(self.usrcurrentTime[usr], usr, appID),
                                             self.usrcurrentTime[usr])
        self.remainWrfNum += 1
        # print("Workflow:", wrf)

        # add workflow deadline to the workflow
        pkt = Workflow(self.usrcurrentTime[usr], wrf, appID, usr, self.set.dataset.wsetSlowestT[appID], self.set.dueTimeCoef[usr, appID], self.wrfIndex)  # self.set.gamma / max(self.set.dataset.vmVCPU))

        # In DDMWS we use a new method to assign region_ids to tasks in Workflows via their
        #   execution VM's based on some heuristic
        # If this method is not used, each task and their successors are assigned a randon region_id
        logger.debug(f"Tasks in workflow of: {pkt.get_allTask()}")
        for task in pkt.get_allTask():  # Assuming get_all_tasks() returns all tasks in the workflow
            if task not in pkt.processRegion:
                region_id = pkt.get_task_regionId(task)
                pkt.update_taskLocation(task, region_id)  # Update task information on the task level in the workflow

        self.usr_queues[usr].enqueue(pkt, self.usrcurrentTime[usr], None, usr, 0)  # None means that workflow has not started yet
        self.usrcurrentTime[usr] = nextArrivalTime
        self.totWrfNum -= 1
        self.wrfIndex += 1

    def reset(self, seed, ep_num, train_or_test):
        random.seed(self.set.seed)
        np.random.seed(self.set.seed)
        self.GENindex = seed  # GENindex means the seed-th data in dataGen that is predefined in yaml
        self.indEVALindex = ep_num  # individual is evaluated on ep_num-th instance of GENindex_th dataGen
        self.train_or_test = train_or_test
        self._init()

    def input_task_rule(self, rule):
        self.TaskRule = rule

    def generate_vmid(self):
        vmid = np.random.randint(vmidRange, size=1)[0]
        while vmid in self.VMRemainingTime:
            vmid = np.random.randint(vmidRange, size=1)[0]
        return vmid

    def get_nextWrfFromUsr(self):       # Select the User with the smallest timestamp
        usrInd = np.argmin(self.firstusrWrfGenTime)
        firstPktTime = self.firstusrWrfGenTime[usrInd]
        return usrInd, firstPktTime     # Returns the user and arrival time of the minimum arrival time of the workflow in the current User queue.

    def get_nextWrfFromVM(self):        # Select the machine with the smallest timestamp
        if len(self.firstvmWrfLeaveTime) > 0:
            vmInd = np.argmin(self.firstvmWrfLeaveTime)
            firstPktTime = self.firstvmWrfLeaveTime[vmInd]
            return vmInd, firstPktTime  # Returns vm-id and the minimum end time of the current VM
        else:
            return None, math.inf

    def get_nextTimeStep(self):
        self.PrenextUsr, self.PrenextTimeStep = self.nextUsr, self.nextTimeStep
        tempnextloc, tempnextTimeStep = self.get_nextWrfFromUsr()  
        tempnextloc1, tempnextTimeStep1 = self.get_nextWrfFromVM() 
        if tempnextTimeStep > tempnextTimeStep1:  # task ready time > VM minimum time
            self.nextUsr, self.nextTimeStep = tempnextloc1, tempnextTimeStep1  
                                        # The next step is to process the VM and update it to the timestep of the VM.
            self.nextisUsr = False
            self.nextWrf, self.finishTask = self.vm_queues[self.nextUsr].get_firstDequeueTask()  # Only returns time, does not process task
        else:  # tempnextTimeStep <= tempnextTimeStep1
            if tempnextTimeStep == math.inf:   # tempnextTimeStep：when self.usr_queues.queue is []
                self.nextTimeStep = None       # tempnextTimeStep1：when self.firstvmWrfLeaveTime is []
                self.nextUsr = None
                self.nextWrf = None
                self.nextisUsr = True
            else:
                self.nextUsr, self.nextTimeStep = tempnextloc, tempnextTimeStep  # Next step is to process user & Update to user's timeStep
                self.nextisUsr = True    # Activate new Workflow from Usr_queue
                self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstPkt()  # The current first task in the selected user

    def update_VMRemain_infos(self):
        for key in self.VMRemainingTime:
            ind = self.vm_queues_id.index(key)
            self.VMRemainingTime[key] = self.vm_queues_rentEndTime[ind] - self.nextTimeStep
            maxTimeStep = max(self.vm_queues[ind].currentTimeStep, self.nextTimeStep)  # consider idle gap in VM
            self.VMRemainAvaiTime[key] = self.vm_queues_rentEndTime[ind] - maxTimeStep - self.vm_queues[ind].vmQueueTime() 

    def remove_expired_VMs(self):
        removed_keys = []        
        Indexes = np.where(np.array(self.vm_queues_rentEndTime) < self.nextTimeStep + 0.00001)[0]

        if not self.nextisUsr:
            nextvmid = self.vm_queues[self.nextUsr].vmid
            if self.nextUsr in Indexes:
                self.VMtobeRemove = nextvmid  # get the vm-id
            else:
                self.VMtobeRemove = None

        for ind in Indexes:  
            a = self.vm_queues[ind].get_pendingTaskNum()
            if a == 0:
                removed_keys.append(self.vm_queues_id[ind])

        for key in removed_keys:
            del self.VMRemainingTime[key]
            del self.VMRemainAvaiTime[key]
            ind = self.vm_queues_id.index(key)
            del self.vm_queues_id[ind]
            del self.vm_queues_cpu[ind]
            del self.vm_queues_rentEndTime[ind]
            vm = self.vm_queues.pop(ind)
            del self.firstvmWrfLeaveTime[ind]
            del vm              

        if not self.nextisUsr:
            if nextvmid in self.vm_queues_id:
                self.nextUsr = self.vm_queues_id.index(nextvmid)   
            else:
                print('nextvmid is not in self.vm_queues_id')
                self.uselessAllocation += 1
                print('-----> wrong index:', self.uselessAllocation)

    def extend_specific_VM(self, VMindex):
        logger.debug(f'Extending specific VM {VMindex}')
        key = self.vm_queues_id[VMindex]
        maxTimeStep = max(self.vm_queues[VMindex].currentTimeStep, self.nextTimeStep)
        self.VMRemainAvaiTime[key] = self.vm_queues_rentEndTime[VMindex] - maxTimeStep - self.vm_queues[VMindex].vmQueueTime()  # has idle gap
        while self.VMRemainAvaiTime[key] < -0.00001:  # ignore system error
            self.VMRemainAvaiTime[key] += self.set.VMpayInterval
            self.vm_queues[VMindex].update_vmRentEndTime(self.set.VMpayInterval)
            self.vm_queues_rentEndTime[VMindex] = self.vm_queues[VMindex].rentEndTime

            # print("Extend remove VMs", self.vm_queues[VMindex].loc)  #TODO: Remove .loc attribute or rework
            # print("Extend remove VMs", self.vm_queues[VMindex])

            self.update_VMcost(self.vm_queues[VMindex].regionid, self.vm_queues[VMindex].cpu, True)
            self.VMrentInfos[key] = self.VMrentInfos[key][:4] + [self.vm_queues[VMindex].rentEndTime]  # self.VMrentInfos[key][-1]+self.set.dataset.vmPrice[self.vm_queues[VMindex].cpu]]

    def record_a_completed_workflow(self, ddl_penalty):
        if self.set.is_wf_trace_record:        
            Workflow_Infos = [self.nextWrf.appArivalIndex, self.nextWrf.appID,
                              self.nextWrf.generateTime, self.nextTimeStep, self.nextWrf.deadlineTime, ddl_penalty]

            for task in range(len(self.nextWrf.executeTime)):
                Task_Infos = [task, self.nextWrf.app.nodes[task]['processTime'], self.nextWrf.executeTime[task],
                              self.nextWrf.readyTime[task], self.nextWrf.enqueueTime[task], self.nextWrf.dequeueTime[task]]

                VM_Infos = self.VMrentInfos[self.nextWrf.processDC[task]] + [self.nextWrf.pendingIndexOnDC[task]]

                write_csv_data(self.pkt_trace_file, Workflow_Infos + Task_Infos + VM_Infos)

    # Check whether the machine's lease period needs to be extended
    # TODO: Not in use, consider removing?
    def extend_remove_VMs(self):
        # print("Current VM's", self.vm_queues_id)
        expiredVMid = []
        for key in self.VMRemainingTime:
            ind = self.vm_queues_id.index(key)
            self.VMRemainingTime[key] = self.vm_queues[ind].rentEndTime-self.nextTimeStep
            self.VMRemainAvaiTime[key] = self.VMRemainingTime[key] - self.vm_queues[ind].pendingTaskTime

            if self.VMRemainAvaiTime[key] <= 0:
                if self.vm_queues[ind].currentQlen == 0:  # to be removed
                    expiredVMid.append(key) 
                else:
                    while self.VMRemainAvaiTime[key] <= 0:
                        # print(f'Extending VM {key}')
                        self.VMRemainingTime[key] += self.set.VMpayInterval
                        self.VMRemainAvaiTime[key] += self.set.VMpayInterval
                        self.vm_queues[ind].update_vmRentEndTime(self.set.VMpayInterval)

                        # print("Extend remove VMs", self.vm_queues[ind])
                        # print("Extend remove VMs", ind)
                        self.update_VMcost(self.vm_queues[ind].regionid, self.vm_queues[ind].cpu, True)

        if len(expiredVMid) > 0:  # Really remove here
            # print(f'Remove VM {expiredVMid}')
            if not self.nextisUsr:
                nextvmid = self.vm_queues[self.nextUsr].vmid

            for key in expiredVMid:
                del self.VMRemainingTime[key]
                del self.VMRemainAvaiTime[key]
                ind = self.vm_queues_id.index(key)
                del self.vm_queues_id[ind]
                del self.vm_queues_cpu[ind]
                del self.vm_queues_rentEndTime[ind]
                vm = self.vm_queues.pop(ind)
                del self.firstvmWrfLeaveTime[ind]
                del vm        

            # If there is deletion, you need to adjust the index corresponding to self.nextUsr
            if not self.nextisUsr:
                if nextvmid in self.vm_queues_id:
                    self.nextUsr = self.vm_queues_id.index(nextvmid)   
                else:
                    print('wrong')

    # Function prototype is vf_ob, ac_ob, rew, new, _ = env.step(ac)
    def step(self, action):
        # print(f"TEST ALL VMS: {self.vm_queues}")
        logger.debug(f"TEST ALL VMS: {len(self.vm_queues)}")
        logger.debug(f"TEST ALL VMS: {[vm.cpu for vm in self.vm_queues]}")
        logger.debug(f"TEST ALL VMS: {[self.set.dataset.region_map[vm.regionid] for vm in self.vm_queues]}")
        logger.debug(f"TEST ALL VMS: {self.VMRemainingTime}")
        logger.debug(f"TEST ALL VMS: {self.VMrentInfos}")
        # print(f"TEST ALL VMS: {self.VMRemainAvaiTime}")  # Whats the difference between these?

        # decode the action: the index of the vm which ranges from 0 to len(self.vm_queues)+self.vmtypeNum*self.dcNum
        # ---1) Map & Dispatch
        # maping the action to the vm_id in current VM queue
        diff = action - len(self.vm_queues)
        region_count = len(self.set.dataset.region_map)  # Number of regions available
        logger.debug(f"Scheduling policy DIFF: {diff}")

        # Negative differences spin up new VM's
        if diff > -1:  # a new VM is deployed
            vmid = self.generate_vmid()  # Randomly generate a set of numbers to name
            # print(f"New VM {vmid} deployed.")
            dcid = 0    # This is used to distinguish the number of different resource types for each DC and can be omitted
            vm_type_index = diff % self.VMtypeNum
            vm_cpu = self.set.dataset.vmVCPU[vm_type_index]  # int representing how many cpus on a particular vm

            # TODO: Try and train a baseline and spatial model with this settings
            #  Assumption, give full control to policy for seelcting VM
            #  Assumption, in method 2 allowing new VMs being made to always be the correct region limits exploration
            # Region Selection Rule 1
            # Assign region ID based on diff and the number of regions if distributed cloud setting is enabled
            # region_id = diff % region_count if self.set.distributed_cloud_enabled else 0

            # Region Selection Rule 2
            # Match the region ID of the task being executed
            if self.nextTask in self.nextWrf.processRegion:
                region_id = self.nextWrf.processRegion[self.nextTask]  # Match VM region to task region
            else:
                region_id = 0  # Default to region 0 if no region is set

            selectedVM = VM(vmid, vm_cpu, dcid, dcid, self.nextTimeStep, self.TaskRule, region_id)
            logger.debug(f"Task: {self.nextTask}")
            logger.debug(f"Whole Region: {self.nextWrf.processRegion}")
            logger.debug(f"New VM deployed in region: {self.set.dataset.region_map[selectedVM.regionid]} for next task {self.nextTask} of region ID {region_id}")

            self.vm_queues.append(selectedVM)
            self.firstvmWrfLeaveTime.append(selectedVM.get_firstTaskDequeueTime())  # new VM is math.inf
            self.vm_queues_id.append(vmid)
            self.vm_queues_cpu.append(vm_cpu)
            self.update_VMcost(selectedVM.regionid, vm_cpu, True)
            selectedVMind = -1
            self.VMRemainingTime[vmid] = self.set.VMpayInterval  # initialize the remaining time for the new VM
            self.VMRemainAvaiTime[vmid] = self.set.VMpayInterval            
            self.vm_queues[selectedVMind].update_vmRentEndTime(self.set.VMpayInterval)
            self.vm_queues_rentEndTime.append(self.vm_queues[selectedVMind].rentEndTime)

            self.VMrentInfos[vmid] = [vmid, vm_cpu, self.set.dataset.vm_basefee[selectedVM.regionid] * vm_cpu,
                                      self.nextTimeStep, self.vm_queues[selectedVMind].rentEndTime, selectedVM.regionid]
        # diff <= -1 - the action refers to an existing VM in self.vm_queues
        # The task will be scheduled on one of the currently rented VMs
        #   using the action index as the VM in vm_queues[selectedVMind]
        else:
            selectedVMind = action
            logger.debug(f"Not creating a new VM, using an existing one | VM ID {selectedVMind}, region: {self.vm_queues[selectedVMind].regionid} Task {self.nextTask}")
        reward = 0
        self.PrenextUsr, self.PrenextTimeStep, self.PrenextTask = self.nextUsr, self.nextTimeStep, self.nextTask 

        # dispatch nextWrf to selectedVM and update the wrfLeaveTime on selectedVM 
        parentTasks = self.nextWrf.get_allpreviousTask(self.PrenextTask)
        # print(f"Remaining Parent tasks: {parentTasks}")
        if len(parentTasks) == len(self.nextWrf.completeTaskSet(parentTasks)):  # all its predecessor tasks have been done, just double-check

            # DDMWS: task_enqueue adds Data Transfer Latency if next selected VM is not in the same region
            #   as the previous workflow
            # TODO: Breakdown a workflow into smaller tasks and validate data transfer costs are correct
            vm_region_id = self.vm_queues[selectedVMind].regionid
            task_region = self.nextWrf.get_taskRegion(self.PrenextTask)
            successor_tasks = self.nextWrf.get_allnextTask(self.PrenextTask)

            self.region_mismatch_penalty += self.calculate_region_mismatch_penalty(vm_region_id, task_region, self.PrenextTask, successor_tasks)

            processTime, data_transfer_cost, task_communication_delay = self.vm_queues[selectedVMind].task_enqueue(
                self.PrenextTask,
                self.PrenextTimeStep,
                self.nextWrf,
                self.set.dataset.bandwidth_map,
                self.set.dataset.latency_map,
                self.set.dataset.region_map,
                self.set.dataset.data_transfer_cost_map
            )

            # Chuan added for DDMWS
            # Calculate the latency penalty for inter-region communication.
            total_latency_penalty = task_communication_delay * self.set.latencyPenaltyFactor
            self.SLApenalty += total_latency_penalty

            # print(f"Process Time: {processTime}")
            self.update_VMcost_with_data_transfer_cost(data_transfer_cost)
            self.VMexecHours += processTime/3600
            self.firstvmWrfLeaveTime[selectedVMind] = self.vm_queues[selectedVMind].get_firstTaskDequeueTime()  # return currunt timestap on this machine
            self.extend_specific_VM(selectedVMind)

        # ---2) Dequeue nextTask
        if self.isDequeue:      # True: the nextTask should be popped out 
            if self.nextisUsr:  # True: the nextTask to be deployed comes from the user queue
                self.nextWrf.update_dequeueTime(self.PrenextTimeStep, self.finishTask)
                _, _ = self.usr_queues[self.PrenextUsr].dequeue()  # Here is the actual pop-up of the root task
                self.firstusrWrfGenTime[self.PrenextUsr] = self.usr_queues[self.PrenextUsr].getFirstPktEnqueueTime() 
                                                            # Updated with the arrival time of the next workflow
                self.usr_sent_pktNum[self.PrenextUsr][self.vm_queues[selectedVMind].get_relativeVMloc()] += 1
                self.stat.add_app_arrival_rate(self.PrenextUsr, self.nextWrf.get_appID(), self.nextWrf.get_generateTime()) # record
            else:               # the nextTask to be deployed comes from the vm queues
                _, _ = self.vm_queues[self.PrenextUsr].task_dequeue()  # Here nextTask actually starts to run
                self.firstvmWrfLeaveTime[self.PrenextUsr] = self.vm_queues[self.PrenextUsr].get_firstTaskDequeueTime()
                                                            # Update the current TimeStamp in this machine

        # ---3) Update: self.nextTask, and maybe # self.nextWrf, self.finishTask, self.nextUsr, self.nextTimeStep, self.nextisUsr
        temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)   # all successor tasks of the current self.finishTask
                                    # and one successor task has already enqueued

        if len(temp_Children_finishTask) > 0:
            self.dispatchParallelTaskNum += 1

        while True:  # Handle already scheduled tasks

            # self.nextWrf is completed
            while len(temp_Children_finishTask) == 0:  # self.finishTask is the final task of self.nextWrf
                
                if self.nextisUsr:  # for double-check: Default is False
                    # Because it corresponds to self.finishTask, if temp==0, it means it cannot be entry tasks
                    print('self.nextisUsr maybe wrong')
                _, app = self.vm_queues[self.nextUsr].task_dequeue()  
                self.firstvmWrfLeaveTime[self.nextUsr] = self.vm_queues[self.nextUsr].get_firstTaskDequeueTime() 
                        # If there is no task on the VM, math.inf will be returned
                if self.nextWrf.is_completeTaskSet(self.nextWrf.get_allTask()):     # self.nextWrf has been completed
                    respTime = self.nextTimeStep - self.nextWrf.get_generateTime()
                    self.usr_respTime[app.get_originDC()][app.get_appID()] += respTime
                    self.usr_received_wrfNum[app.get_originDC()][app.get_appID()] += 1                    
                    self.completedWF += 1
                    self.remainWrfNum -= 1

                    ddl_penalty = self.calculate_penalty(app, respTime)
                    self.SLApenalty += ddl_penalty
                    self.record_a_completed_workflow(ddl_penalty)
                    del app, self.nextWrf

                self.get_nextTimeStep()
                if self.nextTimeStep is None:
                    break
                self.update_VMRemain_infos()
                self.remove_expired_VMs()                
                self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

            if self.nextTimeStep is None:
                break

            # Indicates that parallel tasks have not been allocated yet, and len(temp_Children_finishTask)>=1
            if len(temp_Children_finishTask) > self.dispatchParallelTaskNum: 
                to_be_next = None
                while len(temp_Children_finishTask) > self.dispatchParallelTaskNum:
                    temp_nextTask = temp_Children_finishTask[self.dispatchParallelTaskNum]
                    temp_parent_nextTask = self.nextWrf.get_allpreviousTask(temp_nextTask)
                    if len(temp_parent_nextTask) - len(self.nextWrf.completeTaskSet(temp_parent_nextTask)) >0:
                        self.dispatchParallelTaskNum += 1
                    else: 
                        to_be_next = temp_nextTask
                        break

                if to_be_next is not None: 
                    self.nextTask = to_be_next
                    if len(temp_Children_finishTask) - self.dispatchParallelTaskNum > 1:
                        self.isDequeue = False
                    else:
                        self.isDequeue = True
                    break

                else:  # Mainly to loop this part
                    _, _ = self.vm_queues[self.nextUsr].task_dequeue() # Actually start running self.nextTask here
                    self.firstvmWrfLeaveTime[self.nextUsr] = self.vm_queues[self.nextUsr].get_firstTaskDequeueTime()
                    self.get_nextTimeStep() 
                    self.update_VMRemain_infos()
                    self.remove_expired_VMs()                        
                    self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask) 
                    self.dispatchParallelTaskNum = 0                     
                    if self.nextTimeStep is not None:
                        temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)                                

            else:  # i.e., len(temp_Children_finishTask)<=self.dispatchParallelTaskNum
                # self.nextTask is the last imcompleted successor task of the self.finishTask
                if not self.isDequeue:      # Defaults to True
                    print('self.isDequeue maybe wrong')      
                self.get_nextTimeStep()
                self.update_VMRemain_infos()
                self.remove_expired_VMs()                    
                self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                self.dispatchParallelTaskNum = 0  # Restart recording the number of successor tasks of self.finishTask
                if self.nextTimeStep is not None:
                    temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

        self.numTimestep = self.numTimestep + 1  ## useless for GP
        self.notNormalized_arr_hist = self.stat.update_arrival_rate_history()  ## useless for GP

        done = False
        if self.remainWrfNum == 0:
            if len(self.firstvmWrfLeaveTime) == 0:
                done = True
            elif self.firstvmWrfLeaveTime[0] == math.inf and self.firstvmWrfLeaveTime.count(self.firstvmWrfLeaveTime[0]) == len(self.firstvmWrfLeaveTime):
                done = True

        if done:
            reward = -self.VMcost-self.SLApenalty-self.region_mismatch_penalty  # In DDMWS, the latency penalty is incorporated into the SLA Penalty
            self.episode_info = {"VM_execHour": self.VMexecHours, "VM_totHour": self.VMrentHours,  # VM_totHour is the total rent hours of all VMs
                                 "VM_cost": self.VMcost, "SLA_penalty": self.SLApenalty,
                                 "missDeadlineNum": self.missDeadlineNum,
                                 "RegionMismatchPenalty": self.region_mismatch_penalty,
                                 "RegionMismatchNum": self.region_mismatch_count}
            # print('Useless Allocation has ----> ',self.uselessAllocation)
            self._init()  # cannot delete

        return reward, self.usr_respTime, self.usr_received_wrfNum, self.usr_sent_pktNum, done
               ## r,    usr_respTime,      usr_received_appNum,      usr_sent_pktNum,      d

    def update_VMcost(self, region_id: int, cpu: int, add=True):
        """
            Method to calculate the total VM cost during an episode
            In DDMWS VM region is checked and its specific price is used accordingly
            Args:
                region_id: Int - region id of the selected VM, used to determine the price of VM to use
                cpu: Int - number of CPU's to use for the VM
                add: Boolean that determines if a new VM needs to be leased or not
        """
        if add:
            temp = 1
        else:
            temp = 0
        self.VMcost += temp * self.set.dataset.vm_basefee[region_id] * cpu
        self.VMrentHours += temp
        # print("VM Region ID:", region_id)
        # print("Episode cpu:", cpu)
        # print("Episode VMCost:", self.VMcost)
        # print("Episode VMrentHours:", self.VMrentHours)
        # print("VM Location:", (self.set.dataset.vm_basefee[region_id]))

    # TODO: Isolate all source VM's to US and all successor tasks to asia and record total bits of all tasks
    #  for each workflow. Then verify if additional data transfer costs adds up vs if not distributed
    def update_VMcost_with_data_transfer_cost(self, data_transfer_cost):
        logger.debug(f"Data Transfer Cost to add: {data_transfer_cost}")
        self.VMcost += data_transfer_cost

    def calculate_penalty(self, app, respTime):
        appID = app.get_appID()
        threshold = app.get_Deadline() - app.get_generateTime()
        if respTime < threshold or round(respTime - threshold, 5) == 0:
            return 0
        else:
            self.missDeadlineNum += 1
            return 1+self.set.dataset.wsetBeta[appID]*(respTime-threshold)/3600

    def calculate_region_mismatch_penalty(self, selected_region, task_region, task, successor_tasks):
        """
        Calculates a penalty when a task is executed in a region different from its original region.

        Args:
            selected_region (int): The region ID chosen by the policy for task execution.
            task_region (int or None): The original region of the task (if predefined), or None if dynamically assigned.
            task (int): The task ID.
            successor_tasks (list): The list of successor tasks of the current task.

        Returns:
            float: Penalty score based on the mismatch.
        """
        penalty = 0

        # If the task had a predefined region and was moved to a different region, add penalty
        if task_region is not None and selected_region != task_region:
            region_latency = self.set.dataset.latency_map[task_region][selected_region] / 1000  # Convert ms to seconds
            penalty += region_latency * self.set.regionMismatchPenaltyFactor  # Scale penalty
            self.region_mismatch_count += 1
            logger.debug(
                f"Task {task} moved from region {task_region} to {selected_region}, latency penalty: {penalty}")
        else:
            logger.debug(f"No Mismatch penalty for task {task} being executed in VM region {selected_region}")
        return penalty

    def state_info_construct(self):
        '''
        states:
        1.	Number of child tasks: childNum
        2.	Completion ratio: completionRatio
        3.	Workflow arrival rate: arrivalRate (a vector of historical arrivalRate)
        4.  Task Region ID:
        5.	Whether the VM can satisfy the deadline regardless the extra cost: meetDeadline (0:No, 1:Yes)
        6.	Total_overhead_cost = potential vm rental fee + deadline violation penalty: extraCost
        7.  VM_remainTime: after allocation, currentRemainTime - taskExeTime ( + newVMrentPeriod if applicable)
        8.	BestFit - among all the VMs, whether the current one introduces the lowest extra cost? (0, 1)
        9.  VM Region ID:
        '''

        ob = []

        # ---1)task related state:
        # Task region ID
        # print("Current Task:", self.nextTask)
        # print("All successors:", self.nextWrf.get_allnextTask(self.nextTask))
        task_region = self.nextWrf.processRegion[self.nextTask]  # Assuming tasks have an originDC (region ID)
        # print("Verify task region", {task_region})

        # TODO: 14th Feb - 1) Train without spatial data
        #                  2) Train with increased Latency and RegionMismatch penalty factor - done
        # get region coordinates
        latitude, longitude = self.set.dataset.region_coords[task_region]

        childNum = len(self.nextWrf.get_allnextTask(self.nextTask))  # number of child tasks
        completionRatio = self.nextWrf.get_completeTaskNum() / self.nextWrf.get_totNumofTask()  # self.nextWrf: current Wrf
        arrivalRate = np.sum(np.sum(self.notNormalized_arr_hist, axis=0), axis=0)
        task_ob = [childNum, completionRatio, task_region, latitude, longitude]
        task_ob.extend(list(copy.deepcopy(arrivalRate)))

        # calculate the sub-deadline for a task
        if self.nextWrf not in self.appSubDeadline:
            self.appSubDeadline[self.nextWrf] = {}
            deadline = self.nextWrf.get_maxProcessTime()*self.set.dueTimeCoef[self.nextWrf.get_originDC()][self.nextWrf.get_appID()]
            psd = calPSD(self.nextWrf, deadline, self.set.dataset.vmVCPU)  # get deadline distribution based on upward rank, e.g., {task:sub_deadline}
            for key in psd:
                self.appSubDeadline[self.nextWrf][key] = psd[key]+self.nextTimeStep  # transform into absolute deadline, the task must be completed before appSubDeadline[self.nextWrf][key]

        # ---2)vm related state:
        for vm_ind in range(len(self.vm_queues)):  # for currently rent VM
            vm_region_id = self.vm_queues[vm_ind].regionid  # Assuming each VM has a region ID

            task_est_startTime = self.vm_queues[vm_ind].vmLatestTime()  # get_taskWaitingTime(self.nextWrf,self.nextTask)   # relative time
            task_exe_time = self.nextWrf.get_taskProcessTime(self.nextTask) / self.vm_queues_cpu[vm_ind]
            task_est_finishTime = task_exe_time + task_est_startTime
            temp = round(self.VMRemainingTime[self.vm_queues_id[vm_ind]] - task_est_finishTime, 5)
            if temp > 0:  # the vm can process the task within its current rental hour
                extra_VM_hour = 0
                vm_remainTime = temp
            else:  # need extra VM rental hour
                extra_VM_hour = math.ceil(- temp / self.set.VMpayInterval)
                vm_remainTime = round(self.set.VMpayInterval * extra_VM_hour + self.VMRemainingTime[self.vm_queues_id[vm_ind]] - task_est_finishTime, 5)
            extraCost = (self.set.dataset.vm_basefee[self.vm_queues[vm_ind].regionid] * self.vm_queues_cpu[vm_ind]) * extra_VM_hour
            # print(f"Extra Cost for renting another hour of VM: {extraCost}")
            # print(f"Extra Cost for renting another hour of VM: {self.vm_queues[vm_ind].get_relativeVMloc()}")

            if task_est_finishTime + self.nextTimeStep < self.appSubDeadline[self.nextWrf][self.nextTask]:  # vm can satisfy task sub-deadline
                meetDeadline = 1  # 1: indicate the vm can meet the task sub-deadline
            else:
                meetDeadline = 0
                extraCost += 1 + self.set.dataset.wsetBeta[self.nextWrf.get_appID()] * (task_exe_time + self.nextTimeStep - self.appSubDeadline[self.nextWrf][self.nextTask])  # add SLA penalty
            ob.append([])
            ob[-1] = task_ob + [meetDeadline, extraCost, vm_remainTime, vm_region_id]

        for dcind in range(self.dcNum):  # for new VM that can be rented
            # TODO: Find out what ob does for evaluation (if anything)
            # print(f"For new VM that can be rented dcind: {dcind}")
            for cpuNum in self.set.dataset.vmVCPU:
                task_exe_time = self.nextWrf.get_taskProcessTime(self.nextTask) / cpuNum
                extra_VM_hour = math.ceil(task_exe_time / self.set.VMpayInterval)
                extraCost = self.set.dataset.vm_basefee[dcind] * cpuNum * extra_VM_hour
                # print(f"Extra hour: {extra_VM_hour}, ExtraCost: {extraCost}")
                if task_exe_time + self.nextTimeStep < self.appSubDeadline[self.nextWrf][self.nextTask]:  # vm can satisfy task sub-deadline
                    meetDeadline = 1  # 1: indicate the vm can meet the task sub-deadline
                else:
                    meetDeadline = 0
                    extraCost += 1 + self.set.dataset.wsetBeta[self.nextWrf.get_appID()] * (task_exe_time + self.nextTimeStep - self.appSubDeadline[self.nextWrf][self.nextTask])  # add SLA penalty
                vm_remainTime = round(self.set.VMpayInterval * extra_VM_hour - task_exe_time, 5)
                ob.append([])
                # print(f"Extra Cost: {extraCost} for VM ({cpuNum} CPU) in region: {self.set.dataset.region_map[dcind]}")
                # print(f"VM remaining time: {vm_remainTime}")
                # print(f"New VM that can be rented dcind: {dcind}")
                ob[-1] = task_ob + [meetDeadline, extraCost, vm_remainTime, dcind]

        # if a VM is the best fit, i.e., min(extraCost)
        temp = np.array(ob)
        row_ind = np.where(temp[:, -2] == np.amin(temp[:, -2]))[0]  # relative_row_ind indicates the relative index of VM in vm_satisfyDL_row_ind
        bestFit = np.zeros((len(ob), 1))
        bestFit[row_ind, :] = 1
        ob = np.hstack((temp, bestFit))

        return ob

