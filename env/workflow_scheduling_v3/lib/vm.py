# import numpy as np
import os, sys, inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.workflow_scheduling_v3.lib.simqueue import SimQueue
# from workflow_scheduling.env.poissonSampling import one_sample_poisson
from eval_rl import debug_mode

import math
import heapq
import logging

logging.basicConfig(
    level=logging.DEBUG if debug_mode else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class VM:
    def __init__(self, id, cpu, dcind, abind, t, rule, region_id):
        ##self, vmID, vmCPU, dcID, dataset.datacenter[dcid][0], self.nextTimeStep, task_selection_rule
        self.vmid = id
        self.cpu = cpu
        self.loc = dcind  # the index of dc (ranging from 0 to dcNum)
        self.abloc = abind  # the relative index of dc in the topology (ranging from 0 to usrNum)
        self.vmQueue = SimQueue()  # store the apps waiting to be processed
        self.currentTimeStep = t  # record the leave time of the first processing app
        self.rentStartTime = t
        self.rentEndTime = t
        self.processingApp = None  # store the app with the highest priority
        self.processingtask = None  # the task associated with the highest priority app
        self.totalProcessTime = 0  # record the total processing time required for all queuing tasks
        self.pendingTaskTime = 0
        self.pendingTaskNum = 0
        self.taskSelectRule = rule
        self.currentQlen = 0
        self.regionid = region_id  # DDMWS VM creation step with distributed cloud needs to hold region information

    def get_utilization(self, app, task):
        numOfTask = self.totalProcessTime / (app.get_taskProcessTime(task)/self.cpu)
        util = numOfTask/self.get_capacity(app, task)
        print(f"Capacity of task {task}: {self.get_capacity(app, task)}")
        return util  ## == self.totalProcessTime / 60*60

    def get_capacity(self, app, task):
        return 60*60 / (app.get_taskProcessTime(task)/self.cpu)  # how many tasks can processed in one hour

    def get_vmid(self):
        return self.vmid

    def get_cpu(self):
        return self.cpu

    def get_relativeVMloc(self):
        return self.loc

    def get_absoluteVMloc(self):
        return self.abloc

    ## self-defined
    def cal_priority(self, task, app):
        
        if self.taskSelectRule is None:     # use the FIFO principal
            enqueueTime = app.get_enqueueTime(task)
            return enqueueTime
        else:   
            ## task_selection_rule Terminals: ET, WT, TIQ, NIQ, NOC, NOR, RDL
            task_ExecuteTime_real = app.get_taskProcessTime(task)/self.cpu                # ET
            task_WaitingTime = self.get_taskWaitingTime(app, task)                        # WT
            vm_TotalProcessTime = self.vmQueueTime()                                      # TIQ
            vm_NumInQueue = self.currentQlen                                              # NIQ； 
                            # not self.vmQueue.qlen(), because in self.task_enqueue(resort = Ture), it will changes with throwout
            task_NumChildren = app.get_NumofSuccessors(task)                              # NOC
            workflow_RemainTaskNum = app.get_totNumofTask() - app.get_completeTaskNum()   # NOR
            RemainDueTime = app.get_Deadline() - self.currentTimeStep #- task_ExecuteTime_real # RDL

            priority = self.taskSelectRule(ET = task_ExecuteTime_real, WT = task_WaitingTime, TIQ = vm_TotalProcessTime, 
                    NIQ = vm_NumInQueue, NOC = task_NumChildren, NOR = workflow_RemainTaskNum, RDL= RemainDueTime)
            return priority


    def get_firstTaskEnqueueTimeinVM(self):
        if self.processingApp is None:
            return math.inf
        return self.processingApp.get_enqueueTime(self.processingtask)

    def get_firstTaskDequeueTime(self):
        if self.get_pendingTaskNum() > 0:
            return self.currentTimeStep
        else:
            return math.inf

    def get_firstDequeueTask(self):
        return self.processingApp, self.processingtask

    # how long a new task needs to wait if it is assigned
    def get_pendingTaskNum(self):
        if self.processingApp is None:
            return 0
        else:
            return self.vmQueue.qlen()+1  # 1 is needed

    def task_enqueue(self, task, enqueueTime, app, bandwidth_map, latency_map, region_map, data_transfer_cost_map,
                     resort=False):
        """
        Method to enqueue tasks - when a task is placed into a VM for processing

        Args:
            task: the current task ID to be processed on a workflow
            enqueueTime: the enqueue time of the current task
            bandwidth_map: Dict of bandwidth values for each vCPU VM Type.
            latency_map: Dict of inter-region communication delays.
            region_map: Dict of region_ids to region names.
            data_transfer_cost_map: Dict of inter-region data transfer costs.
        """
        temp = app.get_taskProcessTime(task) / self.cpu
        logger.debug(f"Original Task Process time (Size(t)): {app.get_taskProcessTime(task)}")
        logger.debug(f"Original Task Execution Time (EXT(t)): {temp}")

        # Latency and data transfer cost calculation for DDMWS
        communication_delay, data_transfer_cost = app.process_successor_tasks(enqueueTime, task, self.cpu, self.vmid, self.regionid,
                                                                              bandwidth_map, latency_map, region_map, data_transfer_cost_map)
        # print(f"Task {task} ({app.get_taskProcessTime(task)}) with communication delay: {communication_delay}")

        self.totalProcessTime += temp
        self.pendingTaskTime += temp
        self.currentQlen = self.get_pendingTaskNum()

        app.update_executeTime(temp, task)
        app.update_enqueueTime(enqueueTime, task, self.vmid)
        self.vmQueue.enqueue(app, enqueueTime, task, self.vmid, enqueueTime) # last is priority

        if self.processingApp is None:
            """
            self.processingApp is an attribute of the VM class that represents the task currently being processed by this VM.
            If self.processingApp is None, it means the VM is idle and not working on any task.
            """
            self.process_task()

        logger.debug(f"Task {task} complete -> Execution time: {temp}")
        logger.debug(f"---------")
        return temp + communication_delay, data_transfer_cost, communication_delay

    def task_dequeue(self, resort=True):
        task, app = self.processingtask, self.processingApp

        # self.currentTimeStep always == dequeueTime(env.nextTimeStep)

        qlen = self.vmQueue.qlen()
        if qlen == 0:
            self.processingApp = None
            self.processingtask = None
        else:
            if resort:  
                tempvmQueue = SimQueue()
                for _ in range(qlen):
                    oldtask, oldapp = self.vmQueue.dequeue()        # Take out the tasks in self.vmQueue in turn and recalculate
                    priority = self.cal_priority(oldtask, oldapp)   # re-calculate priority
                    heapq.heappush(tempvmQueue.queue, (priority, oldtask, oldapp))
                self.vmQueue.queue = tempvmQueue.queue

            self.process_task()
            self.currentQlen-=1

        return task, app 

    def process_task(self): #
        logger.debug("\n-------")
        logger.debug("Empty Processing App so running this:")
        self.processingtask, self.processingApp = self.vmQueue.dequeue()

        # Pop and return the smallest item from the heap, the popped item is deleted from the heap
        enqueueTime = self.processingApp.get_enqueueTime(self.processingtask)
        processTime = self.processingApp.get_executeTime(self.processingtask)

        taskStratTime = max(enqueueTime, self.currentTimeStep)
        leaveTime = taskStratTime + processTime

        self.processingApp.update_enqueueTime(taskStratTime, self.processingtask, self.vmid)
        self.pendingTaskTime -= processTime
        self.processingApp.update_pendingIndexVM(self.processingtask, self.pendingTaskNum)
        self.pendingTaskNum += 1
        self.currentTimeStep = leaveTime

    def vmQueueTime(self): 
        return max(round(self.pendingTaskTime, 3), 0)

    def vmTotalTime(self): 
        return self.totalProcessTime
    
    def vmLatestTime(self): 
        # return self.totalProcessTime+self.rentStartTime    
        return self.currentTimeStep + self.pendingTaskTime
    
    def get_vmRentEndTime(self):
        return self.rentEndTime
    
    def update_vmRentEndTime(self, time):
        self.rentEndTime += time

    ## real_waitingTime in dual-tree = currentTime - enqueueTime
    def get_taskWaitingTime(self, app, task): 
        waitingTime = self.currentTimeStep - app.get_enqueueTime(task)
        return waitingTime
