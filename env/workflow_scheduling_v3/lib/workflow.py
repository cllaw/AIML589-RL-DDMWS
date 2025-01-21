
class Workflow:
    def __init__(self, generateTime1, app, appID, originDC, time, ratio, index):
        self.generateTime = generateTime1
        self.origin = originDC  # comes from which user
        self.app = app  # e.g., w1 = [n1, n2, n3, n4, n5, n6],  n1 = [[], [2], 's1'], n2 = [[1], [3], 's14']
        self.appID = appID  ## Workflow is workflow；AppID is used to differentiate different workflows. A workflow includes a number of different tasks
        self.appArivalIndex = index
        self.readyTime = {}  # ready time
        self.enqueueTime = {}  # {nodeID: enqueueTime for the service}
        self.dequeueTime = {}  # {nodeID: dequeueTime for each service} to ensure all parallel services are completed before the aggregated service starts
        self.pendingIndexOnDC ={}
        self.processDC = {}  # {nodeID: dc_ind} ensure parallel services are aggregated to the same destination service
        self.processRegion = {}  # {nodeID: region_id} ensure all tasks have information about what region they were run from
        self.executeTime = {}
        self.queuingTask = []  # tasks that are currently in queue. Mostly for parallel services
        self.maxProcessTime = time  # the max processing time for the app if parallel processing is enabled and no propagation latency
        self.dueTimeCoef = ratio
        self.deadlineTime = self.maxProcessTime * self.dueTimeCoef + self.generateTime

    def __lt__(self, other):  
        return self.generateTime < other.generateTime

    def get_completeTaskNum(self):
        return len(self.dequeueTime)

    def get_maxProcessTime(self):
        return self.maxProcessTime

    # return True if tasks in s_list are completed, otherwise False
    def is_completeTaskSet(self, s_list):
        if set(s_list).issubset(set(self.dequeueTime.keys())):
            return True
        else:
            return False

    # return the set of tasks in s_list that are completed
    def completeTaskSet(self, s_list):
        return set(s_list).intersection(set(self.dequeueTime.keys()))

    def update_pendingIndexVM(self, task, pendingTndex):
        self.pendingIndexOnDC[task] = pendingTndex

    # time: enqueue time for "service" which is allocated in "dc"
    def update_enqueueTime(self, time, task, vmID):
        if task not in self.enqueueTime:
            self.readyTime[task] = time
            self.enqueueTime[task] = time
            self.processDC[task] = vmID  # guarantee results from all parallel services aggregating at the same destination

        elif time > self.enqueueTime[task]:
            self.enqueueTime[task] = time

    def update_dequeueTime(self, time, task):
        self.dequeueTime[task] = time

    def update_executeTime(self, time, task):
        self.executeTime[task] = time

    def get_executeTime(self, task):
        return self.executeTime[task]

    def add_queuingTask(self, task):
        self.queuingTask.append(task)

    def remove_queuingTask(self, task):
        self.queuingTask.remove(task)

    def get_taskDC(self, task):
        if task in self.processDC:
            return self.processDC[task]
        else:
            return None

    def get_taskRegion(self, task):
        if task in self.processRegion:
            return self.processRegion[task]
        else:
            return None

    def get_generateTime(self):
        return self.generateTime

    def get_enqueueTime(self, task):
        return self.enqueueTime[task]

    def get_readyTime(self, task):
        return self.readyTime[task]

    def get_originDC(self):
        return self.origin

    def get_appID(self):
        return self.appID

    def get_taskProcessTime(self, task):
        return self.app.nodes[task]['processTime']  # networkx
        # return self.app.vs[task]['processTime']  # igraph

    # TODO: Currently each task is an object as part of a DAG data structure.
    #  Consider how you will add source region location to a each task in a workflow
    #  What would constitute as inter region data transfer?
    #  - When a sucessor task gets run on a VM outside its source region?
    # return the next task list given the current task ID
    def get_allnextTask(self, task):
        if task is None:
            root = [n for n, d in self.app.in_degree() if d == 0]  # networkx
            return root
        else:
            return list(self.app.successors(task))

    def get_NumofSuccessors(self, task):          
        node_succ = self.get_allnextTask(task)
        return len(node_succ)

    def get_Deadline(self):              
        return self.deadlineTime 
 
    # return the parent tasks given the current task
    def get_allpreviousTask(self, task):
        if task is None:
            return []
        else:
            return list(self.app.predecessors(task))

    def get_totNumofTask(self):
        return self.app.number_of_nodes()  # networkx
        # return self.app.vcount()    # igraph

    def get_allTask(self):
        return list(self.app.nodes)  # networkx
        # return self.app.vs.indices   # igraph

    # Store the region ID for each task based on the VM processing it
    def update_taskLocation(self, task, vm_region_id):
        self.processRegion[task] = vm_region_id

    def calculate_communicationDelay(self, task, successorTask, dataSize_bits, bandwidth_map, latency_map):
        """
        Calculate communication delay between tasks processed in different regions.
        Args:
            task: Current task ID.
            successorTask: ID of the successor task.
            dataSize_bits: Float representing the approximated size in bits of the task to process
            bandwidth_map: Dict of bandwidth values for each vCPU VM Type.
            latency_map: Dict of inter-region communication delays.
        Returns:
            Delay (float) if tasks are in different regions, else 0.
        """
        region1 = self.get_taskRegion(task)
        region2 = self.get_taskRegion(successorTask)

        if region1 is not None and region2 is not None and region1 != region2:
            return (dataSize_bits / 8 * 1e9) + latency_map[region1][region2]
        return 0

    def process_successor_tasks(self, task, data_scaling_factor, bandwidth_map, latency_map):
        """
        Process the successors of the given task with region-based logic.
        Args:
            task: Current task ID.
            data_scaling_factor: Float to use when approximating task execution time to data size
            bandwidth_map: Dict of bandwidth values for each vCPU VM Type.
            latency_map: Dict of inter-region communication delays.
        """
        successors = self.get_allnextTask(task)

        # In DDMWS we estimate the datasize in bits of a task based on its processing time
        dataSize_mb = self.get_taskProcessTime(task) * data_scaling_factor  # Data size in MB
        dataSize_bits = dataSize_mb * 8 * 1e6  # Convert MB to bits

        for successor in successors:
            delay = self.calculate_communicationDelay(task, successor, dataSize_bits, bandwidth_map, latency_map)
            if delay > 0:
                print(f"Adding delay of {delay} for inter-region communication.")
                # Adjust enqueue time or other timing metrics accordingly
                self.enqueueTime[successor] += delay
