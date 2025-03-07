## Dynamic multi-workflow scheduling for geo-distributed cloud services: an evolutionary reinforcement learning approach
#### Master of Artificial Intelligence thesis submission by Chuan Law
#### AIML501 + AIML589
![alt text](reference_workflows/CyberShake_30_geo_distributed.png)

### Overview

This codebase is the complete implementation used in [our paper](Chuan-Law-MAI-Thesis-2024.pdf) to optimize Dynamic Workflow Scheduling for
Geo-Distributed Cloud Services. It consists of three main parts:
- Self Attention Policy Network for Cloud Workflow Scheduling (SPN-CWS) developed by [Ya Shen](https://github.com/YaShen998) et al.
  - Located in `policy`
- Simulated Geo-Distributed Cloud Workflow Environment extended by Chuan Law et al.
  - Located in `env/workflow_scheduling_v3`
- Evolutionary Reinforcement Learning (ERL) system based on [OpenAI's Evolution Strategies](https://openai.com/index/evolution-strategies/)
  - Located in `assembly` and `env/gym_openAI`

---

### Pre-reqs
- Python must be installed on your computer environment.
- Run `pip install -r requirements.txt` to download all needed dependencies.
    - Disable the `pygraphviz` requirement if it is causing installation issues. It is only used for the DAG 
      visualization (see notes below) and can be safely excluded without affecting the simulator functionality.
---

### Usage
The follow details how models are trained and evaluated. The default parameters set are the ones used in our [original 
paper](`Chuan-Law-MAI-Thesis-2024.pdf`).

#### Training
Training of a new model is done by executing the `main.py` python file:

```buildoutcfg
python main.py
```

Each newly trained model uses the configuration settings set in `workflow_scheduling_es_openai.yaml` and outputs a 
timestamped folder in the `logs/WorkflowScheduling-v3` directory. Each log folder contains a `profile.yaml` file 
saves the config used for training each model.

The extended geo-distributed simulator has added the following new env parameters to `workflow_scheduling_es_openai.yaml`
which can be modified:

```yaml
distributed_cloud_enabled: True  # Determines if the model is trained on a single or geo-distributed simulator
data_scaling_factor: 0.5  # Used as the scaling factor to approximate physical size of tasks based on processing time
latency_penalty_factor: 0.5  # Used to scale the communication delay between tasks. Set this and the following to 0 to negate these additions in the reward function
region_mismatch_penalty_factor: 0.5  # Used to punish the policy when selecting an inter-region VM to execute a task.
```

Note that enabling the `distributed_cloud_enabled` parameter creates a geo-distributed version of the usual workflow models by
assigning each task to one of three possible regions. This also enables the sceduling policy to use VM's in these regions.
This functionality can be easily extended and/or configured with more or different regions in the `env/workflow_scheduling_v3/lib/dataset.py` file. 
Disabling this just creates all tasks in one region and limits VM usage to only this region as well.

#### Evaluation

Models can be evaluated using the `eval_rl.py` file. By default, it will evaluate every saved checkpoint (generation)
for each model per log file.

```buildoutcfg
python eval_rl.py
```

The parameters used to evaluate a model are set at the initial function call. Ideally, you should match these with 
the corresponding values of the trained model in `profile.yaml` file for the best comparison.
```python
main(
    gamma=2.0, 
    wf_size="S",
    distributed_cloud_enabled=True,
    data_scaling_factor=0.5,
    latency_penalty_factor=0.5,
    region_mismatch_penalty_factor=0.5
)
```

There is also a `debug_mode` parameter in `eval_rl.py` and `main.py` which you can toggle between `True` and `False` if
you would like to see all actions of the scheduling policy on the simulated environment. This also includes all newly developed 
geo-distributed calculations.

---

### Notes

There is a new function created to visualize each DAG created from each of the workflow models and save them as a `.png` 
image. Simply uncomment the `draw_dag` function in the `buildDAGfromXML.py` file. This functionality requires a working
[pygraphviz](https://github.com/pygraphviz/pygraphviz) installation.