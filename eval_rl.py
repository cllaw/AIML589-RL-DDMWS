
"""
    Evaluate the trained model
"""

import argparse
import numpy as np
import os
import random
import torch
from builder import Builder
from config.eval_config import EvalConfig

debug_mode = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(gamma, wf_size, distributed_cloud_enabled):
    print(f"gamma:{gamma}")
    print(f"wf_size:{wf_size}\n")

    # Ya added: Create a fixed matrix for generate a fixed valid_dataset under different seed
    from config.valid_config import test_Set_Generate
    yaml_path = 'config/workflow_scheduling_es_openai.yaml'
    testMatrix = test_Set_Generate(yaml_path).testMatrix
    print(f"Test Matrix: {testMatrix}")

    # which_log = 'logs/WorkflowScheduling-v0'
    which_log = 'logs/WorkflowScheduling-v3'
    log_folders = [f.path for f in os.scandir(which_log) if f.is_dir()]  # Returns the first level of the "%Y%m%d%H%M%S%f" folder

    print("Log Folders", log_folders)
    for log_path in log_folders:
        for fr in np.arange(0, 3020, 20, dtype=int):  # test on each saved model
        # for fr in np.arange(3000, 3020, 20, dtype=int):  # only test the finally saved model
            # if log file not exits, break the loop
            model = f'{log_path}/saved_models/ep_{fr}.pt'
            print("Test", log_path)
            if not os.path.exists(model):
                break
            eval_config = EvalConfig(fr, log_path, wf_size, gamma, distributed_cloud_enabled)
            print(f'gamma:{gamma}, Wf_size:{wf_size}, Log path:{log_path}, model:{fr}')

            set_seed(eval_config.config["yaml-config"]['env']['seed'])
            Builder(eval_config, testMatrix).build().eval()


if __name__ == "__main__":
    # TODO: add distributed_cloud_enabled param to 'workflow_scheduling_es_openai.yaml'
    main(gamma=2.0, wf_size="S", distributed_cloud_enabled=True)
