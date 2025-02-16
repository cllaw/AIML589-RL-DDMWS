from config.base_config import BaseConfig
from builder import Builder
import random
import numpy as np
import torch

debug_mode = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # Ya added: Create a fixed matrix for generate a fixed valid_dataset under different seed
    from config.valid_config import test_Set_Generate
    yaml_path = 'config/workflow_scheduling_es_openai.yaml'
    testMatrix = test_Set_Generate(yaml_path).testMatrix

    baseconfig = BaseConfig()

    # Set global running seed
    set_seed(baseconfig.config["yaml-config"]['env']['seed'])

    # Start assembling RL and training process
    Builder(baseconfig, testMatrix).build().train()


# TODO: add new DDMWS constants to the optim section of the profile.yml
if __name__ == "__main__":
    main()
