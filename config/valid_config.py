
"""
    Create a fixed matrix to generate a fixed test_set under different random seeds
"""

import numpy as np
import yaml
import env.workflow_scheduling_v3.lib.dataset as dataset


class test_Set_Generate():
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            wf_num_testing = config['env']['wf_num']
            validNum = config['env']['validNum']

            # Modify this to control how many different workflows we evaluate on
            wf_types = 1  # The wf_types for different size datasets are all 4 in this simulator
            np.random.seed(42)
            self.testMatrix = np.random.randint(0, wf_types, (1, validNum, wf_num_testing))
            f.close()

