from assembly.assemble_rl import AssembleRL
from utils.utils import get_state_num, get_action_num, is_discrete_action, get_nn_output_num


class Builder:
    def __init__(self, baseconfig, testMatrix):
        # self.args = baseconfig.config['runtime-config']
        # self.config = baseconfig.config['yaml-config']
        self.config = baseconfig
        self.env = None
        self.policy = None
        self.optim = None

        self.testMatrix = testMatrix

    def build(self):
        env = build_env(self.config.config, self.testMatrix)
        self.config.config['yaml-config']["policy"]["discrete_action"] = is_discrete_action(env)  # based on the environment, decide if the action space is discrete
        self.config.config['yaml-config']["policy"]["state_num"] = get_state_num(env)  # based on the environment, generate the state num to build policy
        self.config.config['yaml-config']["policy"]["action_num"] = get_nn_output_num(env)  # based on the environment, generate the action num to build policy
        # self.config.config['yaml-config']["policy"]["action_num"] = get_action_num(env) # based on the environment, generate the action num to build policy
        policy = build_policy(self.config.config['yaml-config']["policy"])
        optim = build_optim(self.config.config['yaml-config']["optim"])
        return AssembleRL(self.config, env, policy, optim)


def build_env(config, testMatrix):
    env_name = config['yaml-config']["env"]["name"]
    config['yaml-config']['env']['evalNum'] = config['runtime-config']['eval_ep_num']
    if env_name == "WorkflowScheduling-v3":
        from env.workflow_scheduling_v3.simulator_wf import WFEnv
        return WFEnv(env_name, config['yaml-config']["env"], testMatrix)
    else:
        raise AssertionError(f"{env_name} doesn't support, please specify supported a env in yaml.")


def build_policy(config):
    model_name = config["name"]
    if model_name == "model_workflow":
        from policy.wf_model import WFPolicy  # the algorithm
        return WFPolicy(config)
    else:
        raise AssertionError(f"{model_name} doesn't support, please specify supported a model in yaml.")


def build_optim(config):
    optim_name = config["name"]
    if optim_name == "es_openai":
        from optim.es.es_openai import ESOpenAI  # ES to train the algorithm
        return ESOpenAI(config)
    else:
        raise AssertionError(f"{optim_name} doesn't support, please specify supported a optim in yaml.")
