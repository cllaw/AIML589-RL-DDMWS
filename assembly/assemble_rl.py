import multiprocessing as mp
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import pickle as pickle
import yaml

from assembly.base_assemble import BaseAssembleRL
from utils.running_mean_std import RunningMeanStd

from utils.policy_dict import agent_policy
import builder


class AssembleRL(BaseAssembleRL):

    def __init__(self, config, env, policy, optim):
        super(AssembleRL, self).__init__()

        self.config = config

        self.env = env
        self.policy = policy
        self.optim = optim

        #  settings for running
        self.running_mstd = self.config.config['yaml-config']["optim"]['input_running_mean_std']
        if self.running_mstd:  # Init running mean and std
            self.ob_rms = RunningMeanStd(shape=self.env.observation_space.shape)
            self.ob_rms_mean = self.ob_rms.mean
            self.ob_rms_std = np.sqrt(self.ob_rms.var)
        else:
            self.ob_rms = None
            self.ob_rms_mean = None
            self.ob_rms_std = None

        self.generation_num = self.config.config['yaml-config']['optim']['generation_num']
        self.processor_num = self.config.config['runtime-config']['processor_num']
        self.eval_ep_num = self.config.config['runtime-config']['eval_ep_num']
        self.valid_ep_num = self.config.config['yaml-config']['env']['validNum']  # Ya added

        # log settings
        self.log = self.config.config['runtime-config']['log']
        self.save_model_freq = self.config.config['runtime-config']['save_model_freq']
        self.save_mode_dir = None

        self.train_or_test = None  # Ya added: control the model for train or test

    def train(self):
        if self.log:
            # Init log repository
            now = datetime.now()
            curr_time = now.strftime("%Y%m%d%H%M%S%f")
            dir_lst = []
            self.save_mode_dir = f"logs/{self.env.name}/{curr_time}"
            dir_lst.append(self.save_mode_dir)
            dir_lst.append(self.save_mode_dir + "/saved_models/")
            dir_lst.append(self.save_mode_dir + "/train_performance/")
            for _dir in dir_lst:
                os.makedirs(_dir)
            # shutil.copyfile(self.args.config, self.save_mode_dir + "/profile.yaml")
            # save the running YAML as profile.yaml in the log
            with open(self.save_mode_dir + "/profile.yaml", 'w') as file:
                yaml.dump(self.config.config['yaml-config'], file)
                file.close()

        # Start with a population init
        population = self.optim.init_population(self.policy, self.env)

        if self.config.config['yaml-config']['optim']['maximization']:
            best_reward_so_far = float("-inf")
        else:
            best_reward_so_far = float("inf")

        for g in range(self.generation_num):
            start_time = time.time()

            self.train_or_test = 'train'
            self.env.set.gamma = self.config.config['yaml-config']['env']['gamma']  # The gamma during training is set in yaml
            arguments = [(indi, self.env, self.optim, self.eval_ep_num, self.ob_rms_mean, self.ob_rms_std,
                          self.processor_num, g, self.config, self.train_or_test) for indi in population]

            # Ya Added: specify the system (Windows/Linux) to execute the project
            if self.config.config['yaml-config']['env']['system'] == "Windows":
                # start rollout works
                start_time_rollout = time.time()
                if self.processor_num > 1:
                    p = mp.get_context('spawn').Pool(self.processor_num)
                    results = p.map(worker_func, arguments)
                    p.close()
                    p.join()
                else:
                    results = [worker_func(arg) for arg in arguments]
            else:
                # start rollout works
                start_time_rollout = time.time()
                if self.processor_num > 1:
                    p = mp.get_context('fork').Pool(self.processor_num)
                    results = p.map(worker_func, arguments)
                    p.close()
                    p.join()
                else:
                    results = [worker_func(arg) for arg in arguments]

            # end rollout
            end_time_rollout = time.time() - start_time_rollout

            # start eval
            start_time_eval = time.time()
            results_df = pd.DataFrame(results).sort_values(by=['policy_id'])

            population, sigma_curr, best_reward_per_g = self.optim.next_population(self, results_df, g)  # 'self' is the first para
            end_time_eval = time.time() - start_time_eval

            end_time_generation = time.time() - start_time

            # update best reward so far
            if self.config.config['yaml-config']['optim']['maximization'] and (best_reward_per_g > best_reward_so_far):
                best_reward_so_far = best_reward_per_g

            if (not self.config.config['yaml-config']['optim']['maximization']) and (best_reward_per_g < best_reward_so_far):
                best_reward_so_far = best_reward_per_g

            # print runtime infor in training process
            print(
                f"\nepisode: {g}, gamma:{self.env.set.gamma}, best reward so far: {best_reward_so_far:.4f}, best reward of the current generation: {best_reward_per_g:.4f}, sigma: {sigma_curr:.3f}, time_generation: {end_time_generation:.2f}, rollout_time: {end_time_rollout:.2f}, eval_time: {end_time_eval:.2f}", flush=True
            )

            # update mean and std every generation
            if self.running_mstd:
                hist_obs = []
                hist_obs = np.concatenate(results_df['hist_obs'], axis=0)
                # Update future ob_rms_mean  and  ob_rms_std
                self.ob_rms.update(hist_obs)
                self.ob_rms_mean = self.ob_rms.mean
                self.ob_rms_std = np.sqrt(self.ob_rms.var)

            if self.log:
                if self.running_mstd:
                    results_df = results_df.drop(['hist_obs'], axis=1)  # remove hist_obs from  log
                # return row of parent policy, i.e., policy_id = -1
                results_df = results_df.loc[results_df['policy_id'] == -1]
                with open(self.save_mode_dir + "/train_performance" + "/training_record.csv", "a") as f:
                    # results_df.to_csv(f, index=False, header=False)
                    results_df.to_csv(f, index=False, header=False, mode='a')  # Ya added

                elite = self.optim.get_elite_model()
                if (g + 1) % self.save_model_freq == 0 or g == 0:
                    if g == 0:  # 保存模型随机初始化时的性能
                        save_pth = self.save_mode_dir + "/saved_models" + f"/ep_{g}.pt"
                    else:
                        save_pth = self.save_mode_dir + "/saved_models" + f"/ep_{(g + 1)}.pt"
                    torch.save(elite.state_dict(), save_pth)
                    if self.running_mstd:
                        if g == 0:  # 保存模型随机初始化时的ob_rms
                            save_pth = self.save_mode_dir + "/saved_models" + f"/ob_rms_{g}.pickle"
                        else:
                            save_pth = self.save_mode_dir + "/saved_models" + f"/ob_rms_{(g + 1)}.pickle"
                        f = open(save_pth, 'wb')
                        pickle.dump(np.concatenate((self.ob_rms_mean, self.ob_rms_std)), f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                        f.close()

            # Ya added: test the model meeting the condition
            if ((g+1) % self.save_model_freq) == 0 or g == 0:
                from utils.policy_dict import agent_policy
                indi_test = []
                agent_ids_test = self.env.get_agent_ids()
                model_test = self.optim.get_elite_model()
                indi_test.append(agent_policy(agent_ids_test, model_test))
                self.train_or_test = 'test'
                self.env.set.gamma = 1  # set the gamma=1 in testing process under the scenario of training
                arguments = [(indi, self.env, self.optim, self.valid_ep_num, self.ob_rms_mean, self.ob_rms_std,
                              self.processor_num, 0, self.config, self.train_or_test) for indi in indi_test]
                            # there is only one dataGen in valid_set, so g=0 here

                # start rollout works
                start_time_test = time.time()
                results = [worker_func(arg) for arg in arguments]
                end_time_test = time.time() - start_time_test
                results_df = pd.DataFrame(results)

                # print runtime infor in testing process
                testing_reward = results_df['rewards'].tolist()[0]
                VM_cost = results_df["VM_cost"].tolist()[0]
                SLA_penalty = results_df["SLA_penalty"].tolist()[0]
                print(
                    f"episode: {g}, gamma:{self.env.set.gamma}, current testing reward: {testing_reward:.4f}, current testing VM_cost: {VM_cost:.4f}, current testing SLA_penalty: {SLA_penalty:.4f}, current testing_time: {end_time_test:.2f}",
                    flush=True
                )

                if self.log:  # save the testing results
                    results_df = results_df.drop(['hist_obs'], axis=1)  # remove hist_obs from log
                    dir_test = self.save_mode_dir + "/test_performance"
                    if not os.path.exists(dir_test):
                        os.makedirs(dir_test)
                    results_df.to_csv(dir_test + "/testing_record_in_training.csv", index=False, header=False, mode='a')

    def eval(self):
        # load policy from log
        self.policy.load_state_dict(torch.load(self.config.config['runtime-config']['policy_path']))
        # create an individual wrapped with agent id
        indi = agent_policy(self.env.get_agent_ids(), self.policy)
        # load runtime mean and std
        if self.running_mstd:
            with open(self.config.config['runtime-config']['rms_path'], "rb") as f:
                ob_rms = pickle.load(f)
                self.ob_rms_mean = ob_rms[:int(0.5 * len(ob_rms))]
                self.ob_rms_std = ob_rms[int(0.5 * len(ob_rms)):]

        self.policy.eval()

        # use the valid_dataset under different seed for testing
        g = 0  # there is only one dataGen in valid_set
        self.train_or_test = 'test'
        arguments = [(indi, self.env, self.optim, self.valid_ep_num, self.ob_rms_mean, self.ob_rms_std,
                      self.processor_num, g, self.config, self.train_or_test)]

        # start rollout works
        start_time_test = time.time()

        results = [worker_func(arg) for arg in arguments]  # Begin evaluation process on VM's

        end_time_test = time.time() - start_time_test

        results_df = pd.DataFrame(results)

        # print runtime info in testing process
        testing_reward = results_df['rewards'].tolist()[0]
        VM_cost = results_df["VM_cost"].tolist()[0]
        VM_execHour = results_df["VM_execHour"].tolist()[0]
        VM_totHour = results_df["VM_totHour"].tolist()[0]
        SLA_penalty = results_df["SLA_penalty"].tolist()[0]
        print(
            f"current testing reward: {testing_reward:.4f}, current VM cost: {VM_cost:.4f}, current SLA penalty: {SLA_penalty:.4f}, testing_time: {end_time_test:.2f}", flush=True
        )
        # VM_totHour is the total rent hours of all VMs
        print(f"current VM Execution time (hours): {VM_execHour:.4f}, current Total VM Execution time (hours): {VM_totHour:.4f}\n", flush=True)

        if self.log:
            results_df = results_df.drop(['hist_obs'], axis=1)  # remove hist_obs from  log
            dir_test = os.path.dirname(self.config.config['runtime-config']['config']) + "/test_performance"
            if not os.path.exists(dir_test):
                os.makedirs(dir_test)
            # results_df.to_csv(dir_test + "/testing_record.csv", index=False, header=False, mode='a')

            # Ya Added
            test_size = self.config.config['yaml-config']['env']['wf_size']
            gamma_size = self.config.config['yaml-config']['env']['gamma']
            results_df.to_csv(dir_test + "/testing_record_"+str(gamma_size)+"_"+str(test_size)+".csv",
                              index=False, header=False, mode='a')


def worker_func(arguments):
    indi, env, optim, eval_ep_num, ob_rms_mean, ob_rms_std, processor_num, g, config, train_or_test = arguments

    if processor_num > 1:
        # env = builder.build_env(config.config)
        env = builder.build_env(config.config, env.testMatrix)  # Ya added

    hist_rewards = {}  # rewards record all evals
    hist_obs = {}  # observation  record all evals
    hist_actions = {}
    obs = None
    total_reward = 0

    # Ya added
    total_VM_execHour = 0
    total_VM_totHour = 0
    total_VM_cost = 0
    total_SLA_penalty = 0
    total_missDeadlineNum = 0

    for ep_num in range(eval_ep_num):
        states = env.reset(g, ep_num, train_or_test)  # Ya added: specify the i-th eval_ep_num in g-th dataGen

        rewards_per_eval = []  # for recording rewards for the current evaluation episode
        obs_per_eval = []      # for recording observations for the current evaluation episode
        actions_per_eval = []  # for recording actions for the current evaluation episode
        done = False

        for agent_id, model in indi.items():  # indi is a policy with agent ID
            model.reset()
        while not done:
            actions = {}
            for agent_id, model in indi.items():
                s = states[agent_id]["state"]
                # reshape s
                if s.ndim < 2:  # make sure ndim of state = 2
                    s = s[np.newaxis, :]
                # update s
                if ob_rms_mean is not None:
                    s = (s - ob_rms_mean) / ob_rms_std
                # feed s into model
                if "removeVM" in states: 
                    actions[agent_id] = model(s, states["removeVM"])
                else:
                    actions[agent_id] = model(s)

                states, r, done, _ = env.step(actions)  # do the cost calculations calculation

                rewards_per_eval.append(r)
                obs_per_eval.append(s)
                actions_per_eval.append(actions[agent_id])
                total_reward += r

                # trace observations
                if obs is None:
                    obs = states["0"]["state"]
                else:
                    obs = np.append(obs, states["0"]["state"], axis=0)

        hist_rewards[ep_num] = rewards_per_eval
        hist_obs[ep_num] = obs_per_eval
        hist_actions[ep_num] = actions_per_eval

        # Ya added
        total_VM_execHour += env.episode_info["VM_execHour"]
        total_VM_totHour += env.episode_info["VM_totHour"]
        total_VM_cost += env.episode_info["VM_cost"]
        total_SLA_penalty += env.episode_info["SLA_penalty"]
        total_missDeadlineNum += env.episode_info["missDeadlineNum"]

    rewards_mean = total_reward / eval_ep_num

    # Ya added
    VM_execHour_mean = total_VM_execHour / eval_ep_num
    VM_totHour_mean = total_VM_totHour / eval_ep_num
    VM_cost_mean = total_VM_cost / eval_ep_num
    SLA_penalty_mean = total_SLA_penalty / eval_ep_num
    missDeadlineNum_mean = total_missDeadlineNum / eval_ep_num

    if env.name in ["WorkflowScheduling-v0", "WorkflowScheduling-v2", "WorkflowScheduling-v3"] and optim.name == "es_openai":

        if indi['0'].policy_id == -1:
            # return {'policy_id': indi['0'].policy_id,
            #         'rewards': rewards_mean,
            #         'hist_obs': obs,
            #         "VM_execHour": env.episode_info["VM_execHour"],
            #         "VM_totHour": env.episode_info["VM_totHour"],
            #         "VM_cost": env.episode_info["VM_cost"],
            #         "SLA_penalty": env.episode_info["SLA_penalty"],
            #         "missDeadlineNum": env.episode_info["missDeadlineNum"]}

            # Ya Added
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    'hist_obs': obs,
                    "VM_execHour": VM_execHour_mean,
                    "VM_totHour": VM_totHour_mean,
                    "VM_cost": VM_cost_mean,
                    "SLA_penalty": SLA_penalty_mean,
                    "missDeadlineNum": missDeadlineNum_mean}

        else:  # we do not record detailed info for non-parent policy
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    'hist_obs': obs,
                    "VM_execHour": np.nan,
                    "VM_totHour": np.nan,
                    "VM_cost": np.nan,
                    "SLA_penalty": np.nan,
                    "missDeadlineNum": np.nan}

    if ob_rms_mean is not None:
        return {'policy_id': indi['0'].policy_id, 'hist_obs': obs, 'rewards': rewards_mean}

    return {'policy_id': indi['0'].policy_id,
            'rewards': rewards_mean}

    # results_produce(env.name, optim.name)


def discount_rewards(rewards):
    gamma = 0.99  # gamma: discount factor in rl
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards
