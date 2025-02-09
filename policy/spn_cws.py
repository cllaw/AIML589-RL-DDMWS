
"""
    1. Global information between all VMs are captured using the Transformer encoder (MHSA is included in it)
    2. Task enhancement learn through MLP
    3. Concatenate the learned global information of each VM with the learnd Task enhancement features
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policy.base_model import BasePolicy


class SelfAttentionEncoder(nn.Module):
    def __init__(self, task_fea_size, vm_fea_size, output_size,
                 d_model, num_heads, num_en_layers, d_ff, dropout=0.1):
        super(SelfAttentionEncoder, self).__init__()
        # Task task_preprocess
        self.task_embedding = nn.Sequential(nn.Linear(task_fea_size, d_model))
        self.task_feature_enhance = nn.Sequential(nn.Linear(d_model, 2*d_model),
                                                  nn.ReLU(),
                                                  nn.Linear(2*d_model, d_model))

        # VM task_preprocess
        self.vm_embedding = nn.Sequential(nn.Linear(vm_fea_size, d_model))

        # self-attention
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_en_layers)

        # priority mapping
        self.priority = nn.Sequential(nn.Linear(2 * d_model, 4 * d_model),
                                      nn.ReLU(),
                                      nn.Linear(4 * d_model, 4 * d_model),
                                      nn.ReLU(),
                                      nn.Linear(4 * d_model, output_size))

    def forward(self, task_info, vm_info):
        # Task task_preprocess
        task_embedded = self.task_embedding(task_info)
        task_feature_enhance = self.task_feature_enhance(task_embedded)

        # VM task_preprocess
        vm_embedded = self.vm_embedding(vm_info)

        # MHAS
        vm_embedded = vm_embedded.permute(1, 0, 2)
        global_info = self.transformer_encoder(vm_embedded)  
        global_info = global_info.permute(1, 0, 2)

        # Feature concatenation
        concatenation_features = torch.cat((global_info, task_feature_enhance), dim=-1) 

        # priority mapping
        priority = self.priority(concatenation_features)

        return priority


class WFPolicy(BasePolicy):
    def __init__(self, config, policy_id=-1):
        super(WFPolicy, self).__init__()
        self.policy_id = policy_id  # Parent policy when id = -1, Child policy id >= 0
        self.state_num = config['state_num']
        self.action_num = config['action_num']
        self.discrete_action = config['discrete_action']
        if "add_gru" in config:
            self.add_gru = config['add_gru']
        else:
            self.add_gru = True

        # Policy networks
        self.model = SelfAttentionEncoder(task_fea_size=5, vm_fea_size=5, output_size=1, d_model=16,
                                          num_heads=2, num_en_layers=2, d_ff=64, dropout=0.1)  # IMPORTANT: in DDMWS, the task_fea_size and vm_fea_size may changed and not 4

    def forward(self, x, removeVM=None):
        with (torch.no_grad()):  # Will not call Tensor.backward()
            x = torch.from_numpy(x).float()
            task_info = x[:, 0:-5].unsqueeze(1)  # IMPORTANT: if the state number of task is changed in DDMWS, alter 4 to a right number
            vm_info = x[:, -5::].unsqueeze(1)  # IMPORTANT: if the state number of vm is changed in DDMWS, alter 4 to a right number
            x = self.model(task_info, vm_info)
            x = x.permute(1, 0, 2)

            # print("TASK-INFO:", task_info)
            # print("VM-INFO:", vm_info)

            if removeVM is not None:  # doublecheck the vm that should be removed
                x[:, removeVM, :] = float("-inf")
                
            if self.discrete_action:
                x = F.softmax(x.squeeze(), dim=0)  # all the dimensions of input of size 1 removed.
                x = torch.argmax(x)
            else:
                x = torch.relu(x.squeeze())

            x = x.detach().cpu().numpy()

            return x.item(0)


    def xavier_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)  # if relu used, bias is set to 0.01

    def zero_init(self):
        for param in self.parameters():
            param.data = torch.zeros(param.shape)

    def norm_init(self, std=1.0):
        for param in self.parameters():
            shape = param.shape
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            param.data = torch.from_numpy(out)

    def set_policy_id(self, policy_id):
        self.policy_id = policy_id

    def reset(self):
        pass

    def get_param_list(self):
        param_lst = []
        for param in self.parameters():
            param_lst.append(param.data.numpy())
        return param_lst

    def set_param_list(self, param_lst: list):
        lst_idx = 0
        for param in self.parameters():
            param.data = torch.tensor(param_lst[lst_idx]).float()
            lst_idx += 1
