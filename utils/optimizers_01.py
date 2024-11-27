
import numpy as np


class CustomAdamOptimizer:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        初始化 Adam 优化器

        参数:
        - params: 模型参数组成的列表（numpy 数组）
        - lr: 学习率 (默认为 0.001)
        - betas: 用于计算移动平均梯度和平方梯度的指数衰减因子 (默认为 (0.9, 0.999))
        - eps: 为了数值稳定性而添加到分母中的小常数 (默认为 1e-8)
        - weight_decay: 权重衰减 (L2 正则化) 参数 (默认为 0)
        """
        self.params = params  # 模型参数
        self.lr = lr  # 学习率
        self.betas = betas  # beta1 和 beta2 参数
        self.eps = eps  # 用于数值稳定性的小常数
        self.weight_decay = weight_decay  # 权重衰减参数
        self.m = [np.zeros_like(p) for p in params]  # 一阶矩估计（梯度的移动平均值）
        self.v = [np.zeros_like(p) for p in params]  # 二阶矩估计（梯度的平方的移动平均值）
        self.t = 0  # 步数计数器

    def step(self, grads):
        """
        执行一步参数更新

        参数:
        - grads: 梯度组成的列表（numpy 数组），与模型参数对应

        返回值:
        - 更新后的模型参数组成的列表（numpy 数组）
        """
        self.t += 1  # 更新步数计数器
        bias_correction1 = 1 - self.betas[0]**self.t  # 一阶矩估计偏差修正项
        bias_correction2 = 1 - self.betas[1]**self.t  # 二阶矩估计偏差修正项
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # 添加权重衰减（L2 正则化）
            if self.weight_decay != 0:
                grad += self.weight_decay * param

            # 更新一阶矩估计（移动平均梯度）
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            # 更新二阶矩估计（移动平均梯度平方）
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)
            # 计算修正后的一阶矩估计
            m_hat = self.m[i] / bias_correction1
            # 计算修正后的二阶矩估计
            v_hat = self.v[i] / bias_correction2
            # 计算参数更新步长
            step = -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            # 更新参数
            self.params[i] += step

        return self.params
