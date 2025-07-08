import torch
import torch.nn.functional as F
import numpy as np

class Augmentation:
    def __init__(self, prob=0.15):  # 只有15%概率
        self.prob = prob
    
    def add_noise(self, x, noise_factor=0.001):  # 极小噪声
        """添加高斯噪声"""
        noise = torch.randn_like(x) * noise_factor
        return x + noise
    
    def __call__(self, x):
        """只用极轻微的噪声增强"""
        if torch.rand(1) < self.prob:
            x = self.add_noise(x)
        return x