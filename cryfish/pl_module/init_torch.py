import torch
import random
import numpy as np


def init(seed, precision="medium"):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.set_float32_matmul_precision(precision)


def init_cudnn(cudnn_enabled, cudnn_benchmark, cudnn_deterministic):
    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
