import torch
torch.set_num_threads(16)


def set_cuda_precision():
    torch.set_float32_matmul_precision('medium')
