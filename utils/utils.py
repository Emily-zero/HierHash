import math
import torch
import transformers.tokenization_utils_base



def adjust_learning_rate_cos(optimizer, lr, epoch, num_epochs, num_cycles):
    """Decay the learning rate based on schedule"""
    epochs_per_cycle = math.floor(num_epochs / num_cycles)
    lr *= 0.5 * (1. + math.cos(math.pi * (epoch % epochs_per_cycle) / epochs_per_cycle))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return None

# for text input
def squeeze_dim(obj, dim):
    if torch.is_tensor(obj):
        return torch.squeeze(obj, dim=dim)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = squeeze_dim(v, dim)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(squeeze_dim(v, dim))
        return res
    else:
        raise TypeError("Invalid type for squeeze")

def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to_device(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to_device(v, device))
        return res
    elif isinstance(obj,transformers.tokenization_utils_base.BatchEncoding):
        res = {}
        for k, v in obj.items():
            # print(k)
            # print(v)
            res[k] = move_to_device(v, device)
        return res
    else:
        raise TypeError("Invalid type for move_to_device")


def get_init_function(init_value):
    def init_function(m):
        if init_value > 0:
            if hasattr(m,'weight'):
                m.weight.data.uniform_(-init_value,init_value)
            if hasattr(m,'bias'):
                m.bias.data.fill_(0.)
    return init_function