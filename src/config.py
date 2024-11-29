import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_numpy(item):
    if torch.is_tensor(item):
        if item.is_cuda:
            return item.cpu().detach().numpy()
        else:
            return item.detach().numpy()
    else:
        return item

def to_torch(item):
    return torch.tensor(item, device=device, dtype=torch.float)

#torch.manual_seed(0)

object_grey_color = 0.6
center_loss_weight = 0.00

shape_smoothness_error_weight = 0.00
symmetry_threshold = 10.0
max_time = 8*60

#symmetry_threshold = 1

def get_free_cuda_mem():

    mem = torch.cuda.mem_get_info(device=0)
    return mem

    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    return f