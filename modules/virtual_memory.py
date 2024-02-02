import torch
import time
import gc

from safetensors import safe_open
from comfy import model_management
from comfy.diffusers_convert import textenc_conversion_lst


ALWAYS_USE_VM = False

if isinstance(ALWAYS_USE_VM, bool):
    print(f'[Virtual Memory System] Forced = {ALWAYS_USE_VM}')

if 'cpu' in model_management.unet_offload_device().type.lower():
    logic_memory = model_management.total_ram
    global_virtual_memory_activated = ALWAYS_USE_VM if isinstance(ALWAYS_USE_VM, bool) else logic_memory < 30000
    print(f'[Virtual Memory System] Logic target is CPU, memory = {logic_memory}')
else:
    logic_memory = model_management.total_vram
    global_virtual_memory_activated = ALWAYS_USE_VM if isinstance(ALWAYS_USE_VM, bool) else logic_memory < 22000
    print(f'[Virtual Memory System] Logic target is GPU, memory = {logic_memory}')


print(f'[Virtual Memory System] Activated = {global_virtual_memory_activated}')


@torch.no_grad()
def recursive_set(obj, key, value):
    if obj is None:
        return
    if '.' in key:
        k1, k2 = key.split('.', 1)
        recursive_set(getattr(obj, k1, None), k2, value)
    else:
        setattr(obj, key, value)


@torch.no_grad()
def recursive_del(obj, key):
    if obj is None:
        return
    if '.' in key:
        k1, k2 = key.split('.', 1)
        recursive_del(getattr(obj, k1, None), k2)
    else:
        delattr(obj, key)


@torch.no_grad()
def force_load_state_dict(model, state_dict):
    for k in list(state_dict.keys()):
        p = torch.nn.Parameter(state_dict[k], requires_grad=False)
        recursive_set(model, k, p)
        del state_dict[k]
    return


@torch.no_grad()
def only_load_safetensors_keys(filename):
    try:
        with safe_open(filename, framework="pt", device='cpu') as f:
            result = list(f.keys())
        assert len(result) > 0
        return result
    except:
        return None


@torch.no_grad()
def load_from_virtual_memory(model):
    timer = time.time()

    virtual_memory_dict = getattr(model, 'virtual_memory_dict', None)
    if not isinstance(virtual_memory_dict, dict):
        # Not in virtual memory.
        return

    model_file = getattr(model, 'model_file', None)
    assert isinstance(model_file, dict)

    filename = model_file['filename']
    prefix = model_file['prefix']
    original_device = model_file['original_device']

    with safe_open(filename, framework="pt", device=original_device) as f:
        for current_key, (current_key_in_safetensors, current_device, current_flag) in virtual_memory_dict.items():
            tensor = f.get_tensor(current_key_in_safetensors).to(current_device)
            if isinstance(current_flag, tuple) and len(current_flag) == 2:
                a, b = current_flag
                tensor = tensor[a:b]
            parameter = torch.nn.Parameter(tensor, requires_grad=False)
            recursive_set(model, current_key, parameter)

    print(f'[Virtual Memory System] time = {str("%.5f" % (time.time() - timer))}s: {prefix} loaded to {original_device}: {filename}')
    del model.virtual_memory_dict
    return
