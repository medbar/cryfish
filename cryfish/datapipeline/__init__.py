'''
# from traceback_with_variables import Format, default_format as defaults
import torch

# fmt3 = defaults.custom_var_printers.append((torch.Tensor, lambda v: f"{v.__repr__()}, shape={v.shape}"))
# fmt3.append((dict, lambda v: f"DICT({v.keys()}, {v})"))
# print_exc(fmt=fmt3)


from traceback_with_variables.global_hooks import global_print_exc
from traceback_with_variables import core


def dict_to_str(item):
    ov = {
        k: f"shape={v.shape}, {v}" if isinstance(v, torch.Tensor) else v
        for k, v in item.items()
    }
    return f"DICT(keys={item.keys()}, {ov}"


core.default_format = core.default_format.replace(
    custom_var_printers=[
        ((lambda n, t, fn, is_global: is_global), lambda v: None),
        (
            lambda name, type_, filename, is_global: type_ == torch.Tensor,
            lambda v: f"shape={v.shape}, {v}",
        ),
        (lambda name, type_, filename, is_global: type_ == dict, dict_to_str),
        (
            lambda name, type_, filename, is_global: type_ == torch.nn.Module,
            lambda v: "torch.nn.Module",
        ),
    ]
)
global_print_exc()
'''