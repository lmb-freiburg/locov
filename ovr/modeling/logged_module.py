import torch
from torch import nn
import torch.distributed as dist
from detectron2.utils.events import get_event_storage
from torch.nn import functional as F


def stats(tensor):
    t = tensor.cpu().detach().numpy()
    return {
        "device": tensor.device.index,
        "shape": tensor.shape,
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.to(torch.float32).mean()),
        "std": float(tensor.to(torch.float32).std()),
    }


class LoggedModule(nn.Module):
    def __init__(self):
        super(LoggedModule, self).__init__()
        self.log_info = {}
        self._log_print = False
        self._log_raise_nan = False

    def log(self, name, tensor):
        s = stats(tensor)
        self.log_info[name] = s
        if self._log_print:
            print(f"RANK {dist.get_rank()}: {name}", s)
        if self._log_raise_nan and torch.isnan(tensor).any():
            raise ValueError()

    def log_dict(self, d):
        self.log_info.update(d)
        if self._log_print:
            print(f"RANK {dist.get_rank()}: {d}")
        if self._log_raise_nan:
            for v in d.values():
                if torch.isnan(v).any():
                    raise ValueError()


def binary_cross_entropy_with_logits(input, target, *, reduction="mean", **kwargs):
    """
    Same as `torch.nn.functional.binary_cross_entropy_with_logits`, but returns 0 (instead of nan)
    for empty inputs.
    """
    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    return F.binary_cross_entropy_with_logits(input, target, **kwargs)


def normalize_vec(vec_tensor, dim=1):
    """
    vec_norm = (vec_tensor.detach() ** 2).sum(dim=dim, keepdim=True).sqrt()
    vec_tensor = vec_tensor / vec_norm
    vec_tensor = torch.where(
        torch.isnan(vec_tensor),
        torch.zeros_like(vec_tensor),
        vec_tensor)
    """
    vec_tensor = F.normalize(vec_tensor, p=2, dim=dim)
    return vec_tensor


def standardize_vec(vec_tensor, dim=1):
    vec_tensor = (vec_tensor - vec_tensor.mean(dim, keepdim=True)) / (
        vec_tensor.std(dim, keepdim=True) + 1e-12
    )
    return vec_tensor
