import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "weight_decay": weight_decay,
            "epsilon": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))
                state["m"] = group["beta_1"]*m + (1-group["beta_1"])*grad
                state["v"] = group["beta_2"]*v + (1-group["beta_2"])*torch.square(grad)
                lr_t = group["lr"] * math.sqrt(1 - group["beta_2"] ** state.get("t", 1)) / (1 - group["beta_1"] ** state.get("t", 1))
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + group["epsilon"])
                p.data -= group["lr"] * group["weight_decay"] * p.data
                state["t"] = t + 1
        return loss

def get_cosine_anneal_lr(t, lr_max, lr_min, t_warmup, t_anneal):
    if t < t_warmup:
        return lr_max * t/t_warmup
    elif t >= t_warmup and t <= t_anneal:
        scaling_factor = 1 + math.cos(math.pi*(t-t_warmup)/(t_anneal-t_warmup))
        return lr_min + 0.5 * scaling_factor * (lr_max-lr_min)
    else:
        return lr_min

def clip_grad(params, max_norm, epsilon=1e-6):
    grads = [param.grad.data for param in params if param.grad is not None]
    grad = torch.cat(grads, dim=-1)
    norm = grad.square().sum().sqrt()
    if norm >= max_norm:
        scaling_factor = max_norm/(norm + epsilon)
        for idx, _ in enumerate(params):
            if params[idx].grad is not None:
                params[idx].grad.data *= scaling_factor
