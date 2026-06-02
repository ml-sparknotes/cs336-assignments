import torch
import numpy as np
import random

def sample_batch(x: np.ndarray, batch_size: int, context_length: int, device=None):
    start_idxs = random.sample(range(x.shape[0] - context_length), batch_size)
    context, completion = [], []
    for idx in start_idxs:
        context.append(x[idx : idx + context_length])
        completion.append(x[idx+1 : idx+context_length+1])
    return (
        torch.tensor(np.array(context), dtype=torch.long, device=device),
        torch.tensor(np.array(completion), dtype=torch.long, device=device),
    )
