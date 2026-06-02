import torch


def sample(logits: torch.Tensor, temperature=1., top_p=None):
    probs = torch.softmax(logits / temperature, dim=-1)
    if not top_p:
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        values, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(values, dim=-1)
        mask = cumsum < top_p
        mask[..., 1:] = mask[..., :-1]
        mask[..., 0] = True
        orig_indices = torch.argsort(sorted_indices)
        mask = torch.gather(mask, dim=-1, index=orig_indices)
        probs[~mask] = 0.
        probs = probs / probs.sum(dim=-1).unsqueeze(-1)
        return torch.multinomial(probs, num_samples=1)

def decode(model: torch.nn.Module, seq_tokens: torch.Tensor, max_generated_tokens, eot_token, temperature=1., top_p=None):
    completion = seq_tokens
    while True:
        last_logit = model(completion.unsqueeze(0))[:, -1]
        next_token = sample(last_logit, temperature=temperature, top_p=top_p)[0, :]
        completion = torch.cat([completion, next_token], dim=-1)
        if completion.shape[0] == max_generated_tokens + seq_tokens.shape[0] or next_token == eot_token:
            return completion
