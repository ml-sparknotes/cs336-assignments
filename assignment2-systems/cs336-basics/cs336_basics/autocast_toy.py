import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # TODO: print dtypes of intermediate activations here
        x = self.relu(self.fc1(x))
        breakpoint()
        x = self.ln(x)
        x = self.fc2(x)
        return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_features, out_features, batch_size = 8, 4, 16

    model = ToyModel(in_features, out_features).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x = torch.randn(batch_size, in_features, device=device)
    y = torch.randn(batch_size, out_features, device=device)

    print("--- outside autocast ---")
    for name, p in model.named_parameters():
        print(f"param {name:20s} dtype={p.dtype}")

    with torch.autocast(device_type=device, dtype=torch.float16):
        print("--- inside autocast ---")
        for name, p in model.named_parameters():
            print(f"param {name:20s} dtype={p.dtype}")

        logits = model(x)
        loss = loss_fn(logits, y)

    loss.backward()
    print("--- gradients (after backward, outside autocast) ---")
    for name, p in model.named_parameters():
        print(f"grad  {name:20s} dtype={p.grad.dtype}")
    optimizer.step()


if __name__ == "__main__":
    main()
