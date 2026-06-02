import argparse
import contextlib
import statistics
import timeit

import torch

import cs336_basics.model as _basics_model
from cs336_basics.model import BasicsTransformerLM, nvtx_range


MODES = ("forward", "forward-backward", "forward-backward-optim")

# Named model-size presets. `small`..`10B` come from the assignment's
# model-sizing table; `tiny` is a small-but-useful preset for quick local runs.
# All `d_model` values are divisible by their `num_heads`.
MODEL_PRESETS: dict[str, dict[str, int]] = {
    "tiny":   {"d_model": 128,  "d_ff": 512,   "num_layers": 2,  "num_heads": 4,  "vocab_size": 2000,  "batch_size": 4, "context_length": 128},
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12, "vocab_size": 10000, "batch_size": 4, "context_length": 512},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16, "vocab_size": 10000, "batch_size": 4, "context_length": 512},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20, "vocab_size": 10000, "batch_size": 4, "context_length": 512},
    "xl":     {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32, "vocab_size": 10000, "batch_size": 4, "context_length": 512},
    "10B":    {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36, "vocab_size": 10000, "batch_size": 4, "context_length": 512},
}

# Keys in MODEL_PRESETS that can be overridden by matching CLI flags.
PRESET_KEYS: tuple[str, ...] = (
    "d_model",
    "d_ff",
    "num_layers",
    "num_heads",
    "vocab_size",
    "batch_size",
    "context_length",
)

DEFAULTS = {
    "vocab_size": 256,
    "context_length": 32,
    "d_model": 32,
    "num_layers": 2,
    "num_heads": 2,
    "d_ff": 64,
    "rope_theta": 10_000.0,
    "batch_size": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0,
    "warm_up": 5,
    "num_samples": 10,
    "mode": "forward",
    "enable_nsys": False,
    "mixed_precision": False,
    "record_memory_history": False,
    "memory_snapshot_path": "memory_snapshot.pickle",
}


def resolve_model_cfg(args: argparse.Namespace) -> dict:
    """Merge MODEL_PRESETS[args.model_size] (or DEFAULTS) with explicit CLI overrides."""
    if args.model_size is not None:
        cfg = dict(MODEL_PRESETS[args.model_size])
    else:
        cfg = {k: DEFAULTS[k] for k in PRESET_KEYS}
    for k in PRESET_KEYS:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark a forward pass of BasicsTransformerLM.")
    p.add_argument(
        "--model-size",
        choices=list(MODEL_PRESETS.keys()),
        default=None,
        help="Pick a model-size preset (tiny/small/medium/large/xl/10B). "
             "Individual flags below override matching keys from the preset.",
    )
    # Preset-overridable args: default=None so we can detect explicit overrides.
    p.add_argument("--vocab-size", type=int, default=None, help="Override preset vocab_size.")
    p.add_argument("--context-length", type=int, default=None, help="Override preset context_length.")
    p.add_argument("--d-model", type=int, default=None, help="Override preset d_model.")
    p.add_argument("--num-layers", type=int, default=None, help="Override preset num_layers.")
    p.add_argument("--num-heads", type=int, default=None, help="Override preset num_heads.")
    p.add_argument("--d-ff", type=int, default=None, help="Override preset d_ff.")
    p.add_argument("--batch-size", type=int, default=None, help="Override preset batch_size.")
    # Non-preset args keep their existing defaults.
    p.add_argument("--rope-theta", type=float, default=DEFAULTS["rope_theta"])
    p.add_argument("--device", type=str, default=DEFAULTS["device"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--warm-up", type=int, default=DEFAULTS["warm_up"])
    p.add_argument("--num-samples", type=int, default=DEFAULTS["num_samples"])
    p.add_argument("--mode", type=str, default=DEFAULTS["mode"], choices=MODES)
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Run one benchmark per entry in MODEL_PRESETS instead of a single run.",
    )
    p.add_argument(
        "--enable-nsys",
        action="store_true",
        default=DEFAULTS["enable_nsys"],
        help="Enable model NVTX ranges only during sample collection (off during warm-up).",
    )
    p.add_argument(
        "--mixed-precision",
        action="store_true",
        default=DEFAULTS["mixed_precision"],
        help="Wrap the forward pass in torch.autocast(bf16).",
    )
    p.add_argument(
        "--record-memory-history",
        action="store_true",
        default=DEFAULTS["record_memory_history"],
        help="Record CUDA memory history during sample collection and dump a pickle "
             "snapshot loadable by https://pytorch.org/memory_viz.",
    )
    p.add_argument(
        "--memory-snapshot-path",
        type=str,
        default=DEFAULTS["memory_snapshot_path"],
        help="Output path for the memory snapshot pickle.",
    )
    return p.parse_args()


def benchmark(
    model: BasicsTransformerLM,
    inputs: torch.Tensor,
    mode: str,
    warm_up: int,
    num_samples: int,
    device: torch.device,
    enable_nsys: bool = False,
    mixed_precision: bool = False,
    record_memory_history: bool = False,
    memory_snapshot_path: str = DEFAULTS["memory_snapshot_path"],
) -> list[float]:
    optimizer = (
        torch.optim.AdamW(model.parameters()) if mode == "forward-backward-optim" else None
    )

    def autocast_ctx():
        if not mixed_precision:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    def backward_nvtx_ctx():
        if _basics_model.PROFILING_MODE:
            return torch.autograd.profiler.emit_nvtx()
        return contextlib.nullcontext()

    def run_step() -> None:
        if mode == "forward":
            with nvtx_range("full forward pass"), autocast_ctx():
                model(inputs)
                return
        if optimizer is not None:
            with nvtx_range("optimizer zero grad"):
                optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)
        with nvtx_range("full forward pass"), autocast_ctx():
            loss = model(inputs).sum()
        with backward_nvtx_ctx():
            loss.backward()
        if optimizer is not None:
            with nvtx_range("optimizer step"):
                optimizer.step()

    prev_profiling_mode = _basics_model.PROFILING_MODE
    try:
        if enable_nsys:
            _basics_model.PROFILING_MODE = False

        for _ in range(warm_up):
            run_step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        if enable_nsys:
            _basics_model.PROFILING_MODE = True

        record_mem = record_memory_history and device.type == "cuda"
        if record_mem:
            torch.cuda.memory._record_memory_history(max_entries=1_000_000)

        times: list[float] = []
        try:
            for _ in range(num_samples):
                start = timeit.default_timer()
                run_step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = timeit.default_timer()
                times.append(end - start)
        finally:
            if record_mem:
                torch.cuda.memory._dump_snapshot(memory_snapshot_path)
                torch.cuda.memory._record_memory_history(enabled=None)
    finally:
        _basics_model.PROFILING_MODE = prev_profiling_mode

    return times


def run_benchmark(
    vocab_size: int = DEFAULTS["vocab_size"],
    context_length: int = DEFAULTS["context_length"],
    d_model: int = DEFAULTS["d_model"],
    num_layers: int = DEFAULTS["num_layers"],
    num_heads: int = DEFAULTS["num_heads"],
    d_ff: int = DEFAULTS["d_ff"],
    rope_theta: float = DEFAULTS["rope_theta"],
    batch_size: int = DEFAULTS["batch_size"],
    device: str = DEFAULTS["device"],
    seed: int = DEFAULTS["seed"],
    warm_up: int = DEFAULTS["warm_up"],
    num_samples: int = DEFAULTS["num_samples"],
    mode: str = DEFAULTS["mode"],
    enable_nsys: bool = DEFAULTS["enable_nsys"],
    mixed_precision: bool = DEFAULTS["mixed_precision"],
    record_memory_history: bool = DEFAULTS["record_memory_history"],
    memory_snapshot_path: str = DEFAULTS["memory_snapshot_path"],
) -> tuple[float, float]:
    """Build model + inputs, run benchmark, return (mean_seconds, std_seconds)."""
    torch.manual_seed(seed)
    torch_device = torch.device(device)

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(torch_device)

    inputs = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=torch_device,
    )

    times = benchmark(
        model=model,
        inputs=inputs,
        mode=mode,
        warm_up=warm_up,
        num_samples=num_samples,
        device=torch_device,
        enable_nsys=enable_nsys,
        mixed_precision=mixed_precision,
        record_memory_history=record_memory_history,
        memory_snapshot_path=memory_snapshot_path,
    )

    mean_s = statistics.mean(times)
    std_s = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_s, std_s


def format_table(rows: list[tuple], columns: list[str]) -> str:
    """Format `rows` as a simple text table with the given column headers."""
    if not columns:
        return ""
    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(values) -> str:
        return " | ".join(str(v).ljust(w) for v, w in zip(values, widths))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(columns), sep]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def _shared_run_kwargs(args: argparse.Namespace) -> dict:
    """Kwargs passed to run_benchmark() that don't depend on the model preset."""
    return dict(
        rope_theta=args.rope_theta,
        device=args.device,
        seed=args.seed,
        warm_up=args.warm_up,
        num_samples=args.num_samples,
        mode=args.mode,
        enable_nsys=args.enable_nsys,
        mixed_precision=args.mixed_precision,
        record_memory_history=args.record_memory_history,
        memory_snapshot_path=args.memory_snapshot_path,
    )


def main() -> None:
    args = parse_args()

    if args.sweep:
        columns = ["model", *PRESET_KEYS, "mean_ms", "std_ms"]
        rows: list[tuple] = []
        for name, preset in MODEL_PRESETS.items():
            cfg = dict(preset)
            for k in PRESET_KEYS:
                v = getattr(args, k)
                if v is not None:
                    cfg[k] = v
            mean_s, std_s = run_benchmark(**cfg, **_shared_run_kwargs(args))
            row = [name, *(cfg[k] for k in PRESET_KEYS),
                   f"{mean_s * 1000:.3f}", f"{std_s * 1000:.3f}"]
            rows.append(tuple(row))
        print(format_table(rows, columns))
        return

    cfg = resolve_model_cfg(args)
    mean_s, std_s = run_benchmark(**cfg, **_shared_run_kwargs(args))
    mean_ms = mean_s * 1000
    std_ms = std_s * 1000
    tag = args.model_size or "(custom)"
    print(f"model={tag} mode={args.mode} batch_size={cfg['batch_size']} samples={args.num_samples}")
    print(f"per-step: {mean_ms:.3f} ms  (std {std_ms:.3f} ms)")


if __name__ == "__main__":
    main()
