import argparse
import math
import os
import random
import time

import numpy as np
import torch
import wandb
import yaml

from cs336_basics.adamw import AdamW, get_cosine_anneal_lr
from cs336_basics.data import sample_batch
from cs336_basics.model_building_blocks import (
    TransformerLM,
    cross_entropy_from_logits,
    model_size_in_mb,
)
from cs336_basics.sampling import decode
from cs336_basics.tokenizer import Tokenizer


def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "n_iters": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer, map_location=None):
    checkpoint = torch.load(src, weights_only=False, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["n_iters"]


def compute_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.flatten().square().sum().item()
    return math.sqrt(total)


def clip_gradients(model, max_norm):
    norm = compute_grad_norm(model)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data *= scale
    return norm


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, num_batches):
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        x, y = sample_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy_from_logits(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        total += loss.item()
    model.train()
    return total / num_batches


@torch.no_grad()
def generate_text(model, tokenizer, device, prompt, max_tokens, context_length, temperature, top_p):
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    max_gen = min(max_tokens, context_length - len(prompt_ids) - 1)
    if max_gen <= 0:
        return prompt
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    output_ids = decode(
        model, prompt_tensor, max_gen,
        eot_token=-1, temperature=temperature, top_p=top_p,
    )
    return tokenizer.decode(output_ids.tolist())


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, default="config.yaml",
                    help="Path to YAML config file. CLI flags override config values.")

    p.add_argument("--train_data", type=str)
    p.add_argument("--val_data", type=str)
    p.add_argument("--vocab_file", type=str)
    p.add_argument("--merges_file", type=str)
    p.add_argument("--data_dtype", type=str, choices=["uint16", "uint32"])

    p.add_argument("--vocab_size", type=int)
    p.add_argument("--d_model", type=int)
    p.add_argument("--num_heads", type=int)
    p.add_argument("--num_layers", type=int)
    p.add_argument("--d_ff", type=int)
    p.add_argument("--context_length", type=int)
    p.add_argument("--rope_theta", type=float)

    p.add_argument("--batch_size", type=int)
    p.add_argument("--gradient_accumulation_steps", type=int)
    p.add_argument("--max_iters", type=int)
    p.add_argument("--lr_max", type=float)
    p.add_argument("--lr_min", type=float)
    p.add_argument("--warmup_iters", type=int)
    p.add_argument("--cosine_cycle_iters", type=int)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--beta1", type=float)
    p.add_argument("--beta2", type=float)
    p.add_argument("--adamw_eps", type=float)
    p.add_argument("--grad_clip", type=float)

    p.add_argument("--log_interval", type=int)
    p.add_argument("--eval_interval", type=int)
    p.add_argument("--eval_iters", type=int)
    p.add_argument("--checkpoint_interval", type=int)
    p.add_argument("--checkpoint_dir", type=str)
    p.add_argument("--generate_interval", type=int)
    p.add_argument("--generate_max_tokens", type=int)
    p.add_argument("--generate_temperature", type=float)
    p.add_argument("--generate_top_p", type=float)
    p.add_argument("--generate_prompt", type=str)

    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_run_name", type=str)
    p.add_argument("--wandb_entity", type=str)
    p.add_argument("--wandb_disabled", action="store_true")

    p.add_argument("--device", type=str)
    p.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int)
    p.add_argument("--resume", type=str)

    # First pass: grab --config path
    args = p.parse_args()

    # Load YAML defaults
    config = {}
    with open(args.config) as f:
        config = yaml.safe_load(f) or {}
    print(f"Loaded config from {args.config}")

    # Set YAML values as defaults, then re-parse so CLI flags win
    p.set_defaults(**config)
    args = p.parse_args()

    missing = [k for k, v in vars(args).items() if v is None]
    if missing:
        p.error(f"Missing required config values: {', '.join(missing)}")

    return args


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    np_dtype = {"uint16": np.uint16, "uint32": np.uint32}[args.data_dtype]
    train_data = np.memmap(args.train_data, dtype=np_dtype, mode="r")
    val_data = np.memmap(args.val_data, dtype=np_dtype, mode="r")

    tokenizer = Tokenizer.from_files(
        args.vocab_file, args.merges_file, special_tokens=["<|endoftext|>"]
    )

    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.context_length,
        rope_theta=args.rope_theta,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        device=args.device,
        dtype=dtype,
    )

    num_params = sum(p.numel() for p in model.parameters())
    effective_batch = args.batch_size * args.gradient_accumulation_steps

    print(f"Parameters: {num_params:,} | Size: {model_size_in_mb(model):.2f} MB")
    print(f"Device: {args.device} | dtype: {dtype}")
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")
    print(f"Batch: {args.batch_size} x {args.gradient_accumulation_steps} accum = "
          f"{effective_batch} effective | Context: {args.context_length}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.adamw_eps,
    )

    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer, map_location=args.device)
        print(f"Resumed from iteration {start_iter}")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config={k: v for k, v in vars(args).items()},
        mode="disabled" if args.wandb_disabled else "online",
    )
    wandb.log({"model/num_parameters": num_params,
               "model/size_mb": model_size_in_mb(model)}, step=0)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model.train()
    tokens_processed = 0
    t_start = time.time()

    try:
        for step in range(start_iter, args.max_iters):
            step_t0 = time.time()

            lr = get_cosine_anneal_lr(
                step, args.lr_max, args.lr_min,
                args.warmup_iters, args.cosine_cycle_iters,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            accum_loss = 0.0

            for _ in range(args.gradient_accumulation_steps):
                x, y = sample_batch(train_data, args.batch_size,
                                    args.context_length, args.device)
                logits = model(x)
                loss = cross_entropy_from_logits(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                )
                scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()
                accum_loss += loss.item() / args.gradient_accumulation_steps

            if args.grad_clip > 0:
                pre_clip_norm = clip_gradients(model, args.grad_clip)
            else:
                pre_clip_norm = compute_grad_norm(model)

            optimizer.step()

            step_tokens = args.batch_size * args.context_length * args.gradient_accumulation_steps
            tokens_processed += step_tokens
            step_dt = time.time() - step_t0
            tps = step_tokens / step_dt

            if step % args.log_interval == 0:
                elapsed = time.time() - t_start
                print(
                    f"step {step:6d}/{args.max_iters} | loss {accum_loss:.4f} | "
                    f"ppl {math.exp(min(accum_loss, 100)):.1f} | lr {lr:.2e} | "
                    f"gnorm {pre_clip_norm:.3f} | {step_dt*1000:.0f}ms | "
                    f"{tps:.0f} tok/s | {elapsed:.0f}s"
                )
                wandb.log({
                    "train/loss": accum_loss,
                    "train/perplexity": math.exp(min(accum_loss, 100)),
                    "train/lr": lr,
                    "train/grad_norm": pre_clip_norm,
                    "train/tokens_per_sec": tps,
                    "train/step_time_ms": step_dt * 1000,
                    "train/tokens_processed": tokens_processed,
                }, step=step)

            if step > 0 and step % args.eval_interval == 0:
                train_loss = estimate_loss(
                    model, train_data, args.batch_size,
                    args.context_length, args.device, args.eval_iters,
                )
                val_loss = estimate_loss(
                    model, val_data, args.batch_size,
                    args.context_length, args.device, args.eval_iters,
                )
                train_ppl = math.exp(min(train_loss, 100))
                val_ppl = math.exp(min(val_loss, 100))
                print(
                    f"  eval | train_loss {train_loss:.4f} (ppl {train_ppl:.1f}) | "
                    f"val_loss {val_loss:.4f} (ppl {val_ppl:.1f})"
                )
                wandb.log({
                    "eval/train_loss": train_loss,
                    "eval/val_loss": val_loss,
                    "eval/train_perplexity": train_ppl,
                    "eval/val_perplexity": val_ppl,
                }, step=step)

            if step > 0 and step % args.checkpoint_interval == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{step}.pt")
                save_checkpoint(model, optimizer, step, ckpt_path)
                print(f"  checkpoint saved: {ckpt_path}")

            if step > 0 and step % args.generate_interval == 0:
                try:
                    text = generate_text(
                        model, tokenizer, args.device, args.generate_prompt,
                        args.generate_max_tokens, args.context_length,
                        args.generate_temperature, args.generate_top_p,
                    )
                    print(f"  --- generation (step {step}) ---")
                    print(f"  {text}")
                    print(f"  ---")
                    wandb.log({"generation": wandb.Html(f"<pre>{text}</pre>")},
                              step=step)
                except Exception as e:
                    print(f"  generation failed: {e}")
                finally:
                    model.train()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        interrupt_path = os.path.join(args.checkpoint_dir, "ckpt_interrupted.pt")
        save_checkpoint(model, optimizer, step, interrupt_path)
        print(f"Checkpoint saved: {interrupt_path}")

    final_path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)

    total_time = time.time() - t_start
    print(f"\nTraining complete.")
    print(f"Total tokens: {tokens_processed:,} | Time: {total_time:.0f}s | "
          f"Avg: {tokens_processed / max(total_time, 1):.0f} tok/s")
    print(f"Final checkpoint: {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
