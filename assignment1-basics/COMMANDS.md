# Commands

## BPE Training

```sh
uv run python -m cs336_basics.byte_pair_encoding --input_path data/TinyStoriesV2-GPT4-train.txt --vocab_size 10000 --num_workers 20
```

```sh
uv run python -m cs336_basics.byte_pair_encoding --input_path data/owt_train.txt --vocab_size 32000 --num_workers 8
```

## Profiling

### RAM usage over time (mprof)

```sh
uv run mprof run --include-children python -m cs336_basics.byte_pair_encoding --input_path data/TinyStoriesV2-GPT4-train.txt --vocab_size 1000 --num_workers 1
```

Then, once complete:

```sh
uv run mprof plot -o memory_profile.png
```

Add `@profile` to functions in the source to annotate them on the time-series plot.

### CPU flame graph (py-spy)

```sh
uv run py-spy record -o flamegraph.svg --native --rate 10 -- python -m cs336_basics.byte_pair_encoding --input_path data/TinyStoriesV2-GPT4-train.txt --vocab_size 1000 --num_workers 20
```

Open `flamegraph.svg` in a browser. Add `--nonblocking` if you hit permission errors on Linux.

## Fetch outputs from remote

```sh
rsync -avz -e "ssh -p 32700 -i ~/.ssh/id_ed25519" "root@38.80.152.147:/root/cs336-assignments/assignment1-basics/outputs/" assignment1-basics/outputs/
```

Run from the repo root. Creates/updates `assignment1-basics/outputs/` locally without nesting.
