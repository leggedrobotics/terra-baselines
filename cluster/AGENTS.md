# Euler cluster setup + training (generic)

This guide is a **general** template for running Terra + terra-baselines on Euler. Replace paths, usernames, and WANDB credentials with your own. Do **not** use a work directory for conda; use home/scratch/project as per Euler docs.

## 1) Choose storage locations

- **Code (shared/persistent):** `/cluster/home/<user>/terra-workspace/`
  - `terra/` and `terra-baselines/` live here
  - Home is a **hard 50 GB cap** (not purged): code and small config only.
- **Dataset (large, ephemeral):** `/cluster/scratch/<user>/terra/data/terra/train/`
- **WandB logs:** `/cluster/scratch/<user>/wandb/`
- **Checkpoints + run logs:** `/cluster/scratch/<user>/codex_terra_edge_runs/`

### Euler storage contract (do not violate)

- `WANDB_DIR`, checkpoint dirs, and run logs go on **scratch**, NEVER
  `/cluster/home`. On 2026-07-20 a Terra job died with `Disk quota exceeded`
  because these were written to home (home was 44.2/50 GB); the fix was to move
  them to `/cluster/scratch/<user>/codex_terra_edge_runs/` and symlink from the
  home workspace.
- **Scratch is purged when a file is not accessed for ~15 days.** Anything
  needed long-term must be `rsync`'d to `/cluster/work/rsl/<user>` (final large
  checkpoints, tars; ≤200 GB) or `/cluster/project/rsl/<user>` (conda envs /
  venvs, many small files; ≤75 GB, high inode) — both verified writable — or be
  rebuildable. A venv left on scratch was already corrupted by the purge (empty
  `jax` namespace package, imports fail).
- Smoke/preflight should parse `lquota` and abort before training if
  `/cluster/home/<user>` is above ~90% of its hard quota.

## 2) Sync code + dataset

From your local machine (replace `<user>`):

```bash
rsync -az --partial --info=progress2 --exclude 'data/' \
  ~/Desktop/terra-workspace/terra/ \
  <user>@euler.ethz.ch:/cluster/home/<user>/terra-workspace/terra/

rsync -az --partial --info=progress2 \
  ~/Desktop/terra-workspace/terra-baselines/ \
  <user>@euler.ethz.ch:/cluster/home/<user>/terra-workspace/terra-baselines/

rsync -az --info=progress2 \
  ~/Desktop/terra-workspace/terra/data/terra/train/ \
  <user>@euler.ethz.ch:/cluster/scratch/<user>/terra/data/terra/train/
```

## 3) Conda on Euler (scratch storage for env + pkgs)

Euler docs: conda creates many files; use **scratch** or **project**, not work dir.

```bash
mkdir -p /cluster/scratch/<user>/conda/envs /cluster/scratch/<user>/conda/pkgs

cat > ~/.condarc <<'EOF'
envs_dirs:
  - /cluster/scratch/<user>/conda/envs
pkgs_dirs:
  - /cluster/scratch/<user>/conda/pkgs
EOF
```

Install Miniconda (home is ok, env lives in scratch due to `.condarc`):

```bash
cd ~
wget -O Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
```

Accept Anaconda TOS (first time only):

```bash
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Create env from Terra:

```bash
~/miniconda3/bin/conda env create -f /cluster/home/<user>/terra-workspace/terra/environment.yml -n terra
```

Editable install for Terra:

```bash
~/miniconda3/bin/conda run -n terra python -m pip install -e /cluster/home/<user>/terra-workspace/terra
```

**Important:** `terra-baselines` is a flat-layout repo and is **not pip‑installable**. Run scripts directly from its folder.

## 4) JAX + CUDA (GPU)

Use a CUDA12-compatible wheel. A stable combo that keeps NumPy < 2:

```bash
~/miniconda3/bin/conda run -n terra python -m pip install -U --force-reinstall \
  "jax==0.4.28" "jaxlib==0.4.28+cuda12.cudnn89" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

~/miniconda3/bin/conda run -n terra python -m pip install -U --force-reinstall \
  numpy==1.26.4 ml-dtypes==0.3.2 scipy==1.12.0 opt-einsum==3.3.0
```

Ensure cuDNN 8 (needed by the above jaxlib):

```bash
~/miniconda3/bin/conda run -n terra python -m pip install -U --force-reinstall nvidia-cudnn-cu12==8.9.7.29
```

## 5) WANDB (online)

Put your WANDB credentials in `~/.bashrc`:

```bash
export WANDB_API_KEY=YOUR_KEY
export WANDB_PROJECT=YOUR_PROJECT
export WANDB_USERNAME=YOUR_USERNAME
export WANDB_DIR=/cluster/scratch/<user>/wandb
```

If you don’t want wandb, set `WANDB_MODE=offline` in the job script.

## 6) SLURM training job template

Create a script in this folder (edit placeholders):

```bash
#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=gpu:rtx_4090:1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name="terra-train-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output=%j_training.out

source /cluster/home/<user>/.bashrc

module load eth_proxy
module load stack/2024-06 cuda/12.1.1

CONDA_ROOT=/cluster/home/<user>/miniconda3
CONDA_ENV=terra

_eval_conda_hook="$($CONDA_ROOT/bin/conda shell.bash hook)"
eval "$_eval_conda_hook"
conda activate "$CONDA_ENV"

export PYTHONNOUSERSITE=1
export WANDB_MODE=online
export WANDB_DIR=/cluster/scratch/<user>/wandb
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export DATASET_PATH=/cluster/scratch/<user>/terra/data/terra/train
export DATASET_SIZE=<N>

# fail fast
if [ ! -d "$DATASET_PATH" ]; then
  echo "DATASET_PATH not found: $DATASET_PATH" >&2
  exit 1
fi
if [ "$WANDB_MODE" = "online" ] && [ -z "$WANDB_API_KEY" ]; then
  echo "WANDB_API_KEY not set. Export it or set WANDB_MODE=offline." >&2
  exit 1
fi
mkdir -p "$WANDB_DIR"

# ensure CUDA libs from pip are visible
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
for d in "$CONDA_PREFIX"/lib/python3.12/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done

cd /cluster/home/<user>/terra-workspace/terra-baselines
python -u -s train_mixed.py -d 1 -n wheeled-excavator \
  --num_envs_per_device 256 \
  --total_timesteps 30000000 \
  --log_train_interval 10 --log_eval_interval 200 --checkpoint_interval 100
```

## 7) Submit + monitor

```bash
cd /cluster/home/<user>/terra-workspace
sbatch terra-baselines/cluster/<your_script>.sh
squeue -u $USER
```

Logs are created where you submitted the job:

```
/cluster/home/<user>/terra-workspace/<JOBID>_training.out
```

## 8) Checkpoints

Point the checkpoint output at **scratch**, not the in-repo `terra-baselines/checkpoints/`
default (which lands in home and will hit the 50 GB cap):

```
/cluster/scratch/<user>/codex_terra_edge_runs/checkpoints/
```

They are overwritten every `checkpoint_interval`. Scratch is purged after ~15 days of no access,
so `rsync` any checkpoint you want to keep to `/cluster/work/rsl/<user>` (large, persistent)
before it is evaluated or archived.

## Sync local code changes to Euler

If you made **local edits** and want to push them to the cluster, use `rsync` to update only changed files:

```bash
# Terra (code only, exclude data)
rsync -az --partial --info=progress2 --exclude 'data/' \
  ~/Desktop/terra-workspace/terra/ \
  <user>@euler.ethz.ch:/cluster/home/<user>/terra-workspace/terra/

# Terra-baselines
rsync -az --partial --info=progress2 \
  ~/Desktop/terra-workspace/terra-baselines/ \
  <user>@euler.ethz.ch:/cluster/home/<user>/terra-workspace/terra-baselines/
```

Tip: add `--delete` if you want the cluster copy to exactly mirror your local tree (be careful with generated files).

## Retrieve model checkpoints from Euler to local

Checkpoints live on the cluster at:

```
/cluster/scratch/<user>/codex_terra_edge_runs/checkpoints/
```

To pull them to your local machine:

```bash
# Pull all checkpoints
rsync -az --info=progress2 \
  <user>@euler.ethz.ch:/cluster/scratch/<user>/codex_terra_edge_runs/checkpoints/ \
  ~/Desktop/terra-workspace/terra-baselines/checkpoints/

# Or pull a single checkpoint
rsync -az --info=progress2 \
  <user>@euler.ethz.ch:/cluster/scratch/<user>/codex_terra_edge_runs/checkpoints/<checkpoint>.pkl \
  ~/Desktop/terra-workspace/terra-baselines/checkpoints/
```
