# Running Terra Training on the Cluster

This guide provides instructions for setting up and running Terra training jobs on the compute cluster using SLURM.

## Prerequisites

- Access to the cluster with SLURM workload manager
- CUDA-compatible GPUs (RTX 3090)
- Anaconda/Miniconda installed

## Setup

1. **Conda Environment Setup**:
   ```bash
   # Create the conda environment if you haven't already
   conda env create -f /cluster/home/lterenzi/terra_jax/terra/environment.yml -n terra
   ```

2. **Update the Training Script**:
   - The script `train_cluster.sh` has been prepared for you but may need adjustments:
     - Ensure `CONDA_ROOT` points to your conda installation (currently set to `/home/lorenzo/anaconda3`)
     - Verify `CONDA_ENV` is correct (currently set to `terra`)
     - Update any paths if your file structure differs

## Running Training Jobs

### Submit a Training Job

To submit a training job to the SLURM scheduler:

```bash
# Navigate to the project directory
cd /cluster/home/lterenzi/terra_jax

# Submit the job
sbatch terra-baselines/cluster/train_cluster.sh
```

This will submit your job to the SLURM scheduler and return a job ID.

### Monitor Your Job

You can monitor the status of your job using:

```bash
# Check job status
squeue -u lterenzi

# View job details
scontrol show job <job_id>

# Monitor output in real-time
tail -f <job_id>_training.out
```

### Common Commands

- Cancel a job: `scancel <job_id>`
- View job efficiency: `seff <job_id>`
- Check resource availability: `sinfo`

## Customizing Training Parameters

The current training script uses these parameters:

- `DATASET_PATH=/cluster/home/lterenzi/terra_jax/terra/data/terra/train`
- `DATASET_SIZE=200`

To modify these or add other parameters, edit the `train_cluster.sh` script.

## Resource Allocation

The current script requests:
- 8 CPUs
- 1 RTX 3090 GPU
- 4048 MB memory per CPU
- 3 hours maximum runtime

To adjust these resources, modify the SLURM directives at the top of the `train_cluster.sh` script.

## Troubleshooting

### Common Issues

1. **Conda Activation Errors**:
   - The script uses `eval "$($CONDA_ROOT/bin/conda shell.bash hook)"` which should work in most cluster environments
   - If conda still fails to activate, you might need to use a module-based approach specific to your cluster

2. **Module Not Found Errors**:
   - The script sets `PYTHONPATH=$PYTHONPATH:/cluster/home/lterenzi/terra_jax` to find the `terra` module
   - If you encounter "ModuleNotFoundError", verify that the path is correct and the module is present

3. **GPU Memory Issues**:
   - Adjust your batch size in the training code
   - Monitor GPU usage with `nvidia-smi`

### Getting Help

If you encounter persistent issues:
- Check SLURM logs: `less <job_id>_training.out`
- Contact cluster administrators for cluster-specific issues

## Advanced Configuration

### Multi-GPU Training

To utilize multiple GPUs, modify the SLURM parameter:

```bash
#SBATCH --gpus=rtx_3090:2  # Using 2 GPUs
```

And ensure your training code is set up for distributed training.

### Checkpointing

If your training jobs are long-running, consider implementing checkpointing in your training code to save progress periodically. You can add a checkpoint directory to your script:

```bash
export CHECKPOINT_DIR=/cluster/home/lterenzi/terra_jax/checkpoints
``` 