# rl-excavator-planning

## Installation
Install the requirements with:
```
pip install -r requirements.txt
```

Install the Terra environment with:
```
pip install terra
```
or if not available, clone the Terra repo and install it from the Terra folder with:
```
pip install -e .
```
Note: don't forget to install JAX separately (see Terra readme).

## Training
To run a training job, set your hyperparameters in `ppo.yaml`, set your maps in Terra's `config.py`, and set your curriculum
in `curriculum.py`. Note: make sure that the number of map paths you include is the same as the number of curriculum steps
you specify. Untested behavior otherwise.

You can run your training with:
~~~
DATASET_PATH=/path/to/dataset DATASET_SIZE=dataset_size python train.py -n training_job_name -sm model_seed -se env_seed
~~~
The training job is automatically logged on Wandb. If you only want to log it locally, use the environment variable `WANDB_MODE=offline`.

### Visualization from model
The main way to visualize results from the model is using the `pygame` rendering engine from Terra.

Before launching the script, make sure to select in `config.py`, `curriculum.py`, and `ppo.yaml` the settings you
want your model to be tested on.
You can launch the visualization with something like:
~~~
WANDB_MODE=offline DATASET_PATH="/home/antonio/Downloads/img_generator" DATASET_SIZE=1000 python visualize.py -run agents/Terra/models/my_model -nx 3 -ny 4 -steps 120 -d 0 -s 8964231 -pg 0
~~~
This command will load 1000 images per dataset, load my_model (note, without the final .pkl extension), render a visualization
with 3x4 environments (`pygame` will take care of the size of the window) for 120 steps, using sampling (d = deterministic is set to 0), with seed = 8964231, and with a normal type of gif (pg = progressive gif is used to create gifs where the agent trace is left on screen, mainly to generate a static animation for PDF format).

A GIF will be generated and saved to disk.

### Evaluation
You can run the evaluation script with a command like:
~~~
DATASET_PATH=/media/img_generator DATASET_SIZE=3000 python eval.py -run agents/Terra/models/my_model.pkl -nx 500 -ny 1 -steps 300 -d 1
~~~
The flags have the same meaning as in the visualization script.

At the end of the evaluation, the scores will be printed in the terminal.

## Euler cluster workflow
You can use Euler to run multiple jobs at the same time as well as using multiple devices (GPUs) for a single job.

Load modules with
~~~
module load gcc/8.2.0 python_gpu/3.11.2 cuda/11.8.0
~~~

Source your venv with
~~~
source ~/venvs/agent/bin/activate
~~~

Launch job from `rl-excavation-planning` folder with something like:
~~~
sbatch -c 32 --mem-per-cpu=4096 --tmp=1024 --gres=gpumem:36000 --gpus=1 --wrap="DATASET_PATH=/cluster/home/aarbues/img_generator DATASET_SIZE=5000 python train.py -n euler_experiment_name"
~~~

It is then possible to track the experiment from the log file on wandb (or in the slurm log file that gets created at job lauch).

Note: if you launch the jobs this way, the files get loaded as soon as the job STARTS. Meaning, if you want to run multiple
jobs with different config files, you need to wait that the last job is running (using `squeue`) before changing config file.

## Profiling
You can profile the code by setting to True the `profile` flag in the `ppo.yaml` config file.

This way, the profile stream will be available on the local port 5555.

You can capture the profile stream in a file by running the following in a different terminal.
~~~
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m jax.collect_profile 5555 <duration_in_ms>
~~~

This will dump the profiling logs in a file and generate a Perfetto link to a dashboard.
Click the link to access the profiling.
