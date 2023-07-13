# rl-excavator-planning
Install the Terra environment with:
```
pip install terra
```
or if not available, clone the Terra repo and install it from the Terra folder with:
```
pip install -e .
```

## Visualization
If you want to render the environment, you are going to need the following:

on Ubuntu:
~~~
sudo apt install libcairo2-dev
sudo apt install libgirepository1.0-dev
~~~

~~~
matplotlib
PyGObject (you could use another backend like PyQt5 but it will clash with cv2 as it's QT-based)
~~~

## Euler cluster workflow
Load modules with
~~~
module load gcc/8.2.0 python_gpu/3.11.2 cuda/11.8.0
~~~

Source venv with
~~~
source ~/venvs/agent/bin/activate
~~~

Launch job from `rl-excavation-planning` folder with something like:
~~~
sbatch -c 32 --mem-per-cpu=4096 --tmp=1024 --gres=gpumem:36000 --gpus=1 --wrap="DATASET_PATH=/cluster/home/aarbues/img_generator DATASET_SIZE=5000 python train.py -n euler_experiment_name"
~~~

It is then possible to track the experiment from the log file on wandb (or in the slurm log file that gets created at job lauch).

## Profiling
You can profile the code by setting to True the `profile` flag in the `ppo.yaml` config file.

This way, the profile stream will be available on the local port 5555.

You can capture the profile stream in a file by running the following in a different terminal.
~~~
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m jax.collect_profile 5555 <duration_in_ms>
~~~

This will dump the profiling logs in a file and generate a Perfetto link to a dashboard.
Click the link to access the profiling.
