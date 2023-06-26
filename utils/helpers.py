"""
Partially from https://github.com/RobertTLange/gymnax-blines
"""
import pickle
from pathlib import Path

def load_config(config_fname, seed_env=None, seed_model=None, lrate=None, wandb=None, run_name=None):
    """Load training configuration and random seed of experiment."""
    import yaml
    import re

    def load_yaml(config_fname: str) -> dict:
        """Load in YAML config file."""
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        with open(config_fname) as file:
            yaml_config = yaml.load(file, Loader=loader)
        return yaml_config

    config = load_yaml(config_fname)
    if run_name is not None:
        config["train_config"]["run_name"] = run_name
    if wandb is not None:
        config["train_config"]["wandb"] = wandb
    if seed_env is not None:
        config["train_config"]["seed_env"] = seed_env
    if seed_model is not None:
        config["train_config"]["seed_model"] = seed_model
    if lrate is not None:
        if "lr_begin" in config["train_config"].keys():
            config["train_config"]["lr_begin"] = lrate
            config["train_config"]["lr_end"] = lrate
        else:
            try:
                config["train_config"]["opt_params"]["lrate_init"] = lrate
            except Exception:
                pass
    return config


def save_pkl_object(obj, filename):
    """Helper to store pickle objects."""
    output_file = Path(filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    print(f"Stored data at {filename}.")

def append_to_pkl_object(obj, step, foldername):
    """Helper to append to pickle objects."""
    output_file = Path(foldername) / ("eval_" + str(step) + ".pkl")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(str(output_file), "ab") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    print(f"Appended data at {str(output_file)}.")


def load_pkl_object(filename: str):
    """Helper to reload pickle objects."""
    import pickle

    with open(filename, "rb") as input:
        obj = pickle.load(input)
    print(f"Loaded data from {filename}.")
    return obj
