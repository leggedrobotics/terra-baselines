# Hybrid LLM and RL Approach

In this folder you find the scripts to run Terra using an hybrid LLM and RL approach.

## Setup and usage
1. Make sure that you have Terra and Terra baselines installed and also the additional dependency for this hybrid version. This are listed in the XXX.yaml file (follow the instruction in Terra).

2. Most of the parameters of the experiment can be configured in the `config_llm.yaml` file. The details of each parameters can be founded there as a comment. You can adapt the prompts by changing the corresponding file in the `prompts` folder

3. Before running the main script make sure that you export the API-Key(s) of the model(s) that you wanted to use. This can be done with:
    - Google Models (Gemini): `export GOOGLE_API_KEY=<API_KEY>`
    - OpenAI Models (GPT, o3, ...): `export OPENAI_API_KEY=<API_KEY>`
    - Anthropic Models (Claude): `export ANTHROPIC_API_KEY=<API_KEY>`

4. The main script can then be run with the command:
    ```python
    DATASET_PATH=<PATH_TO_DATASET> DATASET_SIZE=<SIZE> python -m llm.main_llm --model_name <MODEL_NAME> --model_key <MODEL_KEY> --num_timesteps <STEPS> -s <SEED> -n <NUM_ENV> -run <POLICY_PATH> --level_index <LEVEL_INDEX>
    ```
    where:
    - <PATH_TO_DATASET>
    - <SIZE>
    - <MODEL_NAME>
    - MODEL_KEY
    - SEED
    - NUM_ENV
    - POLICY_PATH
    - LEVEL_INDEX

### Supported models

### Level index map

The level index can be choosen according to the following table

| Level index                   | Number |
| --------                      | ------- |
| all                           | None    |
| foundations                   | 0 |
| trenches/single               | 1 |
| trenches/double               | 2 |
| trenches/double_diagonal      | 3 |
| trenches/triple               | 4 |
| trenches/triple_diagonal      | 5 |

## Structure of the folder

```
- llm
    - api_keys
        - ANTHROPIC_API_KEY.txt
        - GOOGLE_API_KEY.txt
        - OPENAI_API_KEY.txt
    - prompts
        - delegation_no_intervention.txt <- delegation prompt intervention disabled
        - delegation.txt <- delegation prompt
        - excavator_action.txt
        - excavator_llm_simple.txt
        - partitioning_exact.txt <- partitioning with an exact number of partitions (sometimes fails)
        - partitioning.txt <-partitioning
    - __init__.py
    - config_llm.py
    - env_llm.py
    - env_manager_llm.py
    - eval_llm.py
    
```

## Note
If you find any bug or issue do not esitate to open a issue and tag @gioelemo 