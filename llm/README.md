How to run the code

```python
time DATASET_PATH=/home/gioelemo/Documents/terra/data/terra/train DATASET_SIZE=10 python -m llm.main_llm --model_name gemini-2.5-pro-preview-05-06 --model_key gemini --num_timesteps 120 -s 58

time DATASET_PATH=/home/gioelemo/Documents/terra/data_big/terra/train DATASET_SIZE=10 python -m llm.main_llm --model_name gemini-2.5-pro-preview-05-06 --model_key gemini --num_timesteps 120 -s 58
```
Structure of the folder

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
    - ...
```

The configuration of the experiment can be defined in the `llm.yaml` file:

DATASET_PATH=/home/gioelemo/Documents/terra/train4/train DATASET_SIZE=500 python -m llm.main_llm --model_name gemini-2.5-pro --model_key gemini --num_timesteps 400 -s 1 -n 1 --level_index 0

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

### Note
If you find any bug or issue do not esitate to open a issue and tag @gioelemo 