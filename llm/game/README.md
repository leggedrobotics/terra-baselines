# LLM Game - Terra Game AI Player

This module implements an LLM-based AI player for the Terra game, replacing manual keyboard controls with intelligent automated gameplay.

## Code Execution
The code can be executed with the following command, similar as the hybrid RL-LLM policy.

```python
DATASET_PATH=<PATH> DATASET_SIZE=<SIZE> python -m llm.game.main_manual_llm --model_name <MODEL_NAME> --model_key <MODEL_KEY> --num_timesteps <NUM_TIMESTEP>
```
The parameters are the same that are defined in the `README.md` file in the parent folder.

**Note**: Ensure that the corresponding API keys are properly exported as environment variables before running the command.

## Running Analysis
To analyze the performance of your LLM game player:

```python
python llm/game/analyze_py.
```

This script will:
- Process the gameplay logs
- Generate cumulative reward plots
- Create final reward visualizations
- Save all outputs to the `analysis/` folder

## Performance Considerations

- API Rate Limits: Be mindful of your LLM provider's rate limits
- Cost Management: Monitor API usage to control costs
- Response Time: Consider caching strategies for frequently occurring game states

## References
This implementation is inspired by the [Atari-GPT](https://github.com/nwayt001/atari-gpt) project, which pioneered the use of LLMs for game playing.