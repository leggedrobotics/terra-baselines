#!/usr/bin/env python3
"""
Training script for mixed agent environments (Tracked Excavators + Skid Steers)
This uses the unified network with agent type conditioning.


TRAINING MODES:
===============

1. EXCAVATORS ONLY (--agent_config excavators):
   - Both agents use the same type (final_agent1_type = final_agent2_type)
   - Good for initial learning of basic skills
   - Sufficient agent density for alternating training

2. MIXED AGENTS (--agent_config mixed):  
   - Agent 1: final_agent1_type (e.g., tracked excavator = 0)
   - Agent 2: final_agent2_type (e.g., skid steer = 2)
   - Full heterogeneous multi-agent training from start

3. CURRICULUM (--agent_config curriculum):
   - Start: initial_agent1_type + initial_agent2_type (e.g., both skidsteers = 2,2)
   - Switch: final_agent1_type + final_agent2_type (e.g., excavator+skidsteer = 0,2) at specified curriculum level
   - Gradual introduction of complexity

USAGE EXAMPLES:
===============

# Train with both skidsteers initially (no curriculum)
python train_mixed_agents.py --agent_config excavators --final_agent1_type 2 --final_agent2_type 2 --name "skidsteers-only"

# Train with mixed agents from start (excavator + skidsteer)
python train_mixed_agents.py --agent_config mixed --final_agent1_type 0 --final_agent2_type 2 --name "full-mixed"

# Curriculum: skidsteers → mixed at level 1
python train_mixed_agents.py --agent_config curriculum --initial_agent1_type 2 --initial_agent2_type 2 --final_agent1_type 0 --final_agent2_type 2 --curriculum_switch 1 --name "curriculum"

# Curriculum: skidsteers → mixed at level 2 (later switch)
python train_mixed_agents.py --agent_config curriculum --initial_agent1_type 2 --initial_agent2_type 2 --final_agent1_type 0 --final_agent2_type 2 --curriculum_switch 2 --name "late-switch"

SHARED NETWORK BENEFITS:
========================

The unified network with agent type conditioning can handle:
- Any combination of agent types (0,0), (0,2), (1,2), etc.
- Dynamic switching during training (curriculum)
- Transfer learning between agent types
- Efficient parameter sharing for common skills

"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from utils.models import get_model_ready
from terra.env import TerraEnvBatch
from terra.config import EnvConfig, BatchConfig, Rewards
from flax.training.train_state import TrainState
import optax
import wandb
import eval_ppo
from datetime import datetime
from dataclasses import asdict, dataclass
import time
from tqdm import tqdm
from functools import partial
from flax.jax_utils import replicate, unreplicate
from flax import struct
import utils.helpers as helpers
from utils.utils_ppo import select_action_ppo, wrap_action, obs_to_model_input, policy
import os

# Import the base training infrastructure
from train import get_curriculum_levels, calculate_gae, ppo_update_networks, Transition

jax.config.update("jax_threefry_partitionable", True)

def safe_jax_to_python(value):
    """Safely convert JAX arrays to Python scalars"""
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (ValueError, TypeError):
            # If it's an array with multiple elements, take the first one
            if hasattr(value, 'shape') and value.shape:
                return value.ravel()[0].item()
            else:
                return float(value)
    elif hasattr(value, '__array__'):
        try:
            return float(value)
        except (ValueError, TypeError):
            return str(value)
    else:
        return value

'''
For 8 devices: 
    -2048 envs per device
    -32 steps per env
    -3 epochs per update
    -8 minibatches per update
    -20B total timesteps

For 4 devices: 
    -2048 envs per device
    -32 steps per env
    -3 epochs per update
    -8 minibatches per update
    -5B total timesteps
'''

@dataclass 
class MixedAgentTrainConfig:
    """Configuration for training mixed agent environments"""
    name: str
    num_devices: int = 0
    project: str = "mixed-agents"
    group: str = "tracked-skidsteer"
    num_envs_per_device: int = 2048 #1536  # Increased for better dual skidsteer training 2048
    num_steps: int = 32  # Keep longer rollouts for better temporal learning  32
    update_epochs: int = 5  # Reduced from 4 to 2 for faster training
    num_minibatches: int = 32 #8  # Reduced from 16 to 8 for faster training
    total_timesteps: int = 120_000_000_000  # Increased from 1B to 5B for sufficient training 60_000_000_000
    lr: float = 3e-4    #3e-4
    clip_eps: float = 0.5  # Less conservative clipping for escaping local optima
    gamma: float = 0.9984
    gae_lambda: float = 0.95
    ent_coef: float = 0.09  # 0.1 Higher entropy to escape "do nothing" optima and encourage exploration
    vf_coef: float = 5.0
    max_grad_norm: float = 0.5
    eval_episodes: int = 100
    seed: int = 42
    log_train_interval: int = 1  # Number of updates between logging train stats
    log_eval_interval: int = 50  # Less frequent evaluation for speed
    checkpoint_interval: int = 50  # Less frequent checkpoints for speed
    
    # Model settings optimized for mixed agents
    num_prev_actions = 10
    clip_action_maps = True  # clips the action maps to [-1, 1]
    local_map_normalization_bounds = [-16, 16]
    maps_net_normalization_bounds = [-10, 10]  # Required field for network initialization
    loaded_max = 100
    num_rollouts_eval = 500  # max length of an episode in Terra for eval
    cache_clear_interval = 1000  # Less frequent cache clearing for speed
    
    # Agent type configuration - NEW!
    first_agent1_type: int = 0  # 0=tracked, 1=wheeled, 2=skidsteer (used when no curriculum)
    first_agent2_type: int = 0  # 0=tracked, 1=wheeled, 2=skidsteer (used when no curriculum)
    second_agent1_type: int = 2  # 0=tracked, 1=wheeled, 2=skidsteer (used after first curriculum switch)
    second_agent2_type: int = 2  # 0=tracked, 1=wheeled, 2=skidsteer (used after first curriculum switch)
    
    # Training curriculum for agent types - NEW!
    use_agent_type_curriculum: bool = False  # Enable gradual introduction of agent types
    curriculum_switch_level: int = 1  # Switch agent types when advancing to this curriculum level (default: level 1)
    
    # Second curriculum switch (optional) - NEW!
    use_second_curriculum_switch: bool = False  # Enable second curriculum switch
    second_curriculum_switch_level: int = 2  # Switch to third agent types at this level
    third_agent1_type: int = 0  # Third agent 1 type (after second switch)
    third_agent2_type: int = 2  # Third agent 2 type (after second switch)

    def __post_init__(self):
        self.num_devices = (
            jax.local_device_count() if self.num_devices == 0 else self.num_devices
        )
        self.num_envs = self.num_envs_per_device * self.num_devices
        self.total_timesteps_per_device = self.total_timesteps // self.num_devices
        self.eval_episodes_per_device = self.eval_episodes // self.num_devices
        assert (
            self.num_envs % self.num_devices == 0
        ), "Number of environments must be divisible by the number of devices."
        self.num_updates = (
            self.total_timesteps // (self.num_steps * self.num_envs)
        ) // self.num_devices

        print(f"Devices: {jax.devices()}")
        print(f"Mixed Agent Training - Devices: {self.num_devices}, Updates: {self.num_updates}")
        
        # Agent type curriculum info
        if self.use_agent_type_curriculum:
            type_names = {0: "Tracked", 1: "Wheeled", 2: "SkidSteer"}
            first_str = f"{type_names.get(self.first_agent1_type, 'Unknown')}-{type_names.get(self.first_agent2_type, 'Unknown')}"
            second_str = f"{type_names.get(self.second_agent1_type, 'Unknown')}-{type_names.get(self.second_agent2_type, 'Unknown')}"
            
            if self.use_second_curriculum_switch:
                third_str = f"{type_names.get(self.third_agent1_type, 'Unknown')}-{type_names.get(self.third_agent2_type, 'Unknown')}"
                print(f"🔄 Agent Type Curriculum: {first_str} → {second_str} → {third_str} (levels {self.curriculum_switch_level} → {self.second_curriculum_switch_level})")
            else:
                print(f"🔄 Agent Type Curriculum: {first_str} → {second_str} at level {self.curriculum_switch_level}")
        else:
            type_names = {0: "Tracked", 1: "Wheeled", 2: "SkidSteer"}
            print(f"🤖 Fixed Agent Types: Agent1={type_names.get(self.first_agent1_type, 'Unknown')} ({self.first_agent1_type}), Agent2={type_names.get(self.first_agent2_type, 'Unknown')} ({self.first_agent2_type})")

    # make object subscriptable - required for compatibility with existing code
    def __getitem__(self, key):
        return getattr(self, key)


def create_mixed_agent_env_config(agent_types=(0, 2)):
    """Create environment configuration optimized for mixed agent training"""
    
    # Use the existing dense rewards from config - they already include skid steer rewards
    env_config = EnvConfig()  # This automatically uses Rewards.dense() which includes all our skid steer rewards
    
    # Set the agent types from the training configuration
    env_config = env_config._replace(agent_types=agent_types)
    
    # You can override specific settings if needed for mixed agent training
    # env_config = env_config._replace(
    #     rewards=env_config.rewards._replace(
    #         # Only override if you need mixed-agent specific tuning
    #         ent_coef=0.01  # Example override
    #     )
    # )
    
    return env_config


class ConfigurableAgentManager:
    """Manages agent type configuration with optional curriculum tied to map levels"""
    
    def __init__(self, config: MixedAgentTrainConfig):
        self.config = config
        self.current_timestep = 0
        self.current_level = 0
        
    def get_current_agent_types(self, global_timestep: int = None, curriculum_level: int = None) -> tuple[int, int]:
        """Get the current agent types based on curriculum settings"""
        if global_timestep is not None:
            self.current_timestep = global_timestep
            
        if curriculum_level is not None:
            self.current_level = curriculum_level
            
        if not self.config.use_agent_type_curriculum:
            # Fixed agent types - use first types when no curriculum
            return (self.config.first_agent1_type, self.config.first_agent2_type)
        
        # Curriculum with potential second switch
        if self.current_level < self.config.curriculum_switch_level:
            # First types (before first switch)
            return (self.config.first_agent1_type, self.config.first_agent2_type)
        elif (self.config.use_second_curriculum_switch and 
              self.current_level >= self.config.second_curriculum_switch_level):
            # Third types (after second switch)
            return (self.config.third_agent1_type, self.config.third_agent2_type)
        else:
            # Second types (after first switch, before second switch)
            return (self.config.second_agent1_type, self.config.second_agent2_type)
    
    def get_agent_curriculum_info(self) -> dict:
        """Get information about current agent curriculum state"""
        agent1_type, agent2_type = self.get_current_agent_types()
        type_names = {0: "Tracked", 1: "Wheeled", 2: "SkidSteer"}
        
        info = {
            "agent1_type": agent1_type,
            "agent2_type": agent2_type,
            "agent1_name": type_names.get(agent1_type, "Unknown"),
            "agent2_name": type_names.get(agent2_type, "Unknown"),
            "curriculum_active": self.config.use_agent_type_curriculum,
            "current_timestep": self.current_timestep,
            "current_level": self.current_level,
        }
        
        if self.config.use_agent_type_curriculum:
            info.update({
                "switch_level": self.config.curriculum_switch_level,
                "switched": self.current_level >= self.config.curriculum_switch_level,
                "first_agent1_name": type_names.get(self.config.first_agent1_type, "Unknown"),
                "first_agent2_name": type_names.get(self.config.first_agent2_type, "Unknown"),
            })
            
            if self.config.use_second_curriculum_switch:
                info.update({
                    "second_switch_level": self.config.second_curriculum_switch_level,
                    "second_switched": self.current_level >= self.config.second_curriculum_switch_level,
                    "third_agent1_name": type_names.get(self.config.third_agent1_type, "Unknown"),
                    "third_agent2_name": type_names.get(self.config.third_agent2_type, "Unknown"),
                })
        
        return info








def make_mixed_agent_states(config: MixedAgentTrainConfig, env_params: EnvConfig = None):
    """Initialize states for mixed agent training - compatible with make_states interface"""
    
    # Create agent manager for flexible agent type configuration
    agent_manager = ConfigurableAgentManager(config)
    
    # Create batch config - this determines the agent types used
    batch_cfg = BatchConfig()
    
    # Initialize environment with configurable agents
    env = TerraEnvBatch(batch_cfg=batch_cfg)
    
    # Get environment parameters with agent types from config
    if env_params is None:
        agent_types = agent_manager.get_current_agent_types()
        env_params = create_mixed_agent_env_config(agent_types=agent_types)
    
    num_devices = config.num_devices
    num_envs_per_device = config.num_envs_per_device

    env_params = jax.tree_map(
        lambda x: jnp.array(x)[None, None]
        .repeat(num_devices, 0)
        .repeat(num_envs_per_device, 1),
        env_params,
    )
    
    print(f"Mixed Agent Environment - Tile size shape: {env_params.tile_size.shape}")

    rng = jax.random.PRNGKey(config.seed)
    rng, _rng = jax.random.split(rng)

    # Create the unified network with agent type features
    network, network_params = get_model_ready(_rng, config, env)
    
    # Optimizer with mixed agent considerations
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.lr, eps=1e-5),
    )
    
    train_state = TrainState.create(
        apply_fn=network.apply, params=network_params, tx=tx
    )

    # Get initial agent configuration
    agent_info = agent_manager.get_agent_curriculum_info()
    
    print(f"🤖 Mixed Agent Network: {sum(x.size for x in jax.tree_leaves(network_params)):,} parameters")
    print(f"🔧 Agent Configuration: Agent1={agent_info['agent1_name']} (type {agent_info['agent1_type']}), "
          f"Agent2={agent_info['agent2_name']} (type {agent_info['agent2_type']})")
    print(f"🧠 Network: Unified with agent type conditioning")
    
    if agent_info['curriculum_active']:
        print(f"📈 Agent Type Curriculum: Phase='{'Switched' if agent_info['switched'] else 'Initial'}', Switch at level {agent_info['switch_level']}")
    
    # Note: agent_manager is not stored in train_state since TrainState is frozen
    # and agent_manager is not currently used during training
    
    return rng, env, env_params, train_state


def train_mixed_agents(config: MixedAgentTrainConfig):
    """Main training function for mixed agents - with full feature parity to original train.py"""
    
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
        tags=["mixed-agents", "tracked-excavator", "skid-steer", "unified-network"]
    )
    
    # Log source files - same as original train.py
    train_py_path = os.path.abspath(__file__)
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "terra", "terra", "config.py")
    models_path = os.path.join(os.path.dirname(__file__), "utils", "models.py")
    
    code_artifact = wandb.Artifact(name="mixed_agent_source_code", type="code")
    
    for file_path, name in [(train_py_path, "train_mixed_agents.py"), 
                           (config_path, "config.py"),
                           (models_path, "models.py")]:
        if os.path.exists(file_path):
            code_artifact.add_file(file_path, name=name)
    
    if code_artifact.files:
        run.log_artifact(code_artifact)

    # Initialize training components
    rng, env, env_params, train_state = make_mixed_agent_states(config)

    # Create agent manager for curriculum monitoring
    agent_manager = ConfigurableAgentManager(config)
    
    # Add curriculum monitoring to log agent type changes
    def log_agent_curriculum_status(timestep, update_num):
        """Log agent curriculum status and potential switches"""
        try:
            if not agent_manager.config.use_agent_type_curriculum:
                return
                
            # Get current curriculum level (use mean across environments)
            current_level = jnp.mean(timestep.env_cfg.curriculum.level).item()
            
            # Get target agent types for this level
            target_agent_types = agent_manager.get_current_agent_types(curriculum_level=int(current_level))
            current_agent_types = timestep.env_cfg.agent_types
            
            # Handle batched agent types - average across all environments on first device
            if hasattr(current_agent_types, 'shape') and len(current_agent_types.shape) > 1:
                # If batched, average across all environments on first device
                # Shape is typically (num_devices, num_envs_per_device, num_agents)
                current_agent1_type = safe_jax_to_python(jnp.mean(current_agent_types[0, :, 0]))  # Average across all envs on device 0
                current_agent2_type = safe_jax_to_python(jnp.mean(current_agent_types[0, :, 1]))  # Average across all envs on device 0
            else:
                # If not batched, take directly
                current_agent1_type = safe_jax_to_python(current_agent_types[0])
                current_agent2_type = safe_jax_to_python(current_agent_types[1])
            
            # Log curriculum info - ensure step is always positive and increasing
            if update_num > 0:  # Only log if we have a valid step number
                curriculum_info = {
                    "agent_curriculum/current_level": current_level,
                    "agent_curriculum/target_agent1_type": target_agent_types[0],
                    "agent_curriculum/target_agent2_type": target_agent_types[1],
                    "agent_curriculum/current_agent1_type": current_agent1_type,
                    "agent_curriculum/current_agent2_type": current_agent2_type,
                    "agent_curriculum/should_switch": int(current_level >= agent_manager.config.curriculum_switch_level),
                }
                
                # Add second switch info if enabled
                if agent_manager.config.use_second_curriculum_switch:
                    curriculum_info.update({
                        "agent_curriculum/second_switch_level": agent_manager.config.second_curriculum_switch_level,
                        "agent_curriculum/should_second_switch": int(current_level >= agent_manager.config.second_curriculum_switch_level),
                    })
                
                # Log to wandb
                wandb.log(curriculum_info, step=update_num)
            
            # Print status if switch should happen
            type_names = {0: "Tracked", 1: "Wheeled", 2: "SkidSteer"}
            
            # Check first switch
            if (current_level >= agent_manager.config.curriculum_switch_level and 
                current_level < agent_manager.config.second_curriculum_switch_level and
                (current_agent1_type != target_agent_types[0] or 
                 current_agent2_type != target_agent_types[1])):
                
                current_str = f"{type_names.get(current_agent1_type, 'Unknown')} + {type_names.get(current_agent2_type, 'Unknown')}"
                target_str = f"{type_names.get(target_agent_types[0], 'Unknown')} + {type_names.get(target_agent_types[1], 'Unknown')}"
                
                print(f"⚠️  Agent Type Curriculum: Level {current_level:.1f} reached! Should switch from {current_str} to {target_str}")
                print(f"   Note: Restart training with fixed agent types {target_agent_types} to continue curriculum")
            
            # Check second switch
            elif (agent_manager.config.use_second_curriculum_switch and
                  current_level >= agent_manager.config.second_curriculum_switch_level and
                  (current_agent1_type != target_agent_types[0] or 
                   current_agent2_type != target_agent_types[1])):
                
                current_str = f"{type_names.get(current_agent1_type, 'Unknown')} + {type_names.get(current_agent2_type, 'Unknown')}"
                target_str = f"{type_names.get(target_agent_types[0], 'Unknown')} + {type_names.get(target_agent_types[1], 'Unknown')}"
                
                print(f"⚠️  Agent Type Curriculum: Level {current_level:.1f} reached! Should switch from {current_str} to {target_str}")
                print(f"   Note: Restart training with fixed agent types {target_agent_types} to continue curriculum")
                
        except Exception as e:
            # Log the error but don't crash the training
            print(f"⚠️  Warning: Failed to log agent curriculum status at step {update_num}: {e}")
            # Don't try to log anything else to avoid cascading errors
    
    def log_environment_metrics(timestep, update_num):
        """Log environment metrics for all mixed agent training"""
        try:
            # Extract data from timestep observation
            obs = timestep.observation
            
            # 1. Agent type distribution from env_cfg - average across all environments on first device
            agent_types = timestep.env_cfg.agent_types
            if hasattr(agent_types, 'shape') and len(agent_types.shape) > 1:
                # If batched, average across all environments on first device
                # Shape is typically (num_devices, num_envs_per_device, num_agents)
                agent1_type = safe_jax_to_python(jnp.mean(agent_types[0, :, 0]))  # Average across all envs on device 0
                agent2_type = safe_jax_to_python(jnp.mean(agent_types[0, :, 1]))  # Average across all envs on device 0
            else:
                # If not batched, take directly
                agent1_type = safe_jax_to_python(agent_types[0])
                agent2_type = safe_jax_to_python(agent_types[1])
            
            # Count agent types (this will be the same for all environments in the batch)
            agent1_tracked = int(agent1_type == 0)
            agent1_wheeled = int(agent1_type == 1)
            agent1_skidsteer = int(agent1_type == 2)
            agent2_tracked = int(agent2_type == 0)
            agent2_wheeled = int(agent2_type == 1)
            agent2_skidsteer = int(agent2_type == 2)
            
            # 2. Basic episode metrics
            episode_done = timestep.done
            completion_rate = safe_jax_to_python(jnp.mean(episode_done))
            
            # 3. Calculate completion percentage for each environment
            # Removed completion calculation as it was not working properly
            
            # Log the metrics - ensure step is always positive and increasing
            if update_num > 0:  # Only log if we have a valid step number
                wandb.log({
                    "agent_types/agent1_tracked": agent1_tracked,
                    "agent_types/agent1_wheeled": agent1_wheeled,
                    "agent_types/agent1_skidsteer": agent1_skidsteer,
                    "agent_types/agent2_tracked": agent2_tracked,
                    "agent_types/agent2_wheeled": agent2_wheeled,
                    "agent_types/agent2_skidsteer": agent2_skidsteer,
                    "progress/episode_completion_rate": completion_rate,
                }, step=update_num)
                
        except Exception as e:
            # Log the error but don't crash the training
            print(f"⚠️  Warning: Failed to log environment metrics at step {update_num}: {e}")
            # Optionally log a minimal set of metrics without step to avoid the warning
            try:
                wandb.log({
                    "agent_types/agent1_tracked": 0,
                    "agent_types/agent1_wheeled": 0,
                    "agent_types/agent1_skidsteer": 0,
                    "agent_types/agent2_tracked": 0,
                    "agent_types/agent2_wheeled": 0,
                    "agent_types/agent2_skidsteer": 0,
                    "progress/episode_completion_rate": 0.0,
                })
            except:
                pass  # If even this fails, just continue training
    
    def make_mixed_agent_train(env, env_params, config):
        def train(rng: jax.Array, train_state: TrainState):
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(
                _rng, config.num_envs_per_device * config.num_devices
            )
            reset_rng = reset_rng.reshape(
                (config.num_devices, config.num_envs_per_device, -1)
            )

            # TERRA: Reset envs
            reset_fn_p = jax.pmap(env.reset, axis_name="devices")  # vmapped inside
            timestep = reset_fn_p(env_params, reset_rng)
            prev_actions = jnp.zeros(
                (config.num_devices, config.num_envs_per_device, config.num_prev_actions), dtype=jnp.int32
            )
            prev_reward = jnp.zeros((config.num_devices, config.num_envs_per_device))

            # TRAIN LOOP
            @partial(jax.pmap, axis_name="devices")
            def _update_step(runner_state, _):
                # COLLECT TRAJECTORIES
                def _env_step(runner_state, _):
                    rng, train_state, prev_timestep, prev_actions, prev_reward = runner_state

                    # SELECT ACTION
                    rng, _rng_model, _rng_env = jax.random.split(rng, 3)
                    action, log_prob, value, _ = select_action_ppo(
                        train_state, prev_timestep.observation, prev_actions, _rng_model, config
                    )

                    # STEP ENV
                    _rng_env = jax.random.split(_rng_env, config.num_envs_per_device)
                    action_env = wrap_action(action, env.batch_cfg.action_type)
                    timestep = env.step(prev_timestep, action_env, _rng_env)
                    transition = Transition(
                        done=timestep.done,
                        action=action,
                        value=value,
                        reward=timestep.reward,
                        log_prob=log_prob,
                        obs=prev_timestep.observation,
                        prev_actions=prev_actions,
                        prev_reward=prev_reward,
                    )

                    # UPDATE PREVIOUS ACTIONS
                    prev_actions = jnp.roll(prev_actions, shift=1, axis=-1)
                    prev_actions = prev_actions.at[..., 0].set(action)

                    runner_state = (rng, train_state, timestep, prev_actions, timestep.reward)
                    return runner_state, transition

                # transitions: [seq_len, batch_size, ...]
                runner_state, transitions = jax.lax.scan(
                    _env_step, runner_state, None, config.num_steps
                )

                # CALCULATE ADVANTAGE
                rng, train_state, timestep, prev_actions, prev_reward = runner_state
                rng, _rng = jax.random.split(rng)
                _, _, last_val, _ = select_action_ppo(
                    train_state, timestep.observation, prev_actions, _rng, config
                )
                advantages, targets = calculate_gae(
                    transitions, last_val, config.gamma, config.gae_lambda
                )

                # UPDATE NETWORK
                def _update_epoch(update_state, _):
                    def _update_minbatch(train_state, batch_info):
                        transitions, advantages, targets = batch_info
                        new_train_state, update_info = ppo_update_networks(
                            train_state=train_state,
                            transitions=transitions,
                            advantages=advantages,
                            targets=targets,
                            config=config,
                        )
                        return new_train_state, update_info

                    rng, train_state, transitions, advantages, targets = update_state

                    # MINIBATCHES PREPARATION
                    rng, _rng = jax.random.split(rng)
                    permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                    # [seq_len, batch_size, ...]
                    batch = (transitions, advantages, targets)
                    # [batch_size, seq_len, ...], as our model assumes
                    batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                    shuffled_batch = jtu.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    # [num_minibatches, minibatch_size, seq_len, ...]
                    minibatches = jtu.tree_map(
                        lambda x: jnp.reshape(
                            x, (config.num_minibatches, -1) + x.shape[1:]
                        ),
                        shuffled_batch,
                    )
                    train_state, update_info = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                    )

                    update_state = (rng, train_state, transitions, advantages, targets)
                    return update_state, update_info

                # [seq_len, batch_size, num_layers, hidden_dim]
                update_state = (rng, train_state, transitions, advantages, targets)
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config.update_epochs
                )

                # averaging over minibatches then over epochs
                loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

                rng, train_state = update_state[:2]
                # EVALUATE AGENT
                rng, _rng = jax.random.split(rng)

                runner_state = (rng, train_state, timestep, prev_actions, prev_reward)
                return runner_state, loss_info

            # Setup runner state for multiple devices
            rng, rng_rollout = jax.random.split(rng)
            rng = jax.random.split(rng, num=config.num_devices)
            train_state = replicate(train_state, jax.local_devices()[: config.num_devices])
            runner_state = (rng, train_state, timestep, prev_actions, prev_reward)
            
            for i in tqdm(range(config.num_updates), desc="Training"):
                start_time = time.time()
                runner_state, loss_info = jax.block_until_ready(
                    _update_step(runner_state, None)
                )
                end_time = time.time()

                iteration_duration = end_time - start_time
                iterations_per_second = 1 / iteration_duration
                steps_per_second = (
                    iterations_per_second
                    * config.num_steps
                    * config.num_envs
                    * config.num_devices
                )

                tqdm.write(f"Steps/s: {steps_per_second:.2f}")

                # Use data from the first device for stats and eval
                loss_info_single = unreplicate(loss_info)
                runner_state_single = unreplicate(runner_state)
                _, train_state, timestep, prev_actions = runner_state_single[:4]
                env_params_single = timestep.env_cfg

                if i % config.log_train_interval == 0:
                    # Original logging
                    curriculum_levels = get_curriculum_levels(
                        env_params_single, env.batch_cfg.curriculum_global.levels
                    )
                    wandb.log(
                        {
                            "performance/steps_per_second": steps_per_second,
                            "performance/iterations_per_second": iterations_per_second,
                            "curriculum_levels": curriculum_levels,
                            "lr": config.lr,
                            **loss_info_single,
                        }
                    )
                    
                    # Add our custom logging
                    log_agent_curriculum_status(timestep, i)
                    log_environment_metrics(timestep, i)

                if i % config.checkpoint_interval == 0:
                    checkpoint = {
                        "train_config": config,
                        "env_config": env_params_single,
                        "model": runner_state_single[1].params,
                        "loss_info": loss_info_single,
                    }
                    helpers.save_pkl_object(checkpoint, f"checkpoints/{config.name}.pkl")

                if i % config.log_eval_interval == 0:
                    eval_stats = eval_ppo.rollout(
                        rng_rollout,
                        env,
                        env_params_single,
                        train_state,
                        config,
                    )

                    n = config.num_envs_per_device * eval_stats.length
                    avg_positive_episode_length = jnp.where(
                        eval_stats.positive_terminations > 0,
                        eval_stats.positive_terminations_steps / eval_stats.positive_terminations,
                        jnp.zeros_like(eval_stats.positive_terminations_steps)
                    )
                    loss_info_single.update(
                        {
                            "eval/rewards": eval_stats.reward / n,
                            "eval/max_reward": eval_stats.max_reward,
                            "eval/min_reward": eval_stats.min_reward,
                            "eval/lengths": eval_stats.length,
                            "eval/FORWARD %": eval_stats.action_0 / n,
                            "eval/BACKWARD %": eval_stats.action_1 / n,
                            "eval/CLOCK %": eval_stats.action_2 / n,
                            "eval/ANTICLOCK %": eval_stats.action_3 / n,
                            "eval/CABIN_CLOCK %": eval_stats.action_4 / n,
                            "eval/CABIN_ANTICLOCK %": eval_stats.action_5 / n,
                            "eval/DO": eval_stats.action_6 / n,
                            "eval/positive_terminations": eval_stats.positive_terminations
                            / config.num_envs_per_device,
                            "eval/total_terminations": eval_stats.terminations
                            / config.num_envs_per_device,
                            "eval/avg_positive_episode_length": avg_positive_episode_length
                        }
                    )

                    wandb.log(loss_info_single)

                # Clear JAX caches and run garbage collection to stabilize memory use
                if i % config.cache_clear_interval == 0:
                    jax.clear_caches()
                    import gc
                    gc.collect()

            return {"runner_state": runner_state_single, "loss_info": loss_info_single}

        return train
    
    train_fn = make_mixed_agent_train(env, env_params, config)
    
    def train_with_monitoring(rng, train_state):
        return train_fn(rng, train_state)

    print("🚀 Starting Mixed Agent Training...")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   - Environments per device: {config.num_envs_per_device}")
    print(f"   - Total environments: {config.num_envs}")
    print(f"   - Training steps: {config.num_steps}")
    print(f"   - Total timesteps: {config.total_timesteps:,}")
    print(f"   - Learning rate: {config.lr}")
    agent_types = agent_manager.get_current_agent_types()
    type_names = {0: "Tracked", 1: "Wheeled", 2: "SkidSteer"}
    print(f"   - Agent types: {type_names.get(agent_types[0], 'Unknown')} ({agent_types[0]}) + {type_names.get(agent_types[1], 'Unknown')} ({agent_types[1]})")
    
    # Print reward values from environment configuration
    print(f"\n🎯 Reward Configuration:")
    try:
        # Try to access rewards safely - handle different possible structures
        if hasattr(env_params.rewards, 'shape') and len(env_params.rewards.shape) >= 2:
            rewards = env_params.rewards[0, 0]  # Get rewards from first environment
        else:
            rewards = env_params.rewards  # Direct access if not batched
        
        # Use our safe conversion function
        
        print(f"   - Existence penalty: {safe_jax_to_python(rewards.existence)}")
        print(f"   - Movement penalty: {safe_jax_to_python(rewards.move)}")
        print(f"   - Holding dirt penalty: {safe_jax_to_python(rewards.holding_dirt)}")
        print(f"   - Skid auto-load reward: {safe_jax_to_python(rewards.skid_auto_load)}")
        print(f"   - Skid dump correct reward: {safe_jax_to_python(rewards.skid_dump_correct)}")
        print(f"   - Skid dump wrong penalty: {safe_jax_to_python(rewards.skid_dump_wrong)}")
        print(f"   - Move to dump zone reward: {safe_jax_to_python(rewards.move_to_dump_zone)}")
        print(f"   - Skid lift shovel with dirt: {safe_jax_to_python(rewards.skid_lift_shovel_with_dirt)}")
        print(f"   - Skid move loaded shovel up: {safe_jax_to_python(rewards.skid_move_loaded_shovel_up)}")
        print(f"   - Auto load dump zone penalty: {safe_jax_to_python(rewards.skid_auto_load_from_dumpzone_penalty)}")
        print(f"   - Reward normalizer: {safe_jax_to_python(rewards.normalizer)}")
        
    except Exception as e:
        print(f"   ⚠️  Could not print reward configuration: {e}")
        print(f"   - Rewards structure: {type(env_params.rewards)}")
        if hasattr(env_params.rewards, 'shape'):
            print(f"   - Rewards shape: {env_params.rewards.shape}")
    print("=" * 60)
    
    try:
        t = time.time()
        train_info = jax.block_until_ready(train_with_monitoring(rng, train_state))
        elapsed_time = time.time() - t
        print(f"✅ Mixed agent training completed in {elapsed_time:.2f}s")
        
        # Save final checkpoint with special naming - enhanced metadata
        agent_types = agent_manager.get_current_agent_types()
        type_names = {0: "tracked", 1: "wheeled", 2: "skidsteer"}
        agent_types_str = f"{type_names.get(agent_types[0], 'unknown')}_{type_names.get(agent_types[1], 'unknown')}"
        
        final_checkpoint = {
            "train_config": config,
            "env_config": train_info["runner_state"][2].env_cfg,  # timestep.env_cfg
            "model": train_info["runner_state"][1].params,
            "loss_info": train_info["loss_info"],
            "agent_types": agent_types_str,
            "network_type": "unified_with_agent_type_conditioning",
            "training_duration": elapsed_time,
            "final_reward": train_info.get("final_reward", None)
        }
        helpers.save_pkl_object(final_checkpoint, f"checkpoints/{config.name}_FINAL.pkl")
        print(f"💾 Final mixed agent model saved to checkpoints/{config.name}_FINAL.pkl")
        
    except KeyboardInterrupt:
        print("⏹️ Training interrupted. Finalizing...")
    finally:
        run.finish()
        print("📈 Wandb session finished.")


if __name__ == "__main__":
    DT = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    import argparse

    parser = argparse.ArgumentParser(description="Train mixed agent policies (Tracked + Skid Steer)")
    parser.add_argument(
        "-n", "--name", type=str, default="mixed-agents-skidsteer-skidsteer",
        help="Experiment name"
    )
    parser.add_argument(
        "-m", "--machine", type=str, default="local",
        help="Machine identifier"
    )
    parser.add_argument(
        "-d", "--num_devices", type=int, default=0,
        help="Number of devices to use. If 0, uses all available devices."
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--agent_config", type=str, default="mixed", 
        choices=["excavators", "mixed", "curriculum"],
        help="Agent configuration: 'excavators' (both tracked), 'mixed' (tracked+skidsteer), 'curriculum' (excavators→mixed)"
    )
    parser.add_argument(
        "--curriculum_switch", type=int, default=1,
        help="Curriculum level to switch from initial to final agent types (only for curriculum mode)"
    )
    parser.add_argument(
        "--first_agent1_type", type=int, default=2, choices=[0, 1, 2],
        help="First agent 1 type (used when no curriculum): 0=tracked, 1=wheeled, 2=skidsteer"
    )
    parser.add_argument(
        "--first_agent2_type", type=int, default=2, choices=[0, 1, 2],
        help="First agent 2 type (used when no curriculum): 0=tracked, 1=wheeled, 2=skidsteer"
    )
    parser.add_argument(
        "--second_agent1_type", type=int, default=2, choices=[0, 1, 2],
        help="Second agent 1 type (after first curriculum switch): 0=tracked, 1=wheeled, 2=skidsteer"
    )
    parser.add_argument(
        "--second_agent2_type", type=int, default=2, choices=[0, 1, 2],
        help="Second agent 2 type (after first curriculum switch): 0=tracked, 1=wheeled, 2=skidsteer"
    )
    parser.add_argument(
        "--use_second_curriculum_switch", action="store_true",
        help="Enable second curriculum switch to third agent types"
    )
    parser.add_argument(
        "--second_curriculum_switch", type=int, default=2,
        help="Curriculum level to switch to third agent types (only when use_second_curriculum_switch is enabled)"
    )
    parser.add_argument(
        "--third_agent1_type", type=int, default=0, choices=[0, 1, 2],
        help="Third agent 1 type (after second curriculum switch): 0=tracked, 1=wheeled, 2=skidsteer"
    )
    parser.add_argument(
        "--third_agent2_type", type=int, default=2, choices=[0, 1, 2],
        help="Third agent 2 type (after second curriculum switch): 0=tracked, 1=wheeled, 2=skidsteer"
    )
    
    args, _ = parser.parse_known_args()

    name = f"{args.name}-{args.machine}-{DT}"
    
    # Configure agent types based on preset
    if args.agent_config == "excavators":
        # Both agents are tracked excavators
        first_agent1_type, first_agent2_type = args.first_agent1_type, args.first_agent2_type
        second_agent1_type, second_agent2_type = args.first_agent1_type, args.first_agent2_type  # Same as first for excavators-only
        use_curriculum = False
        type_names = {0: "Tracked", 1: "Wheeled", 2: "SkidSteer"}
        print(f"🏗️  Training Configuration: Both Agents ({type_names.get(first_agent1_type, 'Unknown')} + {type_names.get(first_agent2_type, 'Unknown')})")
        
    elif args.agent_config == "mixed":
        # Mixed agents from start
        first_agent1_type, first_agent2_type = args.first_agent1_type, args.first_agent2_type
        second_agent1_type, second_agent2_type = args.first_agent1_type, args.first_agent2_type  # Same as first for mixed from start
        use_curriculum = False
        type_names = {0: "Tracked", 1: "Wheeled", 2: "SkidSteer"}
        print(f"🔄 Training Configuration: Mixed Agents ({type_names.get(first_agent1_type, 'Unknown')} + {type_names.get(first_agent2_type, 'Unknown')})")
        
    elif args.agent_config == "curriculum":
        # Curriculum: start with first types, switch to second types
        first_agent1_type, first_agent2_type = args.first_agent1_type, args.first_agent2_type
        second_agent1_type, second_agent2_type = args.second_agent1_type, args.second_agent2_type
        use_curriculum = True
        type_names = {0: "Tracked", 1: "Wheeled", 2: "SkidSteer"}
        first_config = f"{type_names.get(first_agent1_type, 'Unknown')} + {type_names.get(first_agent2_type, 'Unknown')}"
        second_config = f"{type_names.get(second_agent1_type, 'Unknown')} + {type_names.get(second_agent2_type, 'Unknown')}"
        
        if args.use_second_curriculum_switch:
            third_config = f"{type_names.get(args.third_agent1_type, 'Unknown')} + {type_names.get(args.third_agent2_type, 'Unknown')}"
            print(f"📈 Training Configuration: Curriculum ({first_config} → {second_config} at level {args.curriculum_switch} → {third_config} at level {args.second_curriculum_switch})")
        else:
            print(f"📈 Training Configuration: Curriculum ({first_config} → {second_config} at level {args.curriculum_switch})")
    
    config = MixedAgentTrainConfig(
        name=name, 
        num_devices=args.num_devices,
        lr=args.lr,
        first_agent1_type=first_agent1_type,
        first_agent2_type=first_agent2_type,
        second_agent1_type=second_agent1_type,
        second_agent2_type=second_agent2_type,
        use_agent_type_curriculum=use_curriculum,
        curriculum_switch_level=args.curriculum_switch,
        use_second_curriculum_switch=args.use_second_curriculum_switch,
        second_curriculum_switch_level=args.second_curriculum_switch,
        third_agent1_type=args.third_agent1_type,
        third_agent2_type=args.third_agent2_type
    )
    
    train_mixed_agents(config) 