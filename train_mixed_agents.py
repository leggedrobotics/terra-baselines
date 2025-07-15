#!/usr/bin/env python3
"""
Training script for mixed agent environments (Tracked Excavators + Skid Steers)
This uses the unified network with agent type conditioning.


TRAINING MODES:
===============

1. EXCAVATORS ONLY (--agent_config excavators):
   - Both agents are tracked excavators (0, 0)
   - Good for initial learning of basic skills
   - Sufficient agent density for alternating training

2. MIXED AGENTS (--agent_config mixed):  
   - Agent 1: Tracked excavator (0)
   - Agent 2: Skid steer (2)
   - Full heterogeneous multi-agent training

3. CURRICULUM (--agent_config curriculum):
   - Start: Both excavators (0, 0) 
   - Switch: Mixed agents (0, 2) at specified timestep
   - Gradual introduction of complexity

USAGE EXAMPLES:
===============

# Train with both excavators initially
python train_mixed_agents.py --agent_config excavators --name "excavators-only"

# Train with mixed agents from start  
python train_mixed_agents.py --agent_config mixed --name "full-mixed"

# Curriculum: excavators → mixed at 10B timesteps
python train_mixed_agents.py --agent_config curriculum --curriculum_switch 10000000000 --name "curriculum"

# Curriculum: excavators → mixed at 5B timesteps (earlier switch)
python train_mixed_agents.py --agent_config curriculum --curriculum_switch 5000000000 --name "early-switch"

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
from train import make_train, get_curriculum_levels, calculate_gae, ppo_update_networks, Transition

jax.config.update("jax_threefry_partitionable", True)

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
    num_envs_per_device: int = 2048  # Increased for better dual skidsteer training 2048
    num_steps: int = 32  # Keep longer rollouts for better temporal learning  32
    update_epochs: int = 3  # Reduced from 4 to 2 for faster training
    num_minibatches: int = 8  # Reduced from 16 to 8 for faster training
    total_timesteps: int = 5_000_000_000  # Increased from 1B to 5B for sufficient training
    lr: float = 3e-4  
    clip_eps: float = 0.2  # Less conservative clipping for escaping local optima
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.06  # 0.1 Higher entropy to escape "do nothing" optima and encourage exploration
    vf_coef: float = 5.0
    max_grad_norm: float = 0.5
    eval_episodes: int = 100
    seed: int = 42
    log_train_interval: int = 3  # Number of updates between logging train stats
    log_eval_interval: int = 100  # Less frequent evaluation for speed
    checkpoint_interval: int = 400  # Less frequent checkpoints for speed
    
    # Model settings optimized for mixed agents
    num_prev_actions = 10
    clip_action_maps = True  # clips the action maps to [-1, 1]
    local_map_normalization_bounds = [-16, 16]
    maps_net_normalization_bounds = [-10, 10]  # Required field for network initialization
    loaded_max = 100
    num_rollouts_eval = 500  # max length of an episode in Terra for eval
    cache_clear_interval = 1000  # Less frequent cache clearing for speed
    
    # Agent type configuration - NEW!
    agent1_type: int = 0  # 0=tracked, 1=wheeled, 2=skidsteer
    agent2_type: int = 2  # 0=tracked, 1=wheeled, 2=skidsteer
    
    # Training curriculum for agent types - NEW!
    use_agent_type_curriculum: bool = False  # Enable gradual introduction of agent types
    curriculum_switch_timestep: int = 10_000_000_000  # When to switch from excavators to mixed

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
            switch_update = self.curriculum_switch_timestep // (self.num_steps * self.num_envs)
            print(f"🔄 Agent Type Curriculum: Excavators (0-{switch_update}) → Mixed ({switch_update}+)")
        else:
            print(f"🤖 Fixed Agent Types: Agent1={self.agent1_type}, Agent2={self.agent2_type}")

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
    """Manages agent type configuration with optional curriculum"""
    
    def __init__(self, config: MixedAgentTrainConfig):
        self.config = config
        self.current_timestep = 0
        
    def get_current_agent_types(self, global_timestep: int = None) -> tuple[int, int]:
        """Get the current agent types based on curriculum settings"""
        if global_timestep is not None:
            self.current_timestep = global_timestep
            
        if not self.config.use_agent_type_curriculum:
            # Fixed agent types
            return (self.config.agent1_type, self.config.agent2_type)
        
        # Curriculum: start with excavators, switch to mixed at threshold
        if self.current_timestep < self.config.curriculum_switch_timestep:
            return (0, 0)  # Both excavators during early training
        else:
            return (self.config.agent1_type, self.config.agent2_type)  # Switch to configured types
    
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
        }
        
        if self.config.use_agent_type_curriculum:
            info["switch_timestep"] = self.config.curriculum_switch_timestep
            info["phase"] = "excavators" if self.current_timestep < self.config.curriculum_switch_timestep else "mixed"
            
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
        print(f"📈 Agent Type Curriculum: Phase='{agent_info['phase']}', Switch at timestep {agent_info['switch_timestep']:,}")
    
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

    # Create agent manager for displaying current configuration
    agent_manager = ConfigurableAgentManager(config)

    # Use the existing train function (it's already generic)
    train_fn = make_train(env, env_params, config)

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
        
        # Extract scalar values from JAX arrays
        def get_scalar(value):
            if hasattr(value, 'item'):
                try:
                    return value.item()
                except ValueError:
                    # If it's an array with multiple elements, take the first one
                    if hasattr(value, 'shape') and value.shape:
                        return value.ravel()[0].item()
                    else:
                        return str(value)
            elif hasattr(value, '__array__'):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return str(value)
            else:
                return value
        
        print(f"   - Existence penalty: {get_scalar(rewards.existence)}")
        print(f"   - Movement penalty: {get_scalar(rewards.move)}")
        print(f"   - Holding dirt penalty: {get_scalar(rewards.holding_dirt)}")
        print(f"   - Skid auto-load reward: {get_scalar(rewards.skid_auto_load)}")
        print(f"   - Skid dump correct reward: {get_scalar(rewards.skid_dump_correct)}")
        print(f"   - Skid dump wrong penalty: {get_scalar(rewards.skid_dump_wrong)}")
        print(f"   - Move to dump zone reward: {get_scalar(rewards.move_to_dump_zone)}")
        print(f"   - Skid lift shovel with dirt: {get_scalar(rewards.skid_lift_shovel_with_dirt)}")
        print(f"   - Skid move loaded shovel up: {get_scalar(rewards.skid_move_loaded_shovel_up)}")
        print(f"   - Auto load dump zone penalty: {get_scalar(rewards.skid_auto_load_from_dumpzone_penalty)}")
        print(f"   - Reward normalizer: {get_scalar(rewards.normalizer)}")
        
    except Exception as e:
        print(f"   ⚠️  Could not print reward configuration: {e}")
        print(f"   - Rewards structure: {type(env_params.rewards)}")
        if hasattr(env_params.rewards, 'shape'):
            print(f"   - Rewards shape: {env_params.rewards.shape}")
    print("=" * 60)
    
    try:
        t = time.time()
        train_info = jax.block_until_ready(train_fn(rng, train_state))
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
        "--curriculum_switch", type=int, default=10_000_000_000,
        help="Timestep to switch from excavators to mixed agents (only for curriculum mode)"
    )
    
    args, _ = parser.parse_known_args()

    name = f"{args.name}-{args.machine}-{DT}"
    
    # Configure agent types based on preset
    if args.agent_config == "excavators":
        # Both agents are tracked excavators
        agent1_type, agent2_type = 0, 0
        use_curriculum = False
        print("🏗️  Training Configuration: Both Excavators (Tracked)")
        
    elif args.agent_config == "mixed":
        
        # Examples: (0, 2) = tracked + skidsteer, (2, 2) = dual skidsteers, (0, 0) = dual tracked
        agent1_type, agent2_type = 2, 2  # ← CHANGE THESE VALUES TO TRAIN DIFFERENT AGENT COMBINATIONS
        use_curriculum = False
        type_names = {0: "Tracked", 1: "Wheeled", 2: "SkidSteer"}
        print(f"🔄 Training Configuration: Mixed Agents ({type_names.get(agent1_type, 'Unknown')} + {type_names.get(agent2_type, 'Unknown')})")
        
    elif args.agent_config == "curriculum":
        # Curriculum: start with excavators, switch to mixed
        # You can change the final target types here too!
        agent1_type, agent2_type = 2, 2  # ← CHANGE THESE VALUES FOR DIFFERENT CURRICULUM TARGET
        use_curriculum = True
        type_names = {0: "Tracked", 1: "Wheeled", 2: "SkidSteer"}
        target_config = f"{type_names.get(agent1_type, 'Unknown')} + {type_names.get(agent2_type, 'Unknown')}"
        print(f"📈 Training Configuration: Curriculum (Excavators → {target_config} at {args.curriculum_switch:,})")
    
    config = MixedAgentTrainConfig(
        name=name, 
        num_devices=args.num_devices,
        lr=args.lr,
        agent1_type=agent1_type,
        agent2_type=agent2_type,
        use_agent_type_curriculum=use_curriculum,
        curriculum_switch_timestep=args.curriculum_switch
    )
    
    train_mixed_agents(config) 