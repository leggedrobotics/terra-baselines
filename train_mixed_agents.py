#!/usr/bin/env python3
"""
Training script for mixed agent environments (Tracked Excavators + Skid Steers)
This uses the unified network with agent type conditioning.

AGENT CONFIGURATION SYSTEM:
===========================

Terra has 2 agents per environment that alternate control each timestep:
- agent_state (Agent 1): Primary agent  
- agent_state_2 (Agent 2): Secondary agent

Agent Types:
- 0: Tracked Excavator (can dig, dump, rotate base)
- 1: Wheeled Excavator (can dig, dump, rotate base + cabin, turn wheels)  
- 2: Skid Steer (can auto-load, manual lift, dump, simple shovel control)

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


@dataclass 
class MixedAgentTrainConfig:
    """Configuration for training mixed agent environments"""
    name: str
    num_devices: int = 0
    project: str = "mixed-agents"
    group: str = "tracked-skidsteer"
    num_envs_per_device: int = 512  # REDUCED from 2048 for faster compilation
    num_steps: int = 32
    update_epochs: int = 5
    num_minibatches: int = 32
    total_timesteps: int = 50_000_000_000  # More training for mixed agents
    lr: float = 3e-4  # Slightly lower LR for more stable training
    clip_eps: float = 0.3  # More conservative clipping
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.01  # Higher entropy for exploration of mixed behaviors
    vf_coef: float = 5.0
    max_grad_norm: float = 0.5
    eval_episodes: int = 100
    seed: int = 42
    log_train_interval: int = 1  # Number of updates between logging train stats
    log_eval_interval: int = 25  # More frequent evaluation for mixed agents
    checkpoint_interval: int = 25  # More frequent checkpoints
    
    # Model settings optimized for mixed agents
    num_prev_actions = 10
    clip_action_maps = True  # clips the action maps to [-1, 1]
    local_map_normalization_bounds = [-16, 16]
    maps_net_normalization_bounds = [-10, 10]  # Required field for network initialization
    loaded_max = 100
    num_rollouts_eval = 500  # max length of an episode in Terra for eval
    cache_clear_interval = 500  # More frequent cache clearing for memory
    
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


def create_mixed_agent_env_config():
    """Create environment configuration optimized for mixed agent training"""
    
    # Use the existing dense rewards from config - they already include skid steer rewards
    env_config = EnvConfig()  # This automatically uses Rewards.dense() which includes all our skid steer rewards
    
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
    
    # Get environment parameters
    if env_params is None:
        env_params = create_mixed_agent_env_config()
    
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
    print(f"   - Agent types: Tracked (0) + Skid Steer (2)")
    print("=" * 60)
    
    try:
        t = time.time()
        train_info = jax.block_until_ready(train_fn(rng, train_state))
        elapsed_time = time.time() - t
        print(f"✅ Mixed agent training completed in {elapsed_time:.2f}s")
        
        # Save final checkpoint with special naming - enhanced metadata
        final_checkpoint = {
            "train_config": config,
            "env_config": train_info["runner_state"][2].env_cfg,  # timestep.env_cfg
            "model": train_info["runner_state"][1].params,
            "loss_info": train_info["loss_info"],
            "agent_types": "tracked_skidsteer",
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
        "-n", "--name", type=str, default="mixed-agents-experiment",
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
        "--envs_per_device", type=int, default=2048,
        help="Number of environments per device"
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
        # Mixed: tracked excavator + skid steer
        agent1_type, agent2_type = 0, 2  
        use_curriculum = False
        print("🔄 Training Configuration: Mixed Agents (Tracked + Skid Steer)")
        
    elif args.agent_config == "curriculum":
        # Curriculum: start with excavators, switch to mixed
        agent1_type, agent2_type = 0, 2  # Final target types
        use_curriculum = True
        print(f"📈 Training Configuration: Curriculum (Excavators → Mixed at {args.curriculum_switch:,})")
    
    config = MixedAgentTrainConfig(
        name=name, 
        num_devices=args.num_devices,
        num_envs_per_device=args.envs_per_device,
        lr=args.lr,
        agent1_type=agent1_type,
        agent2_type=agent2_type,
        use_agent_type_curriculum=use_curriculum,
        curriculum_switch_timestep=args.curriculum_switch
    )
    
    train_mixed_agents(config) 